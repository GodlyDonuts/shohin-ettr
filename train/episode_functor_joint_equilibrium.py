"""Tied joint assignment-semantics equilibrium compiler.

This module replaces the sequential physical-key controller and categorical
revision core with one alternating loop. Every cycle lets the provisional
machine revise physical-key binding and lets that revised binding reconstruct
and revise the machine. Causal cut modes execute identical modules while
removing one or both directed edges.

The module is source-attached and query-blind. None of its recurrent state or
diagnostics may cross the existing hard-machine seal.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from episode_functor_constrained_transport import (
    PRIMARY_ACTIONS,
    PRIMARY_ANSWERS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
    project_key_assignment_logits,
)
from episode_functor_joint_assignment_semantics import (
    JointSemanticCompatibility,
    joint_semantic_compatibility,
)
from episode_functor_machine import (
    MAX_ACTIONS,
    MAX_STATES,
)
from episode_functor_physical_key_nerve import (
    PhysicalKeyNerveResult,
    physical_key_nerve,
)
from episode_functor_witness_compiler import (
    MAX_RECORDS,
    WitnessCompilerBatch,
    WitnessCompilerOutput,
    assemble_relation_evidence,
)


JOINT_EQUILIBRIUM_MODES = frozenset(
    {
        "causal",
        "machine-to-assignment-cut",
        "assignment-to-machine-cut",
        "both-cut",
        "broken-glue",
        "one-step-only",
        "sign-reversed",
        "open-loop",
    }
)
MACHINE_ROWS = (
    PRIMARY_ACTIONS * PRIMARY_STATES
    + PRIMARY_OBSERVERS * PRIMARY_STATES
)
MACHINE_CATEGORIES = PRIMARY_STATES
TRANSITION_ROWS = PRIMARY_ACTIONS * PRIMARY_STATES
ACTIVE_SLOTS = (
    tuple(range(PRIMARY_STATES))
    + tuple(MAX_STATES + index for index in range(PRIMARY_ACTIONS))
    + tuple(
        MAX_STATES + MAX_ACTIONS + index
        for index in range(PRIMARY_OBSERVERS)
    )
)


class JointEquilibriumError(ValueError):
    """Joint equilibrium geometry, values, or control mode failed closed."""


@dataclass(frozen=True, slots=True)
class JointEquilibriumResult:
    """Attached recurrent diagnostics; only final tensors may be sealed."""

    raw_key_assignment_logits: torch.Tensor
    key_assignment_logits: torch.Tensor
    transition_probabilities: torch.Tensor
    observer_probabilities: torch.Tensor
    cycle_key_assignment_logits: tuple[torch.Tensor, ...]
    cycle_transition_probabilities: tuple[torch.Tensor, ...]
    cycle_observer_probabilities: tuple[torch.Tensor, ...]
    cycle_assignment_correction: tuple[torch.Tensor, ...]
    cycle_machine_direction: tuple[torch.Tensor, ...]
    cycle_machine_step: tuple[torch.Tensor, ...]
    cycle_semantic_compatibility: tuple[JointSemanticCompatibility, ...]
    cycle_nerve: tuple[PhysicalKeyNerveResult, ...]
    mode: str

    def __post_init__(self) -> None:
        batch = int(self.raw_key_assignment_logits.shape[0])
        cycles = len(self.cycle_key_assignment_logits)
        if (
            self.raw_key_assignment_logits.ndim != 3
            or self.key_assignment_logits.shape
            != self.raw_key_assignment_logits.shape
            or self.transition_probabilities.shape
            != (
                batch,
                PRIMARY_ACTIONS,
                PRIMARY_STATES,
                PRIMARY_STATES,
            )
            or self.observer_probabilities.shape
            != (
                batch,
                PRIMARY_OBSERVERS,
                PRIMARY_STATES,
                PRIMARY_ANSWERS,
            )
            or cycles < 1
            or len(self.cycle_transition_probabilities) != cycles
            or len(self.cycle_observer_probabilities) != cycles
            or len(self.cycle_assignment_correction) != cycles
            or len(self.cycle_machine_direction) != cycles
            or len(self.cycle_machine_step) != cycles
            or len(self.cycle_semantic_compatibility) != cycles
            or len(self.cycle_nerve) != cycles
            or self.mode not in JOINT_EQUILIBRIUM_MODES
        ):
            raise JointEquilibriumError(
                "joint equilibrium result geometry differs"
            )
        values = (
            self.raw_key_assignment_logits,
            self.key_assignment_logits,
            self.transition_probabilities,
            self.observer_probabilities,
            *self.cycle_key_assignment_logits,
            *self.cycle_transition_probabilities,
            *self.cycle_observer_probabilities,
            *self.cycle_assignment_correction,
            *self.cycle_machine_direction,
            *self.cycle_machine_step,
        )
        if any(
            not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
            for value in values
        ):
            raise JointEquilibriumError(
                "joint equilibrium result values differ"
            )


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int,
    keepdim: bool,
) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum(dim, keepdim=keepdim) / weights.sum(
        dim,
        keepdim=keepdim,
    ).clamp_min(1.0)


def _normalize(values: torch.Tensor) -> torch.Tensor:
    total = values.sum(-1, keepdim=True)
    normalized = values / total.clamp_min(
        torch.finfo(values.dtype).tiny
    )
    return torch.where(total.gt(0), normalized, torch.zeros_like(values))


def _support(device: torch.device) -> torch.Tensor:
    support = torch.ones(
        MACHINE_ROWS,
        MACHINE_CATEGORIES,
        dtype=torch.bool,
        device=device,
    )
    support[TRANSITION_ROWS:, PRIMARY_ANSWERS:] = False
    return support


def _rows_from_machine(
    transition: torch.Tensor,
    observer: torch.Tensor,
) -> torch.Tensor:
    batch = int(transition.shape[0])
    return torch.cat(
        (
            transition.reshape(
                batch,
                TRANSITION_ROWS,
                PRIMARY_STATES,
            ),
            F.pad(
                observer.reshape(
                    batch,
                    PRIMARY_OBSERVERS * PRIMARY_STATES,
                    PRIMARY_ANSWERS,
                ),
                (0, MACHINE_CATEGORIES - PRIMARY_ANSWERS),
            ),
        ),
        dim=1,
    )


def _machine_from_rows(
    rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = int(rows.shape[0])
    return (
        rows[:, :TRANSITION_ROWS].reshape(
            batch,
            PRIMARY_ACTIONS,
            PRIMARY_STATES,
            PRIMARY_STATES,
        ),
        rows[
            :,
            TRANSITION_ROWS:,
            :PRIMARY_ANSWERS,
        ].reshape(
            batch,
            PRIMARY_OBSERVERS,
            PRIMARY_STATES,
            PRIMARY_ANSWERS,
        ),
    )


class ContractiveTiedCell(nn.Module):
    """Dense input map with an exactly bounded diagonal recurrent Jacobian."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = int(width)
        self.input = nn.Linear(width, width)
        self.recurrent_diagonal = nn.Parameter(torch.zeros(width))

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        if inputs.shape != hidden.shape or inputs.shape[-1] != self.width:
            raise JointEquilibriumError(
                "contractive tied-cell geometry differs"
            )
        recurrent = (
            0.9
            * torch.tanh(self.recurrent_diagonal)
            * hidden
        )
        candidate = torch.tanh(self.input(inputs) + recurrent)
        return 0.75 * hidden + 0.25 * candidate


class JointAssignmentSemanticsEquilibrium(nn.Module):
    """Four-cycle tied assignment-machine equilibrium controller."""

    assignment_feature_width = 27
    machine_feature_width = 16

    def __init__(
        self,
        *,
        assignment_width: int = 600,
        assignment_context_width: int = 1200,
        machine_width: int = 960,
        machine_context_width: int = 1920,
        cycles: int = 4,
        sinkhorn_iterations: int = 64,
        max_assignment_correction: float = 2.0,
        max_machine_step: float = 0.1,
    ) -> None:
        super().__init__()
        if (
            assignment_width < 32
            or assignment_context_width < 64
            or machine_width < 64
            or machine_context_width < 128
            or not 1 <= cycles <= 8
            or sinkhorn_iterations < 8
            or not math.isfinite(max_assignment_correction)
            or not 0.0 < max_assignment_correction <= 8.0
            or not math.isfinite(max_machine_step)
            or not 0.0 < max_machine_step <= 0.5
        ):
            raise JointEquilibriumError(
                "joint equilibrium configuration differs"
            )
        self.assignment_width = int(assignment_width)
        self.assignment_context_width = int(assignment_context_width)
        self.machine_width = int(machine_width)
        self.machine_context_width = int(machine_context_width)
        self.cycles = int(cycles)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.max_assignment_correction = float(
            max_assignment_correction
        )
        self.max_machine_step = float(max_machine_step)

        self.assignment_encoder = nn.Sequential(
            nn.Linear(
                self.assignment_feature_width,
                self.assignment_width,
            ),
            nn.SiLU(),
            nn.Linear(self.assignment_width, self.assignment_width),
        )
        self.assignment_mixer = nn.Sequential(
            nn.Linear(
                4 * self.assignment_width,
                self.assignment_context_width,
            ),
            nn.SiLU(),
            nn.Linear(
                self.assignment_context_width,
                self.assignment_width,
            ),
        )
        self.assignment_direction = nn.Linear(
            self.assignment_width,
            1,
        )
        self.assignment_gate = nn.Linear(self.assignment_width, 1)

        self.machine_encoder = nn.Sequential(
            nn.Linear(self.machine_feature_width, self.machine_width),
            nn.SiLU(),
            nn.Linear(self.machine_width, self.machine_width),
        )
        self.record_encoder = nn.Sequential(
            nn.Linear(32, self.machine_width),
            nn.SiLU(),
            nn.Linear(self.machine_width, self.machine_width),
        )
        self.machine_mixer = nn.Sequential(
            nn.Linear(
                5 * self.machine_width,
                self.machine_context_width,
            ),
            nn.SiLU(),
            nn.Linear(self.machine_context_width, self.machine_width),
        )
        self.contractive_cell = ContractiveTiedCell(self.machine_width)
        self.machine_direction = nn.Sequential(
            nn.Linear(self.machine_width, self.machine_width),
            nn.SiLU(),
            nn.Linear(self.machine_width, 1),
        )
        step_width = max(64, self.machine_width // 4)
        self.machine_step = nn.Sequential(
            nn.Linear(self.machine_width, step_width),
            nn.SiLU(),
            nn.Linear(step_width, 1),
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _type_ranges() -> tuple[tuple[int, int], ...]:
        return (
            (0, PRIMARY_STATES),
            (
                PRIMARY_STATES,
                PRIMARY_STATES + PRIMARY_ACTIONS,
            ),
            (
                PRIMARY_STATES + PRIMARY_ACTIONS,
                PRIMARY_STATES + PRIMARY_ACTIONS + PRIMARY_OBSERVERS,
            ),
        )

    @classmethod
    def _type_assignment_context(
        cls,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden)
        for start, end in cls._type_ranges():
            output[:, start:end] = hidden[:, start:end].mean(
                1,
                keepdim=True,
            )
        return output

    @classmethod
    def _type_key_context(
        cls,
        hidden: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(hidden)
        key_mask = valid[:, None, :, None]
        for start, end in cls._type_ranges():
            pooled = _masked_mean(
                hidden[:, start:end],
                key_mask,
                dim=2,
                keepdim=True,
            ).mean(1, keepdim=True)
            output[:, start:end] = pooled
        return output

    @classmethod
    def _type_column_mean(
        cls,
        values: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(values)
        for start, end in cls._type_ranges():
            output[:, start:end] = values[:, start:end].mean(
                1,
                keepdim=True,
            )
        return output

    @staticmethod
    def _action_channel(
        template: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros_like(template)
        output[
            :,
            PRIMARY_STATES : PRIMARY_STATES + PRIMARY_ACTIONS,
        ] = values
        return output

    def _assignment_features(
        self,
        witness: WitnessCompilerOutput,
        nerve: PhysicalKeyNerveResult,
        semantics: JointSemanticCompatibility,
        raw_logits: torch.Tensor,
        projected_logits: torch.Tensor,
        *,
        machine_to_assignment_cut: bool,
    ) -> torch.Tensor:
        active = torch.tensor(
            ACTIVE_SLOTS,
            dtype=torch.long,
            device=raw_logits.device,
        )
        logits = raw_logits.index_select(1, active).float()
        projected = projected_logits.index_select(1, active).float()
        valid = witness.unique_key_valid
        mask = valid[:, None]
        probabilities = projected.softmax(-1)
        centered_logits = logits - _masked_mean(
            logits,
            mask,
            dim=-1,
            keepdim=True,
        )
        centered_logits = centered_logits.masked_fill(~mask, 0.0)
        path_compatibility = torch.cat(
            (
                nerve.state_compatibility,
                nerve.action_compatibility,
                nerve.observer_compatibility,
            ),
            dim=1,
        )
        semantic = semantics.assignment_compatibility
        if machine_to_assignment_cut:
            semantic = path_compatibility + 0.0 * (
                semantic - path_compatibility
            )
        row_mean = _masked_mean(
            path_compatibility,
            mask,
            dim=-1,
            keepdim=True,
        ).expand_as(path_compatibility)
        column_mean = self._type_column_mean(path_compatibility)
        semantic_center = semantic - _masked_mean(
            semantic,
            mask,
            dim=-1,
            keepdim=True,
        )
        tiny = torch.finfo(probabilities.dtype).tiny
        entropy = -(
            probabilities
            * probabilities.clamp_min(tiny).log()
        ).sum(-1, keepdim=True).expand_as(probabilities)
        transition = nerve.transition_relation
        observation = nerve.observation_relation
        key_statistics = (
            _normalize(transition.sum((2, 3))),
            _normalize(transition.sum((1, 2))),
            _normalize(transition.sum((1, 3))),
            _normalize(observation.sum((2, 3))),
            _normalize(observation.sum((1, 3))),
        )
        expanded_key_statistics = tuple(
            value[:, None].expand_as(probabilities)
            for value in key_statistics
        )
        action_channels = tuple(
            self._action_channel(probabilities, value)
            for value in (
                nerve.action_left_compatibility,
                nerve.action_right_compatibility,
                nerve.action_observer_compatibility,
                nerve.action_commutator_compatibility,
            )
        )
        type_indicators = [
            torch.zeros_like(probabilities)
            for _ in range(3)
        ]
        for index, (start, end) in enumerate(self._type_ranges()):
            type_indicators[index][:, start:end] = 1.0
        path_scale = (
            nerve.path_mass / (1.0 + nerve.path_mass)
        )[:, None, None].expand_as(probabilities)
        features = torch.stack(
            (
                probabilities,
                centered_logits,
                path_compatibility,
                path_compatibility - row_mean,
                path_compatibility - column_mean,
                entropy,
                mask.to(probabilities.dtype).expand_as(probabilities),
                path_scale,
                *expanded_key_statistics,
                *action_channels,
                *(value.abs() for value in action_channels),
                *type_indicators,
                semantic,
                semantic_center,
                semantic_center.abs(),
            ),
            dim=-1,
        )
        if features.shape != logits.shape + (
            self.assignment_feature_width,
        ):
            raise JointEquilibriumError(
                "joint assignment features differ"
            )
        return features

    def _revise_assignment(
        self,
        witness: WitnessCompilerOutput,
        features: torch.Tensor,
        raw_logits: torch.Tensor,
        *,
        zero_update: bool,
        reverse_update: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.assignment_encoder(features)
        valid = witness.unique_key_valid
        slot_context = _masked_mean(
            hidden,
            valid[:, None, :, None],
            dim=2,
            keepdim=True,
        ).expand_as(hidden)
        type_context = self._type_assignment_context(hidden)
        type_key_context = self._type_key_context(hidden, valid)
        mixed = self.assignment_mixer(
            torch.cat(
                (
                    hidden,
                    slot_context,
                    type_context,
                    type_key_context,
                ),
                dim=-1,
            )
        )
        correction = (
            self.max_assignment_correction
            * torch.sigmoid(self.assignment_gate(mixed).squeeze(-1))
            * torch.tanh(
                self.assignment_direction(mixed).squeeze(-1)
            )
        )
        valid_mask = valid[:, None].to(correction.dtype)
        correction = correction * valid_mask
        correction = correction - _masked_mean(
            correction,
            valid[:, None],
            dim=-1,
            keepdim=True,
        )
        correction = correction * valid_mask
        if zero_update:
            correction = correction * 0.0
        elif reverse_update:
            correction = -correction
        full_correction = torch.zeros_like(raw_logits)
        active = torch.tensor(
            ACTIVE_SLOTS,
            dtype=torch.long,
            device=raw_logits.device,
        )
        full_correction[:, active] = correction
        revised_raw = raw_logits.float() + full_correction
        revised_raw = revised_raw.masked_fill(
            ~valid[:, None],
            torch.finfo(revised_raw.dtype).min,
        )
        projected = project_key_assignment_logits(
            slot_assignment_logits=revised_raw,
            source_unique_key_valid=valid,
            sinkhorn_iterations=self.sinkhorn_iterations,
            straight_through=False,
        )
        return revised_raw, projected, full_correction

    @staticmethod
    def _machine_type_context(hidden: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(hidden)
        output[:, :TRANSITION_ROWS] = hidden[
            :,
            :TRANSITION_ROWS,
        ].mean((1, 2), keepdim=True)
        output[:, TRANSITION_ROWS:] = hidden[
            :,
            TRANSITION_ROWS:,
            :PRIMARY_ANSWERS,
        ].mean((1, 2), keepdim=True)
        return output

    def _machine_features(
        self,
        current: torch.Tensor,
        direct: torch.Tensor,
        previous_direction: torch.Tensor,
        assignment: torch.Tensor,
        semantic: JointSemanticCompatibility,
        support: torch.Tensor,
        *,
        cycle: int,
    ) -> torch.Tensor:
        mask = support[None]
        tiny = torch.finfo(current.dtype).tiny
        log_probability = current.clamp_min(tiny).log()
        gauge = log_probability - _masked_mean(
            log_probability,
            mask,
            dim=-1,
            keepdim=True,
        )
        current_entropy = -(
            current * current.clamp_min(tiny).log()
        ).sum(-1, keepdim=True).expand_as(current)
        direct_entropy = -(
            direct * direct.clamp_min(tiny).log()
        ).sum(-1, keepdim=True).expand_as(direct)
        current_max = current.amax(-1, keepdim=True).expand_as(current)
        direct_max = direct.amax(-1, keepdim=True).expand_as(direct)
        transition_type = torch.zeros_like(current)
        observer_type = torch.zeros_like(current)
        transition_type[:, :TRANSITION_ROWS] = 1.0
        observer_type[:, TRANSITION_ROWS:] = 1.0
        cycle_feature = torch.full_like(
            current,
            float(cycle + 1) / float(self.cycles),
        )
        active = torch.tensor(
            ACTIVE_SLOTS,
            dtype=torch.long,
            device=assignment.device,
        )
        assignment_probability = assignment.index_select(
            1,
            active,
        ).softmax(-1)
        assignment_confidence = assignment_probability.amax(-1).mean(
            -1,
            keepdim=True,
        )[:, None].expand_as(current)
        semantic_agreement = semantic.assignment_compatibility.amax(
            -1
        ).mean(-1, keepdim=True)[:, None].expand_as(current)
        residual = direct - current
        return torch.stack(
            (
                current,
                gauge,
                direct,
                residual,
                residual.abs(),
                current_entropy,
                direct_entropy,
                current_max,
                direct_max,
                mask.to(current.dtype).expand_as(current),
                transition_type,
                observer_type,
                cycle_feature,
                previous_direction,
                assignment_confidence,
                semantic_agreement,
            ),
            dim=-1,
        )

    def _revise_machine(
        self,
        current: torch.Tensor,
        direct: torch.Tensor,
        features: torch.Tensor,
        record_features: torch.Tensor,
        record_valid: torch.Tensor,
        hidden: torch.Tensor,
        support: torch.Tensor,
        *,
        zero_update: bool,
        reverse_update: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.machine_encoder(features)
        mask = support[None, :, :, None]
        row_context = _masked_mean(
            encoded,
            mask,
            dim=2,
            keepdim=True,
        ).expand_as(encoded)
        type_context = self._machine_type_context(encoded)
        global_context = _masked_mean(
            encoded,
            mask,
            dim=2,
            keepdim=False,
        ).mean(1, keepdim=True)[:, :, None].expand_as(encoded)
        record_encoded = self.record_encoder(record_features)
        record_context = _masked_mean(
            record_encoded,
            record_valid[..., None],
            dim=1,
            keepdim=True,
        )[:, :, None].expand_as(encoded)
        mixed = self.machine_mixer(
            torch.cat(
                (
                    encoded,
                    row_context,
                    type_context,
                    global_context,
                    record_context,
                ),
                dim=-1,
            )
        )
        hidden = self.contractive_cell(mixed, hidden)
        direction = torch.tanh(
            self.machine_direction(hidden).squeeze(-1)
        )
        direction = direction * support[None].to(direction.dtype)
        direction = direction - _masked_mean(
            direction,
            support[None],
            dim=-1,
            keepdim=True,
        )
        direction = direction * support[None].to(direction.dtype)
        step = self.max_machine_step * torch.sigmoid(
            self.machine_step(hidden).squeeze(-1)
        )
        step = step * support[None].to(step.dtype)
        if zero_update:
            direction = direction * 0.0
        elif reverse_update:
            direction = -direction
        negative = torch.finfo(current.dtype).min
        proposal = (
            current.clamp_min(torch.finfo(current.dtype).tiny).log()
            + step * direction
        ).masked_fill(~support[None], negative).softmax(-1)
        revised = 0.5 * current + 0.5 * proposal
        revised = revised * support[None].to(revised.dtype)
        revised = revised / revised.sum(-1, keepdim=True).clamp_min(
            torch.finfo(revised.dtype).tiny
        )
        return revised, hidden, direction, step

    @staticmethod
    def _relation_evidence(
        batch: WitnessCompilerBatch,
        witness: WitnessCompilerOutput,
        assignment: torch.Tensor,
    ):
        return assemble_relation_evidence(
            record_type_logits=witness.record_type_logits,
            occurrence_role_logits=witness.occurrence_role_logits,
            answer_logits=witness.answer_logits,
            occurrence_valid=batch.pointer.occurrence_valid,
            occurrence_to_record=batch.occurrence_to_record,
            occurrence_to_unique=batch.pointer.occurrence_to_unique,
            source_unique_key_valid=witness.unique_key_valid,
            key_assignment_logits=assignment,
        )

    def forward(
        self,
        batch: WitnessCompilerBatch,
        witness: WitnessCompilerOutput,
        *,
        record_features: torch.Tensor,
        mode: str = "causal",
    ) -> JointEquilibriumResult:
        if mode not in JOINT_EQUILIBRIUM_MODES:
            raise JointEquilibriumError(
                f"unknown joint equilibrium mode: {mode}"
            )
        if (
            record_features.shape
            != (batch.batch_size, MAX_RECORDS, 32)
            or record_features.device
            != witness.raw_key_assignment_logits.device
            or not record_features.is_floating_point()
            or not bool(torch.isfinite(record_features).all())
        ):
            raise JointEquilibriumError(
                "joint equilibrium record features differ"
            )
        raw = witness.raw_key_assignment_logits.float()
        projected = project_key_assignment_logits(
            slot_assignment_logits=raw,
            source_unique_key_valid=witness.unique_key_valid,
            sinkhorn_iterations=self.sinkhorn_iterations,
            straight_through=False,
        )
        initial_projected = projected
        transition = witness.projection.transition_transport.float()
        observer = witness.projection.observer_transport.float()
        current_rows = _rows_from_machine(transition, observer)
        support = _support(current_rows.device)
        current_rows = current_rows * support[None].to(current_rows.dtype)
        hidden = torch.zeros(
            current_rows.shape
            + (self.machine_width,),
            dtype=current_rows.dtype,
            device=current_rows.device,
        )
        previous_direction = torch.zeros_like(current_rows)
        current_witness = replace(
            witness,
            key_assignment_logits=projected,
            raw_key_assignment_logits=raw,
        )
        key_cycles: list[torch.Tensor] = []
        transition_cycles: list[torch.Tensor] = []
        observer_cycles: list[torch.Tensor] = []
        assignment_corrections: list[torch.Tensor] = []
        machine_directions: list[torch.Tensor] = []
        machine_steps: list[torch.Tensor] = []
        semantic_cycles: list[JointSemanticCompatibility] = []
        nerve_cycles: list[PhysicalKeyNerveResult] = []
        m_to_a_cut = mode in {
            "machine-to-assignment-cut",
            "both-cut",
        }
        a_to_m_cut = mode in {
            "assignment-to-machine-cut",
            "both-cut",
        }
        zero_update = mode == "open-loop"
        reverse_update = mode == "sign-reversed"
        nerve_mode = (
            mode
            if mode in {"broken-glue", "one-step-only"}
            else "causal"
        )
        semantic_mode = (
            "one-step-only"
            if mode == "one-step-only"
            else "causal"
        )

        for cycle in range(self.cycles):
            transition, observer = _machine_from_rows(current_rows)
            nerve = physical_key_nerve(
                current_witness,
                batch.record_valid,
                mode=nerve_mode,
                key_assignment_logits=projected,
            )
            semantics = joint_semantic_compatibility(
                nerve,
                transition,
                observer,
                witness.unique_key_valid,
                mode=semantic_mode,
            )
            reported_semantics = semantics
            if m_to_a_cut:
                reported_semantics = joint_semantic_compatibility(
                    nerve,
                    transition,
                    observer,
                    witness.unique_key_valid,
                    mode="machine-to-assignment-cut",
                )
            machine_semantics = semantics
            if a_to_m_cut:
                fixed_nerve = physical_key_nerve(
                    current_witness,
                    batch.record_valid,
                    mode=nerve_mode,
                    key_assignment_logits=initial_projected,
                )
                fixed_semantics = joint_semantic_compatibility(
                    fixed_nerve,
                    transition,
                    observer,
                    witness.unique_key_valid,
                    mode=semantic_mode,
                )
                machine_semantics = replace(
                    fixed_semantics,
                    state_compatibility=(
                        fixed_semantics.state_compatibility
                        + 0.0
                        * (
                            semantics.state_compatibility
                            - fixed_semantics.state_compatibility
                        )
                    ),
                    action_compatibility=(
                        fixed_semantics.action_compatibility
                        + 0.0
                        * (
                            semantics.action_compatibility
                            - fixed_semantics.action_compatibility
                        )
                    ),
                    observer_compatibility=(
                        fixed_semantics.observer_compatibility
                        + 0.0
                        * (
                            semantics.observer_compatibility
                            - fixed_semantics.observer_compatibility
                        )
                    ),
                    assignment_compatibility=(
                        fixed_semantics.assignment_compatibility
                        + 0.0
                        * (
                            semantics.assignment_compatibility
                            - fixed_semantics.assignment_compatibility
                        )
                    ),
                )
            assignment_features = self._assignment_features(
                current_witness,
                nerve,
                semantics,
                raw,
                projected,
                machine_to_assignment_cut=m_to_a_cut,
            )
            raw, projected, assignment_correction = (
                self._revise_assignment(
                    current_witness,
                    assignment_features,
                    raw,
                    zero_update=zero_update,
                    reverse_update=reverse_update,
                )
            )
            machine_assignment = projected
            if a_to_m_cut:
                machine_assignment = initial_projected + 0.0 * (
                    projected - initial_projected
                )
            evidence = self._relation_evidence(
                batch,
                current_witness,
                machine_assignment,
            )
            direct_rows = _rows_from_machine(
                evidence.transition_logits.float().softmax(-1),
                evidence.observer_logits.float().softmax(-1),
            )
            direct_rows = (
                direct_rows * support[None].to(direct_rows.dtype)
            )
            machine_features = self._machine_features(
                current_rows,
                direct_rows,
                previous_direction,
                machine_assignment,
                machine_semantics,
                support,
                cycle=cycle,
            )
            current_rows, hidden, direction, step = self._revise_machine(
                current_rows,
                direct_rows,
                machine_features,
                record_features,
                batch.record_valid,
                hidden,
                support,
                zero_update=zero_update,
                reverse_update=reverse_update,
            )
            previous_direction = direction
            transition, observer = _machine_from_rows(current_rows)
            current_witness = replace(
                current_witness,
                relation_evidence=evidence,
                key_assignment_logits=projected,
                raw_key_assignment_logits=raw,
            )
            key_cycles.append(projected)
            transition_cycles.append(transition)
            observer_cycles.append(observer)
            assignment_corrections.append(assignment_correction)
            machine_directions.append(direction)
            machine_steps.append(step)
            semantic_cycles.append(reported_semantics)
            nerve_cycles.append(nerve)

        return JointEquilibriumResult(
            raw_key_assignment_logits=raw,
            key_assignment_logits=projected,
            transition_probabilities=transition,
            observer_probabilities=observer,
            cycle_key_assignment_logits=tuple(key_cycles),
            cycle_transition_probabilities=tuple(transition_cycles),
            cycle_observer_probabilities=tuple(observer_cycles),
            cycle_assignment_correction=tuple(assignment_corrections),
            cycle_machine_direction=tuple(machine_directions),
            cycle_machine_step=tuple(machine_steps),
            cycle_semantic_compatibility=tuple(semantic_cycles),
            cycle_nerve=tuple(nerve_cycles),
            mode=mode,
        )


__all__ = [
    "ACTIVE_SLOTS",
    "JOINT_EQUILIBRIUM_MODES",
    "JointAssignmentSemanticsEquilibrium",
    "JointEquilibriumError",
    "JointEquilibriumResult",
]
