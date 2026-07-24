"""Learned physical-key path synchronization for EFC slot binding.

The controller consumes only permutation-equivariant physical-key relations
and revises slot-to-key assignment logits. It does not emit machine cells,
execute queries, retain source state after sealing, or invoke a host solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn

from episode_functor_constrained_transport import (
    PRIMARY_ACTIONS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
)
from episode_functor_machine import MAX_ACTIONS, MAX_STATES
from episode_functor_physical_key_nerve import (
    PhysicalKeyNerveResult,
    physical_key_nerve,
)
from episode_functor_witness_compiler import WitnessCompilerOutput


PATH_CONTROLLER_MODES = frozenset(
    {
        "causal",
        "broken-glue",
        "one-step-only",
        "open-loop",
        "sign-reversed",
    }
)


class PhysicalKeyPathControllerError(ValueError):
    """Physical-key controller geometry, values, or mode failed closed."""


@dataclass(frozen=True, slots=True)
class PhysicalKeyPathControllerResult:
    raw_key_assignment_logits: torch.Tensor
    correction: torch.Tensor
    gate: torch.Tensor
    nerve: PhysicalKeyNerveResult
    mode: str

    def __post_init__(self) -> None:
        if (
            self.raw_key_assignment_logits.ndim != 3
            or self.correction.shape != self.raw_key_assignment_logits.shape
            or self.gate.shape != self.raw_key_assignment_logits.shape
            or self.mode not in PATH_CONTROLLER_MODES
            or not bool(torch.isfinite(self.raw_key_assignment_logits).all())
            or not bool(torch.isfinite(self.correction).all())
            or not bool(torch.isfinite(self.gate).all())
        ):
            raise PhysicalKeyPathControllerError(
                "physical-key controller result differs"
            )


def _normalize(values: torch.Tensor) -> torch.Tensor:
    total = values.sum(-1, keepdim=True)
    normalized = values / total.clamp_min(
        torch.finfo(values.dtype).tiny
    )
    return torch.where(total.gt(0), normalized, torch.zeros_like(values))


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int,
    keepdim: bool,
) -> torch.Tensor:
    weights = mask.to(values.dtype)
    numerator = (values * weights).sum(dim, keepdim=keepdim)
    denominator = weights.sum(dim, keepdim=keepdim).clamp_min(1.0)
    return numerator / denominator


class PhysicalKeyPathController(nn.Module):
    """Revise anonymous slot binding from exact physical-key path structure."""

    feature_width = 24

    def __init__(
        self,
        *,
        cell_width: int = 600,
        context_width: int = 1200,
        max_correction: float = 2.0,
    ) -> None:
        super().__init__()
        if (
            cell_width < 32
            or context_width < 64
            or not math.isfinite(max_correction)
            or not 0.0 < max_correction <= 8.0
        ):
            raise PhysicalKeyPathControllerError(
                "physical-key maximum correction differs"
            )
        self.cell_width = int(cell_width)
        self.context_width = int(context_width)
        self.max_correction = float(max_correction)
        self.cell_encoder = nn.Sequential(
            nn.Linear(self.feature_width, self.cell_width),
            nn.SiLU(),
            nn.Linear(self.cell_width, self.cell_width),
        )
        self.context_mixer = nn.Sequential(
            nn.Linear(4 * self.cell_width, self.context_width),
            nn.SiLU(),
            nn.Linear(self.context_width, self.cell_width),
        )
        self.correction_head = nn.Linear(self.cell_width, 1)
        self.gate_head = nn.Linear(self.cell_width, 1)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _compatibility(nerve: PhysicalKeyNerveResult) -> torch.Tensor:
        return torch.cat(
            (
                nerve.state_compatibility,
                nerve.action_compatibility,
                nerve.observer_compatibility,
            ),
            dim=1,
        )

    @staticmethod
    def _action_channel(
        template: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        if values.shape != (
            template.shape[0],
            PRIMARY_ACTIONS,
            template.shape[2],
        ):
            raise PhysicalKeyPathControllerError(
                "physical-key ordered action channel differs"
            )
        output = torch.zeros_like(template)
        start = PRIMARY_STATES
        output[:, start : start + PRIMARY_ACTIONS] = values
        return output

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

    @staticmethod
    def _active_slots(device: torch.device) -> torch.Tensor:
        return torch.tensor(
            tuple(range(PRIMARY_STATES))
            + tuple(MAX_STATES + index for index in range(PRIMARY_ACTIONS))
            + tuple(
                MAX_STATES + MAX_ACTIONS + index
                for index in range(PRIMARY_OBSERVERS)
            ),
            dtype=torch.long,
            device=device,
        )

    @classmethod
    def _type_context(cls, hidden: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(hidden)
        for start, end in cls._type_ranges():
            output[:, start:end] = hidden[:, start:end].mean(
                1,
                keepdim=True,
            )
        return output

    @classmethod
    def _type_global_context(
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
    def _type_column_mean(cls, values: torch.Tensor) -> torch.Tensor:
        output = torch.zeros_like(values)
        for start, end in cls._type_ranges():
            output[:, start:end] = values[:, start:end].mean(
                1,
                keepdim=True,
            )
        return output

    @classmethod
    def _type_indicators(
        cls,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indicators = [
            torch.zeros_like(values)
            for _ in range(3)
        ]
        for index, (start, end) in enumerate(cls._type_ranges()):
            indicators[index][:, start:end] = 1.0
        return tuple(indicators)  # type: ignore[return-value]

    def _features(
        self,
        witness: WitnessCompilerOutput,
        nerve: PhysicalKeyNerveResult,
    ) -> torch.Tensor:
        active_slots = self._active_slots(
            witness.key_assignment_logits.device
        )
        logits = witness.key_assignment_logits.float().index_select(
            1,
            active_slots,
        )
        valid = witness.unique_key_valid
        if (
            logits.ndim != 3
            or valid.shape != logits.shape[:1] + logits.shape[2:]
            or not bool(valid.any(-1).all())
        ):
            raise PhysicalKeyPathControllerError(
                "physical-key assignment geometry differs"
            )
        mask = valid[:, None]
        negative = torch.finfo(logits.dtype).min
        probabilities = logits.masked_fill(~mask, negative).softmax(-1)
        centered_logits = logits - _masked_mean(
            logits,
            mask,
            dim=-1,
            keepdim=True,
        )
        centered_logits = centered_logits.masked_fill(~mask, 0.0)
        compatibility = self._compatibility(nerve)
        if compatibility.shape != logits.shape:
            raise PhysicalKeyPathControllerError(
                "physical-key compatibility geometry differs"
            )
        row_mean = _masked_mean(
            compatibility,
            mask,
            dim=-1,
            keepdim=True,
        ).expand_as(compatibility)
        column_mean = self._type_column_mean(compatibility)
        tiny = torch.finfo(probabilities.dtype).tiny
        entropy = -(
            probabilities
            * probabilities.clamp_min(tiny).log()
        ).sum(-1, keepdim=True).expand_as(probabilities)
        transition = nerve.transition_relation
        observation = nerve.observation_relation
        source_out = _normalize(transition.sum((2, 3)))
        state_in = _normalize(transition.sum((1, 2)))
        action_mass = _normalize(transition.sum((1, 3)))
        observation_state = _normalize(observation.sum((2, 3)))
        observer_mass = _normalize(observation.sum((1, 3)))
        key_statistics = (
            source_out,
            state_in,
            action_mass,
            observation_state,
            observer_mass,
        )
        expanded_key_statistics = tuple(
            value[:, None].expand_as(probabilities)
            for value in key_statistics
        )
        state_type, action_type, observer_type = self._type_indicators(
            probabilities
        )
        action_left = self._action_channel(
            probabilities,
            nerve.action_left_compatibility,
        )
        action_right = self._action_channel(
            probabilities,
            nerve.action_right_compatibility,
        )
        action_observer = self._action_channel(
            probabilities,
            nerve.action_observer_compatibility,
        )
        action_commutator = self._action_channel(
            probabilities,
            nerve.action_commutator_compatibility,
        )
        path_scale = (
            nerve.path_mass / (1.0 + nerve.path_mass)
        )[:, None, None].expand_as(probabilities)
        features = torch.stack(
            (
                probabilities,
                centered_logits,
                compatibility,
                compatibility - row_mean,
                compatibility - column_mean,
                entropy,
                mask.to(probabilities.dtype).expand_as(probabilities),
                path_scale,
                *expanded_key_statistics,
                action_left,
                action_right,
                action_observer,
                action_commutator,
                action_left.abs(),
                action_right.abs(),
                action_observer.abs(),
                action_commutator.abs(),
                state_type,
                action_type,
                observer_type,
            ),
            dim=-1,
        )
        if (
            features.shape != logits.shape + (self.feature_width,)
            or not bool(torch.isfinite(features).all())
        ):
            raise PhysicalKeyPathControllerError(
                "physical-key cell features differ"
            )
        return features

    def forward(
        self,
        witness: WitnessCompilerOutput,
        record_valid: torch.Tensor,
        *,
        mode: str = "causal",
    ) -> PhysicalKeyPathControllerResult:
        if mode not in PATH_CONTROLLER_MODES:
            raise PhysicalKeyPathControllerError(
                f"unknown physical-key controller mode: {mode}"
            )
        nerve_mode = (
            mode
            if mode in {"causal", "broken-glue", "one-step-only"}
            else "causal"
        )
        nerve = physical_key_nerve(
            witness,
            record_valid,
            mode=nerve_mode,
        )
        features = self._features(witness, nerve)
        hidden = self.cell_encoder(features)
        valid = witness.unique_key_valid
        slot_context = _masked_mean(
            hidden,
            valid[:, None, :, None],
            dim=2,
            keepdim=True,
        ).expand_as(hidden)
        type_key_context = self._type_context(hidden)
        type_global_context = self._type_global_context(hidden, valid)
        mixed = self.context_mixer(
            torch.cat(
                (
                    hidden,
                    slot_context,
                    type_key_context,
                    type_global_context,
                ),
                dim=-1,
            )
        )
        gate = torch.sigmoid(self.gate_head(mixed).squeeze(-1))
        correction = (
            self.max_correction
            * gate
            * torch.tanh(self.correction_head(mixed).squeeze(-1))
        )
        path_present = nerve.path_mass.gt(0)[:, None, None].to(
            correction.dtype
        )
        correction = correction * path_present
        valid_mask = valid[:, None].to(correction.dtype)
        correction = correction * valid_mask
        correction = correction - _masked_mean(
            correction,
            valid[:, None],
            dim=-1,
            keepdim=True,
        )
        correction = correction * valid_mask
        row_magnitude = correction.abs().amax(
            -1,
            keepdim=True,
        ).clamp_min(torch.finfo(correction.dtype).tiny)
        correction = correction * (
            self.max_correction / row_magnitude
        ).clamp_max(1.0)
        if mode == "open-loop":
            correction = correction * 0.0
        elif mode == "sign-reversed":
            correction = -correction
        active_slots = self._active_slots(correction.device)
        full_correction = torch.zeros_like(
            witness.raw_key_assignment_logits,
            dtype=correction.dtype,
        )
        full_gate = torch.zeros_like(full_correction)
        full_correction[:, active_slots] = correction
        full_gate[:, active_slots] = gate
        revised = witness.raw_key_assignment_logits.float() + full_correction
        revised = revised.masked_fill(
            ~valid[:, None],
            torch.finfo(revised.dtype).min,
        )
        return PhysicalKeyPathControllerResult(
            raw_key_assignment_logits=revised,
            correction=full_correction,
            gate=full_gate,
            nerve=nerve,
            mode=mode,
        )


__all__ = [
    "PATH_CONTROLLER_MODES",
    "PhysicalKeyPathController",
    "PhysicalKeyPathControllerError",
    "PhysicalKeyPathControllerResult",
]
