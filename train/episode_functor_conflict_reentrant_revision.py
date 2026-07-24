"""Conflict-gated reentrant revision for an episodic functor machine.

The module is a trainable, permutation-equivariant machine-revision core.  It
receives only compiler-owned categorical claims and record features, revises a
provisional machine through tied contradiction-feedback cycles, and can seal
only the ordinary source-deleted ``HardFunctorMachine``.

It does not parse source bytes, execute late queries, retain a scratchpad, or
invoke a host solver.  The caller must destroy all claims, incidences, record
features, and diagnostics before detached query execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from episode_functor_constrained_transport import (
    LawfulProjection,
    PRIMARY_ACTIONS,
    PRIMARY_ANSWERS,
    PRIMARY_OBSERVERS,
    PRIMARY_STATES,
)
from episode_functor_machine import (
    HardFunctorMachine,
    MAX_ACTIONS,
    MAX_ANSWERS,
    MAX_OBSERVERS,
    MAX_STATES,
    SoftFunctorMachine,
)


TRANSITION_ROWS = PRIMARY_ACTIONS * PRIMARY_STATES
OBSERVER_ROWS = PRIMARY_OBSERVERS * PRIMARY_STATES
MACHINE_ROWS = TRANSITION_ROWS + OBSERVER_ROWS
MACHINE_CATEGORIES = PRIMARY_STATES
REVISION_MODES = frozenset(
    {"causal", "deranged", "open-loop", "sign-scrambled"}
)
UPDATE_METRICS = frozenset({"euclidean", "quotient-fisher"})


class ConflictReentrantRevisionError(ValueError):
    """CGRFC revision geometry, values, or sealing failed closed."""


@dataclass(frozen=True, slots=True)
class ConflictRevisionBatch:
    """Compiler-owned evidence presented to the reentrant revision core."""

    transition_logits: torch.Tensor
    observer_logits: torch.Tensor
    claim_logits: torch.Tensor
    closure_claim_logits: torch.Tensor
    claim_incidence: torch.Tensor
    closure_incidence: torch.Tensor
    record_features: torch.Tensor
    record_valid: torch.Tensor

    def __post_init__(self) -> None:
        if (
            not isinstance(self.transition_logits, torch.Tensor)
            or self.transition_logits.ndim != 4
        ):
            raise ConflictReentrantRevisionError(
                "transition logits must be rank four"
            )
        batch = int(self.transition_logits.shape[0])
        if (
            self.transition_logits.shape
            != (
                batch,
                PRIMARY_ACTIONS,
                PRIMARY_STATES,
                PRIMARY_STATES,
            )
            or self.observer_logits.shape
            != (
                batch,
                PRIMARY_OBSERVERS,
                PRIMARY_STATES,
                PRIMARY_ANSWERS,
            )
            or self.claim_logits.ndim != 4
            or self.claim_logits.shape[0] != batch
            or self.claim_logits.shape[2:]
            != (MACHINE_ROWS, MACHINE_CATEGORIES)
            or self.closure_claim_logits.shape != self.claim_logits.shape
            or self.claim_incidence.shape
            != (
                batch,
                self.claim_logits.shape[1],
                MACHINE_ROWS,
            )
            or self.closure_incidence.shape != self.claim_incidence.shape
            or self.record_features.ndim != 3
            or self.record_features.shape[:2]
            != self.claim_logits.shape[:2]
            or self.record_valid.shape != self.claim_logits.shape[:2]
            or self.record_valid.dtype != torch.bool
        ):
            raise ConflictReentrantRevisionError(
                "conflict revision batch geometry differs"
            )
        floating = (
            self.transition_logits,
            self.observer_logits,
            self.claim_logits,
            self.closure_claim_logits,
            self.claim_incidence,
            self.closure_incidence,
            self.record_features,
        )
        if (
            any(not value.is_floating_point() for value in floating)
            or any(not bool(torch.isfinite(value).all()) for value in floating)
            or bool(self.claim_incidence.lt(0).any())
            or bool(self.closure_incidence.lt(0).any())
        ):
            raise ConflictReentrantRevisionError(
                "conflict revision batch values differ"
            )
        devices = {value.device for value in floating}
        devices.add(self.record_valid.device)
        if len(devices) != 1:
            raise ConflictReentrantRevisionError(
                "conflict revision tensors must share one device"
            )
        invalid_mass = (
            self.claim_incidence + self.closure_incidence
        ) * (~self.record_valid)[..., None].to(
                self.claim_incidence.dtype
        )
        if bool(invalid_mass.ne(0).any()):
            raise ConflictReentrantRevisionError(
                "invalid records carry claim incidence"
            )

    @property
    def batch_size(self) -> int:
        return int(self.transition_logits.shape[0])

    @property
    def record_count(self) -> int:
        return int(self.claim_logits.shape[1])

    @property
    def record_width(self) -> int:
        return int(self.record_features.shape[-1])


@dataclass(frozen=True, slots=True)
class ConflictRevisionResult:
    """Attached revision result; diagnostics must not cross the seal."""

    projection: LawfulProjection
    cycle_transition_logits: tuple[torch.Tensor, ...]
    cycle_observer_logits: tuple[torch.Tensor, ...]
    cycle_claim_incidence: tuple[torch.Tensor, ...]
    cycle_closure_incidence: tuple[torch.Tensor, ...]
    contradiction_energy: torch.Tensor
    step_scale: torch.Tensor
    routing_mode: str
    update_metric: str

    def __post_init__(self) -> None:
        cycles = len(self.cycle_transition_logits)
        if (
            cycles < 1
            or len(self.cycle_observer_logits) != cycles
            or len(self.cycle_claim_incidence) != cycles
            or len(self.cycle_closure_incidence) != cycles
            or self.contradiction_energy.ndim != 2
            or self.contradiction_energy.shape[1] != cycles + 1
            or self.step_scale.shape
            != (
                self.contradiction_energy.shape[0],
                cycles,
                MACHINE_ROWS,
            )
            or self.routing_mode not in REVISION_MODES
            or self.update_metric not in UPDATE_METRICS
            or not bool(torch.isfinite(self.contradiction_energy).all())
            or not bool(torch.isfinite(self.step_scale).all())
        ):
            raise ConflictReentrantRevisionError(
                "conflict revision result differs"
            )


def _support_mask(device: torch.device) -> torch.Tensor:
    support = torch.ones(
        (MACHINE_ROWS, MACHINE_CATEGORIES),
        dtype=torch.bool,
        device=device,
    )
    support[TRANSITION_ROWS:, PRIMARY_ANSWERS:] = False
    return support


def _row_type(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    value = torch.zeros(
        (MACHINE_ROWS, 2),
        dtype=dtype,
        device=device,
    )
    value[:TRANSITION_ROWS, 0] = 1.0
    value[TRANSITION_ROWS:, 1] = 1.0
    return value


def _flatten_machine_logits(
    transition_logits: torch.Tensor,
    observer_logits: torch.Tensor,
) -> torch.Tensor:
    batch = int(transition_logits.shape[0])
    transition = transition_logits.reshape(
        batch,
        TRANSITION_ROWS,
        PRIMARY_STATES,
    )
    observer = F.pad(
        observer_logits.reshape(
            batch,
            OBSERVER_ROWS,
            PRIMARY_ANSWERS,
        ),
        (0, MACHINE_CATEGORIES - PRIMARY_ANSWERS),
        value=-20.0,
    )
    return torch.cat((transition, observer), dim=1).float()


def _split_machine_logits(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = int(logits.shape[0])
    transition = logits[:, :TRANSITION_ROWS].reshape(
        batch,
        PRIMARY_ACTIONS,
        PRIMARY_STATES,
        PRIMARY_STATES,
    )
    observer = logits[
        :,
        TRANSITION_ROWS:,
        :PRIMARY_ANSWERS,
    ].reshape(
        batch,
        PRIMARY_OBSERVERS,
        PRIMARY_STATES,
        PRIMARY_ANSWERS,
    )
    return transition, observer


def _masked_probabilities(
    logits: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    negative = torch.finfo(torch.float32).min
    masked = logits.float().masked_fill(~support, negative)
    return masked.softmax(-1)


def _masked_gauge_fix(
    values: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    weights = support.to(values.dtype)
    mean = (values * weights).sum(-1, keepdim=True) / weights.sum(
        -1,
        keepdim=True,
    )
    return (values - mean).masked_fill(~support, -20.0)


def _active_logits(
    *,
    batch: int,
    maximum: int,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    logits = torch.full(
        (batch, maximum, 2),
        -20.0,
        dtype=torch.float32,
        device=device,
    )
    logits[:, :, 0] = 20.0
    logits[:, :count, 0] = -20.0
    logits[:, :count, 1] = 20.0
    return logits


def _projection_from_rows(rows: torch.Tensor) -> LawfulProjection:
    transition, observer = _split_machine_logits(rows)
    batch = int(rows.shape[0])
    device = rows.device
    action_next = torch.full(
        (batch, MAX_ACTIONS, MAX_STATES, MAX_STATES),
        -20.0,
        dtype=torch.float32,
        device=device,
    )
    action_next[
        :,
        :PRIMARY_ACTIONS,
        :PRIMARY_STATES,
        :PRIMARY_STATES,
    ] = transition
    observer_answer = torch.full(
        (batch, MAX_OBSERVERS, MAX_STATES, MAX_ANSWERS),
        -20.0,
        dtype=torch.float32,
        device=device,
    )
    observer_answer[
        :,
        :PRIMARY_OBSERVERS,
        :PRIMARY_STATES,
        :PRIMARY_ANSWERS,
    ] = observer
    machine = SoftFunctorMachine(
        state_active=_active_logits(
            batch=batch,
            maximum=MAX_STATES,
            count=PRIMARY_STATES,
            device=device,
        ),
        action_active=_active_logits(
            batch=batch,
            maximum=MAX_ACTIONS,
            count=PRIMARY_ACTIONS,
            device=device,
        ),
        observer_active=_active_logits(
            batch=batch,
            maximum=MAX_OBSERVERS,
            count=PRIMARY_OBSERVERS,
            device=device,
        ),
        action_next=action_next,
        observer_answer=observer_answer,
    )
    return LawfulProjection(
        machine=machine,
        transition_transport=transition.softmax(-1),
        observer_transport=observer.softmax(-1),
    )


class ConflictGatedReentrantRevision(nn.Module):
    """Tied primal-dual correction over anonymous categorical machine cells."""

    feature_count = 16

    def __init__(
        self,
        *,
        record_width: int = 512,
        controller_width: int = 960,
        cycles: int = 4,
        update_metric: str = "euclidean",
        max_step: float = 0.1,
        reinterpretation_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if (
            record_width < 32
            or controller_width < 128
            or cycles < 1
            or update_metric not in UPDATE_METRICS
            or not math.isfinite(max_step)
            or not 0.0 < max_step <= 1.0
            or not math.isfinite(reinterpretation_scale)
            or not 0.0 < reinterpretation_scale <= 4.0
        ):
            raise ConflictReentrantRevisionError(
                "conflict revision constructor differs"
            )
        self.record_width = int(record_width)
        self.controller_width = int(controller_width)
        self.cycles = int(cycles)
        self.update_metric = str(update_metric)
        self.max_step = float(max_step)
        self.reinterpretation_scale = float(reinterpretation_scale)
        width = self.controller_width
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_count, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.record_encoder = nn.Sequential(
            nn.LayerNorm(record_width),
            nn.Linear(record_width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.context_update = nn.Sequential(
            nn.LayerNorm(5 * width),
            nn.Linear(5 * width, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, width),
        )
        self.recurrent_cell = nn.GRUCell(width, width)
        self.direction_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        self.step_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 4),
            nn.SiLU(),
            nn.Linear(width // 4, 1),
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _route_incidence(
        incidence: torch.Tensor,
        *,
        mode: str,
    ) -> torch.Tensor:
        if mode not in REVISION_MODES:
            raise ConflictReentrantRevisionError(
                f"unknown conflict routing mode: {mode}"
            )
        if mode != "deranged":
            return incidence
        record_mass = incidence.sum(-1, keepdim=True)
        positive_records = record_mass.squeeze(-1).gt(0).sum(1)
        if bool(
            (
                record_mass.squeeze(-1).gt(0).any(1)
                & positive_records.lt(2)
            ).any()
        ):
            raise ConflictReentrantRevisionError(
                "deranged incidence requires two positive records"
            )
        other_incidence = incidence.sum(1, keepdim=True) - incidence
        other_distribution = other_incidence / other_incidence.sum(
            -1,
            keepdim=True,
        ).clamp_min(torch.finfo(incidence.dtype).tiny)
        return record_mass * other_distribution

    @staticmethod
    def _claim_statistics(
        *,
        claims: torch.Tensor,
        closure_claims: torch.Tensor,
        claim_incidence: torch.Tensor,
        closure_incidence: torch.Tensor,
        probabilities: torch.Tensor,
        support: torch.Tensor,
        mode: str,
    ) -> tuple[torch.Tensor, ...]:
        claim_weights = claim_incidence[..., None]
        closure_weights = closure_incidence[..., None]
        claim_mass_raw = claim_weights.sum(1)
        closure_mass_raw = closure_weights.sum(1)
        claim_mass = claim_mass_raw.clamp_min(
            torch.finfo(claim_weights.dtype).tiny
        )
        closure_mass = closure_mass_raw.clamp_min(
            torch.finfo(closure_weights.dtype).tiny
        )
        claim_mean = (claim_weights * claims).sum(1) / claim_mass
        closure_mean = (
            closure_weights * closure_claims
        ).sum(1) / closure_mass
        centered = claims - claim_mean[:, None]
        variance = (
            claim_weights * centered.square()
        ).sum(1) / claim_mass
        claim_present = claim_mass_raw.gt(0).to(probabilities.dtype)
        closure_present = closure_mass_raw.gt(0).to(probabilities.dtype)
        residual = (claim_mean - probabilities) * claim_present
        closure_residual = (
            closure_mean - probabilities
        ) * closure_present
        if mode == "open-loop":
            claim_mean = probabilities + (claim_mean - probabilities) * 0.0
            closure_mean = (
                probabilities + (closure_mean - probabilities) * 0.0
            )
            residual = residual * 0.0
            closure_residual = closure_residual * 0.0
            variance = variance * 0.0
            claim_mass_raw = claim_mass_raw * 0.0
            closure_mass_raw = closure_mass_raw * 0.0
        elif mode == "sign-scrambled":
            residual = -residual
            closure_residual = -closure_residual
        mask = support.to(probabilities.dtype)
        return (
            claim_mean * mask,
            closure_mean * mask,
            variance * mask,
            residual * mask,
            closure_residual * mask,
            claim_mass_raw.squeeze(-1),
            closure_mass_raw.squeeze(-1),
        )

    @staticmethod
    def _contradiction_energy(
        *,
        variance: torch.Tensor,
        residual: torch.Tensor,
        closure_residual: torch.Tensor,
        support: torch.Tensor,
    ) -> torch.Tensor:
        value = (
            variance
            + residual.square()
            + closure_residual.square()
        ) * support
        return value.sum((-2, -1)) / support.sum()

    @staticmethod
    def _metric_direction(
        cotangent: torch.Tensor,
        probabilities: torch.Tensor,
        support: torch.Tensor,
        *,
        metric: str,
    ) -> torch.Tensor:
        if metric == "euclidean":
            raised = cotangent
        elif metric == "quotient-fisher":
            raised = cotangent / probabilities.clamp_min(1e-6)
        else:
            raise ConflictReentrantRevisionError(
                f"unknown categorical update metric: {metric}"
            )
        weights = support.to(raised.dtype)
        raised = raised - (raised * weights).sum(
            -1,
            keepdim=True,
        ) / weights.sum(-1, keepdim=True)
        raised = raised * weights
        scale = raised.abs().amax(-1, keepdim=True).clamp_min(1.0)
        return raised / scale

    @staticmethod
    def _reweight_incidence(
        base: torch.Tensor,
        compatibility: torch.Tensor,
        record_valid: torch.Tensor,
    ) -> torch.Tensor:
        total = base.sum(-1, keepdim=True)
        logits = base.clamp_min(1e-8).log() + compatibility
        weights = logits.softmax(-1) * total
        return weights * record_valid[..., None].to(weights.dtype)

    def forward(
        self,
        batch: ConflictRevisionBatch,
        *,
        routing_mode: str = "causal",
        update_metric: str | None = None,
    ) -> ConflictRevisionResult:
        if batch.record_width != self.record_width:
            raise ConflictReentrantRevisionError(
                "record feature width differs"
            )
        metric = self.update_metric if update_metric is None else update_metric
        if metric not in UPDATE_METRICS:
            raise ConflictReentrantRevisionError(
                "categorical update metric differs"
            )
        support = _support_mask(batch.transition_logits.device)
        support_batch = support[None]
        rows = _flatten_machine_logits(
            batch.transition_logits,
            batch.observer_logits,
        )
        rows = _masked_gauge_fix(rows, support_batch)
        claim_support = support[None, None]
        base_claim_incidence = self._route_incidence(
            batch.claim_incidence
            * batch.record_valid[..., None].to(
                batch.claim_incidence.dtype
            ),
            mode=routing_mode,
        )
        base_closure_incidence = self._route_incidence(
            batch.closure_incidence
            * batch.record_valid[..., None].to(
                batch.closure_incidence.dtype
            ),
            mode=routing_mode,
        )
        base_claim_present = base_claim_incidence.sum(1).gt(0)[
            :,
            None,
            :,
            None,
        ]
        base_closure_present = base_closure_incidence.sum(1).gt(0)[
            :,
            None,
            :,
            None,
        ]
        base_claim_logits = torch.where(
            base_claim_present,
            batch.claim_logits.float(),
            torch.zeros_like(batch.claim_logits, dtype=torch.float32),
        )
        base_closure_claim_logits = torch.where(
            base_closure_present,
            batch.closure_claim_logits.float(),
            torch.zeros_like(
                batch.closure_claim_logits,
                dtype=torch.float32,
            ),
        )
        claim_logits = base_claim_logits
        closure_claim_logits = base_closure_claim_logits
        claim_incidence = base_claim_incidence
        closure_incidence = base_closure_incidence
        record_states = self.record_encoder(batch.record_features)
        hidden = torch.zeros(
            (
                batch.batch_size,
                MACHINE_ROWS,
                MACHINE_CATEGORIES,
                self.controller_width,
            ),
            dtype=record_states.dtype,
            device=record_states.device,
        )
        record_hidden = torch.zeros_like(record_states)
        previous_update = torch.zeros_like(rows)
        transition_cycles: list[torch.Tensor] = []
        observer_cycles: list[torch.Tensor] = []
        claim_incidence_cycles: list[torch.Tensor] = []
        closure_incidence_cycles: list[torch.Tensor] = []
        energies: list[torch.Tensor] = []
        steps: list[torch.Tensor] = []
        row_types = _row_type(rows.device, rows.dtype)

        for cycle in range(self.cycles):
            claim_present = claim_incidence.sum(1).gt(0)[
                :,
                None,
                :,
                None,
            ]
            closure_present = closure_incidence.sum(1).gt(0)[
                :,
                None,
                :,
                None,
            ]
            claims = _masked_probabilities(
                torch.where(
                    claim_present,
                    claim_logits,
                    torch.zeros_like(claim_logits),
                ),
                claim_support,
            )
            closure_claims = _masked_probabilities(
                torch.where(
                    closure_present,
                    closure_claim_logits,
                    torch.zeros_like(closure_claim_logits),
                ),
                claim_support,
            )
            probabilities = _masked_probabilities(rows, support_batch)
            (
                claim_mean,
                closure_mean,
                variance,
                residual,
                closure_residual,
                claim_mass,
                closure_mass,
            ) = self._claim_statistics(
                claims=claims,
                closure_claims=closure_claims,
                claim_incidence=claim_incidence,
                closure_incidence=closure_incidence,
                probabilities=probabilities,
                support=support_batch,
                mode=routing_mode,
            )
            if cycle == 0:
                energies.append(
                    self._contradiction_energy(
                        variance=variance,
                        residual=residual,
                        closure_residual=closure_residual,
                        support=support_batch,
                    )
                )
            entropy = -(
                probabilities
                * probabilities.clamp_min(
                    torch.finfo(probabilities.dtype).tiny
                ).log()
            ).sum(-1, keepdim=True).expand_as(probabilities)
            maximum = probabilities.amax(
                -1,
                keepdim=True,
            ).expand_as(probabilities)
            type_transition = row_types[:, 0][None, :, None].expand_as(
                probabilities
            )
            type_observer = row_types[:, 1][None, :, None].expand_as(
                probabilities
            )
            cycle_value = torch.full_like(
                probabilities,
                float(cycle) / max(1, self.cycles - 1),
            )
            variance_magnitude = (
                torch.sqrt(variance + 1e-8) - 1e-4
            ).clamp_min(0.0)
            error_magnitude = (
                residual.abs()
                + closure_residual.abs()
                + variance_magnitude
            )
            feature = torch.stack(
                (
                    rows,
                    probabilities,
                    claim_mean,
                    residual,
                    residual.abs(),
                    variance,
                    closure_mean,
                    closure_residual,
                    (
                        claim_mass + closure_mass
                    )[..., None].expand_as(probabilities),
                    entropy,
                    maximum,
                    type_transition,
                    type_observer,
                    cycle_value,
                    previous_update,
                    error_magnitude,
                ),
                dim=-1,
            )
            encoded = self.feature_encoder(feature)
            combined_incidence = claim_incidence + closure_incidence
            record_mass = combined_incidence.sum(1).clamp_min(
                torch.finfo(combined_incidence.dtype).tiny
            )
            row_record_context = torch.einsum(
                "brl,brd->bld",
                combined_incidence,
                record_states + record_hidden,
            ) / record_mass[..., None]
            if routing_mode == "open-loop":
                row_record_context = row_record_context * 0.0
            row_context = hidden.mean(-2, keepdim=True).expand_as(hidden)
            column_context = torch.zeros_like(hidden)
            for start, end in (
                (0, TRANSITION_ROWS),
                (TRANSITION_ROWS, MACHINE_ROWS),
            ):
                column_context[:, start:end] = hidden[
                    :,
                    start:end,
                ].mean(1, keepdim=True)
            type_context = torch.zeros_like(hidden)
            for start, end in (
                (0, TRANSITION_ROWS),
                (TRANSITION_ROWS, MACHINE_ROWS),
            ):
                type_context[:, start:end] = hidden[
                    :,
                    start:end,
                ].mean(
                    (1, 2),
                    keepdim=True,
                )
            source_context = row_record_context[:, :, None].expand_as(hidden)
            mixed = self.context_update(
                torch.cat(
                    (
                        hidden,
                        row_context,
                        column_context,
                        type_context,
                        source_context,
                    ),
                    dim=-1,
                )
            )
            flat_hidden = hidden.reshape(-1, self.controller_width)
            hidden = self.recurrent_cell(
                (encoded + mixed).reshape(-1, self.controller_width),
                flat_hidden,
            ).reshape_as(hidden)
            cotangent = self.direction_head(hidden).squeeze(-1)
            direction = self._metric_direction(
                cotangent,
                probabilities,
                support_batch,
                metric=metric,
            )
            row_hidden = hidden.mean(-2)
            step = self.max_step * torch.sigmoid(
                self.step_head(row_hidden).squeeze(-1)
            )
            evidence_mass = claim_mass + closure_mass
            unobserved = evidence_mass.eq(0).to(error_magnitude.dtype)
            gate = torch.tanh(
                error_magnitude.sum(-1) + unobserved
            )
            step = step * gate
            update = step[..., None] * direction
            rows = _masked_gauge_fix(rows + update, support_batch)
            previous_update = update
            transition, observer = _split_machine_logits(rows)
            transition_cycles.append(transition)
            observer_cycles.append(observer)

            row_hidden = hidden.mean(-2)
            record_total = combined_incidence.sum(
                -1,
                keepdim=True,
            ).clamp_min(torch.finfo(combined_incidence.dtype).tiny)
            machine_to_record = torch.einsum(
                "brl,bld->brd",
                combined_incidence,
                row_hidden,
            ) / record_total
            record_hidden = self.recurrent_cell(
                (record_states + machine_to_record).reshape(
                    -1,
                    self.controller_width,
                ),
                record_hidden.reshape(-1, self.controller_width),
            ).reshape_as(record_hidden)
            compatibility = torch.einsum(
                "brd,bld->brl",
                record_hidden,
                row_hidden,
            ) / math.sqrt(self.controller_width)
            cell_compatibility = torch.einsum(
                "brd,blcd->brlc",
                record_hidden,
                hidden,
            ) / math.sqrt(self.controller_width)
            if routing_mode == "open-loop":
                compatibility = compatibility * 0.0
                cell_compatibility = cell_compatibility * 0.0
                claim_incidence = base_claim_incidence
                closure_incidence = base_closure_incidence
                claim_logits = base_claim_logits
                closure_claim_logits = base_closure_claim_logits
            else:
                compatibility = self.reinterpretation_scale * torch.tanh(
                    compatibility
                )
                claim_incidence = self._reweight_incidence(
                    base_claim_incidence,
                    compatibility,
                    batch.record_valid,
                )
                closure_incidence = self._reweight_incidence(
                    base_closure_incidence,
                    compatibility,
                    batch.record_valid,
                )
                claim_logits = (
                    base_claim_logits
                    + self.reinterpretation_scale
                    * torch.tanh(cell_compatibility)
                )
                closure_claim_logits = (
                    base_closure_claim_logits
                    + self.reinterpretation_scale
                    * torch.tanh(cell_compatibility)
                )
            claim_incidence_cycles.append(claim_incidence)
            closure_incidence_cycles.append(closure_incidence)

            next_claim_present = claim_incidence.sum(1).gt(0)[
                :,
                None,
                :,
                None,
            ]
            next_closure_present = closure_incidence.sum(1).gt(0)[
                :,
                None,
                :,
                None,
            ]
            next_claims = _masked_probabilities(
                torch.where(
                    next_claim_present,
                    claim_logits,
                    torch.zeros_like(claim_logits),
                ),
                claim_support,
            )
            next_closure_claims = _masked_probabilities(
                torch.where(
                    next_closure_present,
                    closure_claim_logits,
                    torch.zeros_like(closure_claim_logits),
                ),
                claim_support,
            )
            next_probabilities = _masked_probabilities(rows, support_batch)
            (
                _,
                _,
                next_variance,
                next_residual,
                next_closure_residual,
                _,
                _,
            ) = self._claim_statistics(
                claims=next_claims,
                closure_claims=next_closure_claims,
                claim_incidence=claim_incidence,
                closure_incidence=closure_incidence,
                probabilities=next_probabilities,
                support=support_batch,
                mode=routing_mode,
            )
            energies.append(
                self._contradiction_energy(
                    variance=next_variance,
                    residual=next_residual,
                    closure_residual=next_closure_residual,
                    support=support_batch,
                )
            )
            steps.append(step)

        projection = _projection_from_rows(rows)
        return ConflictRevisionResult(
            projection=projection,
            cycle_transition_logits=tuple(transition_cycles),
            cycle_observer_logits=tuple(observer_cycles),
            cycle_claim_incidence=tuple(claim_incidence_cycles),
            cycle_closure_incidence=tuple(
                closure_incidence_cycles
            ),
            contradiction_energy=torch.stack(energies, dim=1),
            step_scale=torch.stack(steps, dim=1),
            routing_mode=routing_mode,
            update_metric=metric,
        )

    @torch.no_grad()
    def seal(
        self,
        result: ConflictRevisionResult,
    ) -> HardFunctorMachine:
        """Return only the ordinary categorical machine for detached use."""

        machine = result.projection.machine
        transition = machine.action_next[
            :,
            :PRIMARY_ACTIONS,
            :PRIMARY_STATES,
            :PRIMARY_STATES,
        ]
        observer = machine.observer_answer[
            :,
            :PRIMARY_OBSERVERS,
            :PRIMARY_STATES,
            :PRIMARY_ANSWERS,
        ]
        if (
            bool(transition.topk(2, dim=-1).values.diff().eq(0).any())
            or bool(observer.topk(2, dim=-1).values.diff().eq(0).any())
        ):
            raise ConflictReentrantRevisionError(
                "conflict revision sealing has a categorical tie"
            )
        return machine.harden()


__all__ = [
    "ConflictGatedReentrantRevision",
    "ConflictReentrantRevisionError",
    "ConflictRevisionBatch",
    "ConflictRevisionResult",
    "MACHINE_CATEGORIES",
    "MACHINE_ROWS",
    "OBSERVER_ROWS",
    "REVISION_MODES",
    "TRANSITION_ROWS",
    "UPDATE_METRICS",
]
