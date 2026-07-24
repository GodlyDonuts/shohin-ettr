"""Raw-source compiler wrapper for conflict-gated reentrant revision.

The wrapper connects the existing role-free witness encoder to CGRFC without a
host completion solver. Direct local sections use an equivariant trainable
record-local calibration branch that is algebraically distinct from the
provisional machine aggregation. Behavioral closure remains disabled until a
genuinely independent measurement exists. A tied second compiler pass can
route first-pass conflicts back to their physical source spans before the
final source-deleted seal.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
    project_key_assignment_logits,
)
from episode_functor_conflict_reentrant_revision import (
    ConflictGatedReentrantRevision,
    ConflictRevisionBatch,
    ConflictRevisionResult,
    MACHINE_CATEGORIES,
    MACHINE_ROWS,
    OBSERVER_ROWS,
    TRANSITION_ROWS,
    _flatten_machine_logits,
    _masked_probabilities,
    _projection_from_rows,
    _support_mask,
)
from episode_functor_machine import (
    HardFunctorKeys,
    HardFunctorMachine,
    MAX_ACTIONS,
    MAX_OBSERVERS,
    MAX_STATES,
)
from episode_functor_physical_key_controller import (
    PhysicalKeyPathController,
    PhysicalKeyPathControllerResult,
)
from episode_functor_witness_compiler import (
    MAX_RECORDS,
    RECORD_OBSERVATION,
    RECORD_TRANSITION,
    ROLE_ACTION,
    ROLE_OBSERVATION_STATE,
    ROLE_OBSERVER,
    ROLE_TRANSITION_DESTINATION,
    ROLE_TRANSITION_SOURCE,
    ProofCarryingWitnessCompiler,
    WitnessCompilerBatch,
    WitnessCompilerOutput,
    assemble_relation_evidence,
)
from pipeline.episode_functor_hankel_geometry import enumerate_action_words


class ConflictCompilerError(ValueError):
    """The integrated CGRFC source compiler failed closed."""


@dataclass(frozen=True, slots=True)
class ConflictCompilerOutput:
    """Attached compiler state; only ``seal`` may cross source deletion."""

    first_witness: WitnessCompilerOutput
    first_path_controller: PhysicalKeyPathControllerResult
    first_revision_batch: ConflictRevisionBatch
    first_revision: ConflictRevisionResult
    source_feedback: torch.Tensor
    witness: WitnessCompilerOutput
    path_controller: PhysicalKeyPathControllerResult
    revision_batch: ConflictRevisionBatch
    revision: ConflictRevisionResult

    def __post_init__(self) -> None:
        if (
            self.first_witness.record_type_logits.shape[:2]
            != self.first_revision_batch.record_features.shape[:2]
            or self.first_path_controller.correction.shape
            != self.first_witness.raw_key_assignment_logits.shape
            or self.first_revision.projection.machine.batch_size
            != self.first_revision_batch.batch_size
            or self.source_feedback.ndim != 3
            or self.source_feedback.shape[0]
            != self.first_revision_batch.batch_size
            or not self.source_feedback.is_floating_point()
            or not bool(torch.isfinite(self.source_feedback).all())
            or self.source_feedback.device
            != self.first_revision_batch.record_features.device
            or
            self.witness.record_type_logits.shape[:2]
            != self.revision_batch.record_features.shape[:2]
            or self.path_controller.correction.shape
            != self.witness.raw_key_assignment_logits.shape
            or self.revision.projection.machine.batch_size
            != self.revision_batch.batch_size
        ):
            raise ConflictCompilerError(
                "integrated conflict compiler output differs"
            )


@dataclass(frozen=True, slots=True)
class SealedConflictMachine:
    machine: HardFunctorMachine
    keys: HardFunctorKeys

    def __post_init__(self) -> None:
        if self.machine.batch_size != self.keys.batch_size:
            raise ConflictCompilerError(
                "sealed conflict machine and key batches differ"
            )


class DirectEvidenceProjector(nn.Module):
    """Zero-parameter provisional projection with no completion oracle."""

    sinkhorn_iterations = 0

    def parameter_count(self) -> int:
        return 0

    def forward(
        self,
        transition_logits: torch.Tensor,
        observer_logits: torch.Tensor,
        *,
        straight_through: bool = False,
    ) -> LawfulProjection:
        rows = _flatten_machine_logits(
            transition_logits,
            observer_logits,
        )
        if straight_through:
            transition_top = transition_logits.topk(2, dim=-1).values
            observer_top = observer_logits.topk(2, dim=-1).values
            if (
                bool(transition_top[..., 0].eq(transition_top[..., 1]).any())
                or bool(observer_top[..., 0].eq(observer_top[..., 1]).any())
            ):
                raise ConflictCompilerError(
                    "direct evidence projection has a hardening tie"
                )
            hard_transition = F.one_hot(
                transition_logits.argmax(-1),
                PRIMARY_STATES,
            ).to(rows.dtype)
            hard_observer = F.one_hot(
                observer_logits.argmax(-1),
                PRIMARY_ANSWERS,
            ).to(rows.dtype)
            hard_rows = _flatten_machine_logits(
                hard_transition.mul(40.0).sub(20.0),
                hard_observer.mul(40.0).sub(20.0),
            )
            rows = hard_rows + rows - rows.detach()
        return _projection_from_rows(rows)

    @torch.no_grad()
    def hard_project(
        self,
        transition_logits: torch.Tensor,
        observer_logits: torch.Tensor,
    ) -> HardFunctorMachine:
        return self(
            transition_logits,
            observer_logits,
            straight_through=True,
        ).machine.harden()


def _normalize(values: torch.Tensor) -> torch.Tensor:
    return values / values.sum(-1, keepdim=True).clamp_min(
        torch.finfo(values.dtype).tiny
    )


def _future_signatures(
    transition: torch.Tensor,
    observer: torch.Tensor,
    *,
    max_depth: int,
) -> torch.Tensor:
    """Return source-branch behavior `[B,S,W,O,Y]` without machine labels."""

    batch = int(transition.shape[0])
    words = enumerate_action_words(max_depth)
    identity = torch.eye(
        PRIMARY_STATES,
        dtype=transition.dtype,
        device=transition.device,
    )[None].expand(batch, -1, -1)
    distributions: dict[tuple[int, ...], torch.Tensor] = {(): identity}
    signatures: list[torch.Tensor] = []
    for word in words:
        if word:
            distributions[word] = torch.matmul(
                distributions[word[:-1]],
                transition[:, word[-1]],
            )
        signatures.append(
            torch.einsum(
                "bsj,bojy->bsoy",
                distributions[word],
                observer,
            )
        )
    return torch.stack(signatures, dim=2)


def behaviorally_close_record_claims(
    claim_logits: torch.Tensor,
    incidence: torch.Tensor,
    *,
    max_depth: int = 3,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Turn independent record claims into finite-behavior machine claims."""

    if (
        claim_logits.ndim != 4
        or claim_logits.shape[2:]
        != (MACHINE_ROWS, MACHINE_CATEGORIES)
        or incidence.shape != claim_logits.shape[:3]
        or not 0 <= max_depth <= 6
        or not math.isfinite(temperature)
        or temperature <= 0.0
    ):
        raise ConflictCompilerError(
            "behavioral closure claim geometry differs"
        )
    batch, records = claim_logits.shape[:2]
    support = torch.ones(
        (MACHINE_ROWS, MACHINE_CATEGORIES),
        dtype=torch.bool,
        device=claim_logits.device,
    )
    support[TRANSITION_ROWS:, PRIMARY_ANSWERS:] = False
    negative = torch.finfo(torch.float32).min
    claim = claim_logits.float().masked_fill(
        ~support[None, None],
        negative,
    ).softmax(-1)
    row_mass = incidence.sum(1)[..., None]
    uniform = support.to(claim.dtype)
    uniform = uniform / uniform.sum(-1, keepdim=True)
    row_claim = torch.einsum(
        "brl,brlc->blc",
        incidence,
        claim,
    ) / row_mass.clamp_min(torch.finfo(claim.dtype).tiny)
    row_claim = torch.where(
        row_mass.gt(0),
        row_claim,
        uniform[None],
    )
    transition = row_claim[:, :TRANSITION_ROWS].reshape(
        batch,
        PRIMARY_ACTIONS,
        PRIMARY_STATES,
        PRIMARY_STATES,
    )
    observer = row_claim[
        :,
        TRANSITION_ROWS:,
        :PRIMARY_ANSWERS,
    ].reshape(
        batch,
        PRIMARY_OBSERVERS,
        PRIMARY_STATES,
        PRIMARY_ANSWERS,
    )
    base = _future_signatures(
        transition,
        observer,
        max_depth=max_depth,
    )
    derivative = torch.einsum(
        "basd,bdwoy->baswoy",
        transition,
        base,
    )
    left = derivative[:, :, :, None]
    right = base[:, None, None]
    midpoint = 0.5 * (left + right)
    tiny = torch.finfo(midpoint.dtype).tiny
    distance = 0.5 * (
        left
        * (
            left.clamp_min(tiny).log()
            - midpoint.clamp_min(tiny).log()
        )
        + right
        * (
            right.clamp_min(tiny).log()
            - midpoint.clamp_min(tiny).log()
        )
    ).sum(-1).mean((-2, -1))
    transition_logits = -distance / temperature
    observer_logits = (
        base[:, :, 0]
        .permute(0, 2, 1, 3)
        .clamp_min(tiny)
        .log()
    )
    closed = torch.cat(
        (
            transition_logits.reshape(
                batch,
                TRANSITION_ROWS,
                PRIMARY_STATES,
            ),
            F.pad(
                observer_logits.reshape(
                    batch,
                    OBSERVER_ROWS,
                    PRIMARY_ANSWERS,
                ),
                (0, MACHINE_CATEGORIES - PRIMARY_ANSWERS),
                value=-20.0,
            ),
        ),
        dim=1,
    )
    return closed[:, None].expand(
        batch,
        records,
        MACHINE_ROWS,
        MACHINE_CATEGORIES,
    )


def record_features_from_witness(
    witness: WitnessCompilerOutput,
) -> torch.Tensor:
    """Build recoding-invariant record diagnostics for conflict control."""

    role_unique = witness.relation_evidence.record_role_unique
    role_slot = witness.relation_evidence.record_role_slot
    if (
        witness.record_type_logits.ndim != 3
        or witness.answer_logits.ndim != 3
        or role_unique.ndim != 4
        or role_slot.ndim != 4
        or witness.record_type_logits.shape[:2]
        != witness.answer_logits.shape[:2]
        or role_unique.shape[:2] != witness.record_type_logits.shape[:2]
        or role_slot.shape[:2] != witness.record_type_logits.shape[:2]
    ):
        raise ConflictCompilerError(
            "witness record feature geometry differs"
        )
    def summaries(
        values: torch.Tensor,
        *,
        include_margin: bool,
    ) -> tuple[torch.Tensor, ...]:
        probabilities = values.float().softmax(-1)
        tiny = torch.finfo(probabilities.dtype).tiny
        maximum = probabilities.amax(-1)
        entropy = -(
            probabilities * probabilities.clamp_min(tiny).log()
        ).sum(-1)
        output: tuple[torch.Tensor, ...] = (maximum, entropy)
        if include_margin:
            top = probabilities.topk(2, dim=-1).values
            output = output + (top[..., 0] - top[..., 1],)
        return output

    answer_summary = summaries(
        witness.answer_logits,
        include_margin=True,
    )
    unique_summary = summaries(
        role_unique,
        include_margin=False,
    )
    slot_maximum = role_slot.float().softmax(-1).amax(-1)
    record_type = witness.record_type_logits.float()
    type_centered = record_type - record_type.mean(-1, keepdim=True)
    unique_maximum, unique_entropy = unique_summary
    global_summary = torch.stack(
        (
            role_slot.float().softmax(-1).amax((-2, -1)),
            role_unique.float().softmax(-1).amax((-2, -1)),
            role_slot.float().softmax(-1).mean((-2, -1)),
        ),
        dim=-1,
    )
    features = torch.cat(
        (
            type_centered,
            torch.stack(answer_summary, dim=-1),
            unique_maximum,
            unique_entropy,
            slot_maximum,
            global_summary,
        ),
        dim=-1,
    )
    if features.shape[-1] != 32:
        raise ConflictCompilerError(
            "invariant record feature width differs"
        )
    return features


@torch.no_grad()
def _hard_assign_keys_without_solver(
    witness: WitnessCompilerOutput,
) -> HardFunctorKeys:
    """Copy a unique argmax key per active slot or fail closed."""

    logits = witness.key_assignment_logits.float()
    valid = witness.unique_key_valid
    keys = witness.unique_key_bytes
    if (
        logits.ndim != 3
        or valid.shape != logits.shape[:1] + logits.shape[2:]
        or keys.shape != valid.shape + (8,)
        or not bool(torch.isfinite(logits).all())
    ):
        raise ConflictCompilerError(
            "solver-free key assignment geometry differs"
        )
    active_slots = torch.tensor(
        tuple(range(PRIMARY_STATES))
        + tuple(MAX_STATES + index for index in range(PRIMARY_ACTIONS))
        + tuple(
            MAX_STATES + MAX_ACTIONS + index
            for index in range(PRIMARY_OBSERVERS)
        ),
        dtype=torch.long,
        device=logits.device,
    )
    active = logits.index_select(1, active_slots).masked_fill(
        ~valid[:, None],
        torch.finfo(logits.dtype).min,
    )
    top = active.topk(2, dim=-1)
    if bool(top.values[..., 0].eq(top.values[..., 1]).any()):
        raise ConflictCompilerError(
            "solver-free key assignment has a categorical tie"
        )
    selected = top.indices[..., 0]
    for row in range(int(logits.shape[0])):
        chosen = selected[row]
        if (
            int(valid[row].sum()) != int(active_slots.numel())
            or int(chosen.unique().numel()) != int(active_slots.numel())
            or not bool(valid[row, chosen].all())
        ):
            raise ConflictCompilerError(
                "solver-free key assignment is not a bijection"
            )
    state_keys = torch.zeros(
        (int(logits.shape[0]), MAX_STATES, 8),
        dtype=torch.uint8,
        device=logits.device,
    )
    action_keys = torch.zeros(
        (int(logits.shape[0]), MAX_ACTIONS, 8),
        dtype=torch.uint8,
        device=logits.device,
    )
    observer_keys = torch.zeros(
        (int(logits.shape[0]), MAX_OBSERVERS, 8),
        dtype=torch.uint8,
        device=logits.device,
    )
    rows = torch.arange(
        int(logits.shape[0]),
        device=logits.device,
    )[:, None]
    copied = keys[rows, selected]
    state_keys[:, :PRIMARY_STATES] = copied[:, :PRIMARY_STATES]
    action_keys[:, :PRIMARY_ACTIONS] = copied[
        :,
        PRIMARY_STATES : PRIMARY_STATES + PRIMARY_ACTIONS,
    ]
    observer_keys[:, :PRIMARY_OBSERVERS] = copied[
        :,
        -PRIMARY_OBSERVERS:,
    ]
    return HardFunctorKeys(
        state_keys=state_keys,
        action_keys=action_keys,
        observer_keys=observer_keys,
    )


SOURCE_REENTRY_MODES = frozenset(
    {"causal", "deranged", "open-loop", "sign-scrambled"}
)
CLAIM_CALIBRATION_MODES = frozenset({"causal", "identity"})


class ConflictSourceReentry(nn.Module):
    """Route first-pass categorical conflict back into source feature spans.

    This module does not retain a new runtime memory. It constructs an
    attached correction tensor, the same witness compiler is replayed with
    that tensor added to its read-only trunk features, and the tensor is
    destroyed before sealing.
    """

    cell_feature_width = 3

    def __init__(
        self,
        *,
        record_width: int = 32,
        context_width: int = 512,
        bottleneck_width: int = 128,
        external_feature_width: int = 1728,
        max_feedback: float = 0.25,
    ) -> None:
        super().__init__()
        if (
            record_width < 32
            or context_width < 128
            or bottleneck_width < 16
            or external_feature_width < 1
            or not math.isfinite(max_feedback)
            or not 0.0 < max_feedback <= 1.0
        ):
            raise ConflictCompilerError(
                "conflict source reentry geometry differs"
            )
        self.record_width = int(record_width)
        self.context_width = int(context_width)
        self.bottleneck_width = int(bottleneck_width)
        self.external_feature_width = int(external_feature_width)
        self.max_feedback = float(max_feedback)
        self.cell_encoder = nn.Sequential(
            nn.Linear(
                self.cell_feature_width,
                bottleneck_width,
                bias=False,
            ),
            nn.SiLU(),
            nn.Linear(
                bottleneck_width,
                context_width,
                bias=False,
            ),
        )
        self.span_decoder = nn.Sequential(
            nn.Linear(
                context_width,
                bottleneck_width,
                bias=False,
            ),
            nn.SiLU(),
            nn.Linear(
                bottleneck_width,
                external_feature_width,
                bias=False,
            ),
        )
        self.feedback_gate = nn.Parameter(torch.tensor(-2.0))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _terminal_rows(result: ConflictRevisionResult) -> torch.Tensor:
        return _flatten_machine_logits(
            result.cycle_transition_logits[-1],
            result.cycle_observer_logits[-1],
        )

    def forward(
        self,
        batch: WitnessCompilerBatch,
        revision_batch: ConflictRevisionBatch,
        result: ConflictRevisionResult,
        *,
        byte_count: int,
        mode: str = "causal",
    ) -> torch.Tensor:
        if mode not in SOURCE_REENTRY_MODES:
            raise ConflictCompilerError(
                f"unknown conflict source reentry mode: {mode}"
            )
        if (
            byte_count < 0
            or revision_batch.record_width != self.record_width
            or revision_batch.batch_size != batch.batch_size
        ):
            raise ConflictCompilerError(
                "conflict source reentry input differs"
            )
        support = _support_mask(
            revision_batch.transition_logits.device
        )[None]
        initial = _flatten_machine_logits(
            revision_batch.transition_logits,
            revision_batch.observer_logits,
        )
        terminal = self._terminal_rows(result)
        signed_delta = (
            _masked_probabilities(terminal, support)
            - _masked_probabilities(initial, support)
        )
        if mode == "sign-scrambled":
            signed_delta = -signed_delta
        cell_feature = torch.stack(
            (
                signed_delta,
                signed_delta.abs(),
                signed_delta * signed_delta.abs(),
            ),
            dim=-1,
        )
        row_context = self.cell_encoder(cell_feature).mean(-2)
        incidence = (
            result.cycle_claim_incidence[-1]
            + result.cycle_closure_incidence[-1]
        )
        if mode == "deranged":
            record_mass = incidence.sum(-1, keepdim=True)
            positive_records = record_mass.squeeze(-1).gt(0).sum(1)
            if bool(
                (
                    record_mass.squeeze(-1).gt(0).any(1)
                    & positive_records.lt(2)
                ).any()
            ):
                raise ConflictCompilerError(
                    "deranged source reentry requires two positive records"
                )
            other_incidence = incidence.sum(1, keepdim=True) - incidence
            other_distribution = other_incidence / other_incidence.sum(
                -1,
                keepdim=True,
            ).clamp_min(torch.finfo(incidence.dtype).tiny)
            incidence = record_mass * other_distribution
        record_mass = incidence.sum(-1, keepdim=True).clamp_min(
            torch.finfo(incidence.dtype).tiny
        )
        record_context = torch.einsum(
            "brl,bld->brd",
            incidence,
            row_context,
        ) / record_mass
        record_context = record_context * batch.record_valid[
            ...,
            None,
        ].to(record_context.dtype)
        record_feedback = self.span_decoder(record_context)
        scale = self.max_feedback * torch.sigmoid(self.feedback_gate)
        record_feedback = scale * torch.tanh(record_feedback)
        feedback = torch.zeros(
            (
                batch.batch_size,
                byte_count,
                self.external_feature_width,
            ),
            dtype=record_feedback.dtype,
            device=record_feedback.device,
        )
        for row in range(batch.batch_size):
            record_count = int(batch.record_valid[row].sum())
            for record in range(record_count):
                start, end = (
                    int(value)
                    for value in batch.record_bounds[row, record].tolist()
                )
                if not 0 <= start < end <= byte_count:
                    raise ConflictCompilerError(
                        "conflict source span leaves byte inventory"
                    )
                feedback[row, start:end] = record_feedback[row, record]
        return feedback * (0.0 if mode == "open-loop" else 1.0)


class ConflictClaimAdapter(nn.Module):
    """Construct equivariant record-local claims independent of aggregation.

    The provisional machine aggregates destination and answer evidence
    linearly. This branch applies a learned, category-shared nonlinear
    calibration to each record before aggregation. Sharing the scalar basis
    across categories preserves state/answer recoding equivariance while
    making local claims algebraically distinct from the global machine.
    """

    def __init__(
        self,
        *,
        record_width: int = 32,
        hidden_width: int = 2288,
        basis_width: int = 16,
    ) -> None:
        super().__init__()
        if (
            record_width < 32
            or hidden_width < 64
            or basis_width < 4
        ):
            raise ConflictCompilerError(
                "conflict claim adapter geometry differs"
            )
        self.record_width = int(record_width)
        self.hidden_width = int(hidden_width)
        self.basis_width = int(basis_width)
        self.record_encoder = nn.Sequential(
            nn.LayerNorm(record_width),
            nn.Linear(record_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
        )
        self.destination_coefficients = nn.Linear(
            hidden_width,
            basis_width,
        )
        self.answer_coefficients = nn.Linear(
            hidden_width,
            basis_width,
        )
        self.destination_basis = nn.Sequential(
            nn.Linear(3, basis_width),
            nn.SiLU(),
            nn.Linear(basis_width, basis_width),
        )
        self.answer_basis = nn.Sequential(
            nn.Linear(3, basis_width),
            nn.SiLU(),
            nn.Linear(basis_width, basis_width),
        )
        self.calibration_gate = nn.Parameter(torch.tensor(-2.0))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _calibrate(
        self,
        logits: torch.Tensor,
        context: torch.Tensor,
        *,
        coefficient_head: nn.Linear,
        basis: nn.Sequential,
        mode: str,
    ) -> torch.Tensor:
        probabilities = logits.float().softmax(-1)
        tiny = torch.finfo(probabilities.dtype).tiny
        category_features = torch.stack(
            (
                probabilities,
                probabilities.square(),
                probabilities.clamp_min(tiny).log(),
            ),
            dim=-1,
        )
        category_basis = basis(category_features)
        coefficients = coefficient_head(context)
        correction = torch.einsum(
            "brck,brk->brc",
            category_basis,
            coefficients,
        ) / math.sqrt(self.basis_width)
        correction = correction - correction.mean(-1, keepdim=True)
        if mode == "identity":
            correction = correction * 0.0
        return logits.float() + torch.sigmoid(
            self.calibration_gate
        ) * correction

    def forward(
        self,
        witness: WitnessCompilerOutput,
        *,
        record_features: torch.Tensor,
        record_valid: torch.Tensor,
        mode: str = "causal",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if mode not in CLAIM_CALIBRATION_MODES:
            raise ConflictCompilerError(
                f"unknown claim calibration mode: {mode}"
            )
        record_states = record_features
        if (
            record_states.ndim != 3
            or record_states.shape[1:] != (MAX_RECORDS, self.record_width)
            or record_valid.shape != record_states.shape[:2]
            or record_valid.dtype != torch.bool
            or record_valid.device != record_states.device
        ):
            raise ConflictCompilerError(
                "conflict claim adapter input differs"
            )
        batch = int(record_states.shape[0])
        role_slot = witness.relation_evidence.record_role_slot
        record_type = witness.record_type_logits.float().softmax(-1)
        action = _normalize(
            role_slot[
                :,
                :,
                ROLE_ACTION,
                MAX_STATES : MAX_STATES + PRIMARY_ACTIONS,
            ]
        )
        source = _normalize(
            role_slot[
                :,
                :,
                ROLE_TRANSITION_SOURCE,
                :PRIMARY_STATES,
            ]
        )
        destination = _normalize(
            role_slot[
                :,
                :,
                ROLE_TRANSITION_DESTINATION,
                :PRIMARY_STATES,
            ]
        )
        observer = _normalize(
            role_slot[
                :,
                :,
                ROLE_OBSERVER,
                MAX_STATES
                + MAX_ACTIONS : MAX_STATES
                + MAX_ACTIONS
                + PRIMARY_OBSERVERS,
            ]
        )
        observation_state = _normalize(
            role_slot[
                :,
                :,
                ROLE_OBSERVATION_STATE,
                :PRIMARY_STATES,
            ]
        )
        transition_incidence = torch.einsum(
            "br,bra,brs->bras",
            record_type[:, :, RECORD_TRANSITION],
            action,
            source,
        ).reshape(batch, MAX_RECORDS, TRANSITION_ROWS)
        observer_incidence = torch.einsum(
            "br,bro,brs->bros",
            record_type[:, :, RECORD_OBSERVATION],
            observer,
            observation_state,
        ).reshape(batch, MAX_RECORDS, OBSERVER_ROWS)
        direct_incidence = torch.cat(
            (transition_incidence, observer_incidence),
            dim=-1,
        ) * record_valid[..., None].to(record_states.dtype)

        context = self.record_encoder(record_states)
        tiny = torch.finfo(record_states.dtype).tiny
        transition_claim = self._calibrate(
            destination.clamp_min(tiny).log(),
            context,
            coefficient_head=self.destination_coefficients,
            basis=self.destination_basis,
            mode=mode,
        )
        transition_claim = transition_claim[:, :, None, None].expand(
            -1,
            -1,
            PRIMARY_ACTIONS,
            PRIMARY_STATES,
            -1,
        ).reshape(
            batch,
            MAX_RECORDS,
            TRANSITION_ROWS,
            MACHINE_CATEGORIES,
        )
        observer_claim = self._calibrate(
            witness.answer_logits.float(),
            context,
            coefficient_head=self.answer_coefficients,
            basis=self.answer_basis,
            mode=mode,
        )
        observer_claim = observer_claim[:, :, None, None].expand(
            -1,
            -1,
            PRIMARY_OBSERVERS,
            PRIMARY_STATES,
            -1,
        ).reshape(
            batch,
            MAX_RECORDS,
            OBSERVER_ROWS,
            PRIMARY_ANSWERS,
        )
        observer_claim = F.pad(
            observer_claim,
            (0, MACHINE_CATEGORIES - PRIMARY_ANSWERS),
            value=-20.0,
        )
        direct_claim = torch.cat(
            (transition_claim, observer_claim),
            dim=2,
        )

        closure_claim = torch.zeros_like(direct_claim)
        closure_incidence = torch.zeros_like(direct_incidence)
        return (
            direct_claim,
            closure_claim,
            direct_incidence,
            closure_incidence,
        )


class ConflictProofCarryingCompiler(nn.Module):
    """Maximum raw-source encoder plus reentrant causal commitment."""

    def __init__(
        self,
        *,
        external_feature_width: int = 1728,
        width: int = 512,
        encoder_layers: int = 8,
        decoder_layers: int = 4,
        heads: int = 16,
        feedforward: int = 2048,
        controller_width: int = 960,
        cycles: int = 4,
        update_metric: str = "euclidean",
    ) -> None:
        super().__init__()
        if width < 48:
            raise ConflictCompilerError(
                "preregistered conflict compiler width differs"
            )
        self.witness = ProofCarryingWitnessCompiler(
            width=width,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            heads=heads,
            feedforward=feedforward,
            external_feature_width=external_feature_width,
            projector=DirectEvidenceProjector(),
        )
        record_width = 32
        self.claim_adapter = ConflictClaimAdapter(
            record_width=record_width
        )
        self.path_controller = PhysicalKeyPathController()
        self.source_reentry = ConflictSourceReentry(
            record_width=record_width,
            external_feature_width=external_feature_width,
        )
        self.revision = ConflictGatedReentrantRevision(
            record_width=record_width,
            controller_width=controller_width,
            cycles=cycles,
            update_metric=update_metric,
        )

    @property
    def external_feature_width(self) -> int:
        return self.witness.external_feature_width

    @property
    def projector(self) -> DirectEvidenceProjector:
        """Expose the first-pass projector for trainer compatibility only."""

        return self.witness.projector

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _apply_path_controller(
        self,
        batch: WitnessCompilerBatch,
        witness: WitnessCompilerOutput,
        *,
        mode: str,
    ) -> tuple[WitnessCompilerOutput, PhysicalKeyPathControllerResult]:
        result = self.path_controller(
            witness,
            batch.record_valid,
            mode=mode,
        )
        assignment = project_key_assignment_logits(
            slot_assignment_logits=result.raw_key_assignment_logits,
            source_unique_key_valid=witness.unique_key_valid,
            sinkhorn_iterations=self.witness.key_sinkhorn_iterations,
            straight_through=False,
        )
        relation_evidence = assemble_relation_evidence(
            record_type_logits=witness.record_type_logits,
            occurrence_role_logits=witness.occurrence_role_logits,
            answer_logits=witness.answer_logits,
            occurrence_valid=batch.pointer.occurrence_valid,
            occurrence_to_record=batch.occurrence_to_record,
            occurrence_to_unique=batch.pointer.occurrence_to_unique,
            source_unique_key_valid=witness.unique_key_valid,
            key_assignment_logits=assignment,
        )
        projection = self.projector(
            relation_evidence.transition_logits,
            relation_evidence.observer_logits,
            straight_through=False,
        )
        return (
            replace(
                witness,
                projection=projection,
                relation_evidence=relation_evidence,
                key_assignment_logits=assignment,
                raw_key_assignment_logits=(
                    result.raw_key_assignment_logits
                ),
                projector_auxiliary=None,
            ),
            result,
        )

    def forward(
        self,
        batch: WitnessCompilerBatch,
        *,
        straight_through: bool = False,
        frozen_byte_features: torch.Tensor | None = None,
        routing_mode: str = "causal",
        source_reentry_mode: str = "causal",
        path_controller_mode: str = "causal",
        claim_calibration_mode: str = "causal",
        update_metric: str | None = None,
    ) -> ConflictCompilerOutput:
        if straight_through:
            raise ConflictCompilerError(
                "conflict compiler forbids solver-backed straight-through"
            )
        if frozen_byte_features is None:
            raise ConflictCompilerError(
                "two-pass conflict compiler requires frozen byte features"
            )
        frozen_byte_features = frozen_byte_features.detach()
        first_witness = self.witness(
            batch,
            straight_through=straight_through,
            frozen_byte_features=frozen_byte_features,
        )
        first_witness, first_path_controller = (
            self._apply_path_controller(
                batch,
                first_witness,
                mode=path_controller_mode,
            )
        )
        first_record_features = record_features_from_witness(first_witness)
        (
            first_claim_logits,
            first_closure_claim_logits,
            first_claim_incidence,
            first_closure_incidence,
        ) = self.claim_adapter(
            first_witness,
            record_features=first_record_features,
            record_valid=batch.record_valid,
            mode=claim_calibration_mode,
        )
        first_revision_batch = ConflictRevisionBatch(
            transition_logits=(
                first_witness.relation_evidence.transition_logits
            ),
            observer_logits=first_witness.relation_evidence.observer_logits,
            claim_logits=first_claim_logits,
            closure_claim_logits=first_closure_claim_logits,
            claim_incidence=first_claim_incidence,
            closure_incidence=first_closure_incidence,
            record_features=first_record_features,
            record_valid=batch.record_valid,
        )
        first_revision = self.revision(
            first_revision_batch,
            routing_mode=routing_mode,
            update_metric=update_metric,
        )
        source_feedback = self.source_reentry(
            batch,
            first_revision_batch,
            first_revision,
            byte_count=int(frozen_byte_features.shape[1]),
            mode=source_reentry_mode,
        )
        witness = self.witness(
            batch,
            straight_through=straight_through,
            frozen_byte_features=frozen_byte_features + source_feedback,
        )
        witness, path_controller = self._apply_path_controller(
            batch,
            witness,
            mode=path_controller_mode,
        )
        record_features = record_features_from_witness(witness)
        (
            claim_logits,
            closure_claim_logits,
            claim_incidence,
            closure_incidence,
        ) = self.claim_adapter(
            witness,
            record_features=record_features,
            record_valid=batch.record_valid,
            mode=claim_calibration_mode,
        )
        revision_batch = ConflictRevisionBatch(
            transition_logits=witness.relation_evidence.transition_logits,
            observer_logits=witness.relation_evidence.observer_logits,
            claim_logits=claim_logits,
            closure_claim_logits=closure_claim_logits,
            claim_incidence=claim_incidence,
            closure_incidence=closure_incidence,
            record_features=record_features,
            record_valid=batch.record_valid,
        )
        revision = self.revision(
            revision_batch,
            routing_mode=routing_mode,
            update_metric=update_metric,
        )
        return ConflictCompilerOutput(
            first_witness=first_witness,
            first_path_controller=first_path_controller,
            first_revision_batch=first_revision_batch,
            first_revision=first_revision,
            source_feedback=source_feedback,
            witness=witness,
            path_controller=path_controller,
            revision_batch=revision_batch,
            revision=revision,
        )

    @torch.no_grad()
    def seal(
        self,
        output: ConflictCompilerOutput,
    ) -> SealedConflictMachine:
        machine = self.revision.seal(output.revision)
        keys = _hard_assign_keys_without_solver(output.witness)
        return SealedConflictMachine(machine=machine, keys=keys)


__all__ = [
    "ConflictClaimAdapter",
    "ConflictCompilerError",
    "ConflictCompilerOutput",
    "ConflictProofCarryingCompiler",
    "DirectEvidenceProjector",
    "ConflictSourceReentry",
    "SOURCE_REENTRY_MODES",
    "SealedConflictMachine",
    "behaviorally_close_record_claims",
    "record_features_from_witness",
]
