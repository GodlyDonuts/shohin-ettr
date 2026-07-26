"""Causal, architecture-native objectives for Shohin ETTR.

The loss sees only next-token predictions, sealed typed packets, generic
transaction policies, and declared same-world variant alignments. It has no
task-family parser, semantic executor, answer oracle, or future-step target
path. All per-batch checks and receipt counts stay on device.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from endogenous_typed_theory_reactor import (
    ReactorTrace,
    TRANSACTION_COUNT,
    TypedTheoryState,
)


OBJECTIVE_SCHEMA = "shohin-ettr-composite-objective-v2"


class ETTRObjectiveError(ValueError):
    """An objective geometry or static configuration contract failed."""


def _async_assert(condition: torch.Tensor, message: str) -> None:
    """Assert tensor data without copying a predicate to the host."""

    torch._assert_async(condition, message)  # noqa: SLF001


def _tensor(
    value: object,
    *,
    name: str,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
    dtype: torch.dtype | None = None,
    floating: bool | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ETTRObjectiveError(f"{name} must be a tensor")
    if ndim is not None and value.ndim != ndim:
        raise ETTRObjectiveError(f"{name} rank differs")
    if shape is not None and value.shape != shape:
        raise ETTRObjectiveError(f"{name} shape differs")
    if dtype is not None and value.dtype != dtype:
        raise ETTRObjectiveError(f"{name} dtype differs")
    if floating is True and not value.is_floating_point():
        raise ETTRObjectiveError(f"{name} must be floating point")
    if floating is False and value.is_floating_point():
        raise ETTRObjectiveError(f"{name} must be integral or boolean")
    if value.is_floating_point():
        _async_assert(
            torch.isfinite(value).all(),
            f"{name} contains non-finite values",
        )
    return value


def _same_device(
    tensors: tuple[torch.Tensor, ...],
    *,
    name: str,
) -> torch.device:
    if not tensors:
        raise ETTRObjectiveError(f"{name} contains no tensors")
    device = tensors[0].device
    if any(value.device != device for value in tensors[1:]):
        raise ETTRObjectiveError(f"{name} tensors span devices")
    return device


def _probability(value: torch.Tensor, *, name: str) -> None:
    _async_assert(
        ((value >= 0.0) & (value <= 1.0)).all(),
        f"{name} leaves the probability interval",
    )


def _simplex(value: torch.Tensor, *, name: str) -> None:
    _probability(value, name=name)
    tolerance = 5e-3 if value.dtype in (torch.bfloat16, torch.float16) else 1e-4
    _async_assert(
        (value.sum(-1).float() - 1.0).abs().le(tolerance).all(),
        f"{name} is not a categorical simplex",
    )


def _binary(value: torch.Tensor, *, name: str) -> None:
    _async_assert(
        ((value == 0) | (value == 1)).all(),
        f"{name} is not binary",
    )


def _nonnegative(value: torch.Tensor, *, name: str) -> None:
    _async_assert((value >= 0).all(), f"{name} contains a negative label")


@dataclass(frozen=True, slots=True)
class ETTRObjectiveWeights:
    """Explicit family weights; zero is reserved for declared ablations."""

    token_lm: float = 1.0
    packet: float = 1.0
    world_intervention: float = 1.0
    command_intervention: float = 1.0
    world_query_binding: float = 1.0
    command_query_binding: float = 1.0
    transaction: float = 1.0
    equivariance: float = 0.25
    commit_halt: float = 0.5
    sparsity: float = 0.01
    anti_bypass: float = 0.1

    def __post_init__(self) -> None:
        values = tuple(getattr(self, field.name) for field in fields(self))
        if any(
            not isinstance(value, float) or not math.isfinite(value) or value < 0.0
            for value in values
        ):
            raise ETTRObjectiveError("weights must be finite nonnegative floats")
        if not any(value > 0.0 for value in values):
            raise ETTRObjectiveError("at least one weight must be positive")

    def items(self) -> tuple[tuple[str, float], ...]:
        return tuple((field.name, getattr(self, field.name)) for field in fields(self))


@dataclass(frozen=True, slots=True)
class ETTRObjectiveConfig:
    """Frozen class geometry and sparse packet budgets."""

    vocab_size: int
    num_slots: int = 64
    num_types: int = 8
    num_relations: int = 16
    num_value_codes: int = 256
    transaction_count: int = TRANSACTION_COUNT
    active_slot_budget: int = 64
    relation_edge_budget: int = 256
    ignore_index: int = -100
    probability_epsilon: float = 1e-6
    query_binding_margin: float = 1.0
    causal_lm_shift: int = 1
    require_equivariance_pairs: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.vocab_size,
            self.num_slots,
            self.num_types,
            self.num_relations,
            self.num_value_codes,
            self.transaction_count,
            self.active_slot_budget,
            self.relation_edge_budget,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in positive
        ):
            raise ETTRObjectiveError(
                "objective geometry must contain positive integers"
            )
        if self.transaction_count != TRANSACTION_COUNT:
            raise ETTRObjectiveError(
                "the frozen ETTR objective transaction count differs"
            )
        if self.active_slot_budget > self.num_slots:
            raise ETTRObjectiveError("active-slot budget exceeds packet capacity")
        capacity = self.num_relations * self.num_slots * self.num_slots
        if self.relation_edge_budget > capacity:
            raise ETTRObjectiveError("relation-edge budget exceeds packet capacity")
        if not isinstance(self.ignore_index, int) or isinstance(
            self.ignore_index, bool
        ):
            raise ETTRObjectiveError("ignore index must be an integer")
        if (
            not isinstance(self.probability_epsilon, float)
            or not math.isfinite(self.probability_epsilon)
            or not 0.0 < self.probability_epsilon < 0.5
        ):
            raise ETTRObjectiveError("probability epsilon must be between zero and 0.5")
        if (
            not isinstance(self.query_binding_margin, float)
            or not math.isfinite(self.query_binding_margin)
            or self.query_binding_margin <= 0.0
        ):
            raise ETTRObjectiveError(
                "query-binding margin must be a positive finite float"
            )
        if self.causal_lm_shift != 1:
            raise ETTRObjectiveError(
                "token supervision must use one-token causal shift"
            )
        if not isinstance(self.require_equivariance_pairs, bool):
            raise ETTRObjectiveError("equivariance requirement must be boolean")


@dataclass(frozen=True, slots=True)
class ETTRTokenTargets:
    """Token labels plus explicit independent-stream reset positions."""

    token_ids: torch.Tensor
    mask: torch.Tensor
    reset_mask: torch.Tensor

    def __post_init__(self) -> None:
        token_ids = _tensor(
            self.token_ids,
            name="token_targets.token_ids",
            ndim=2,
            dtype=torch.long,
        )
        mask = _tensor(
            self.mask,
            name="token_targets.mask",
            shape=token_ids.shape,
            dtype=torch.bool,
        )
        reset_mask = _tensor(
            self.reset_mask,
            name="token_targets.reset_mask",
            shape=token_ids.shape,
            dtype=torch.bool,
        )
        _same_device(
            (token_ids, mask, reset_mask),
            name="token targets",
        )
        # Each causal segment is independently right padded. Concatenating
        # WORLD, COMMAND, and QUERY therefore permits a 0 -> 1 transition only
        # where the new valid token explicitly resets transformer context.
        validity_rises = mask[:, 1:] & ~mask[:, :-1]
        _async_assert(
            (~validity_rises | reset_mask[:, 1:]).all(),
            "token validity may restart only at an explicit reset",
        )
        _async_assert(
            reset_mask[:, 0].all(),
            "every token row must begin with an explicit reset",
        )
        _async_assert(
            (~reset_mask | mask).all(),
            "token reset lies outside the valid stream",
        )


@dataclass(frozen=True, slots=True)
class ETTRPacketTargets:
    """Sealed packet labels with explicit support masks.

    Shapes are ``value_code/type_index/active/root/slot_mask=[B,S]``,
    ``relations/relation_mask=[B,R,S,S]``, and
    ``committed/halted=[B]``.
    """

    value_code: torch.Tensor
    type_index: torch.Tensor
    relations: torch.Tensor
    active: torch.Tensor
    root: torch.Tensor
    committed: torch.Tensor
    halted: torch.Tensor
    slot_mask: torch.Tensor
    relation_mask: torch.Tensor

    def __post_init__(self) -> None:
        active = _tensor(
            self.active,
            name="packet_targets.active",
            ndim=2,
            dtype=torch.bool,
        )
        batch, slots = active.shape
        value_code = _tensor(
            self.value_code,
            name="packet_targets.value_code",
            shape=(batch, slots),
            dtype=torch.long,
        )
        type_index = _tensor(
            self.type_index,
            name="packet_targets.type_index",
            shape=(batch, slots),
            dtype=torch.long,
        )
        root = _tensor(
            self.root,
            name="packet_targets.root",
            shape=(batch, slots),
            dtype=torch.bool,
        )
        committed = _tensor(
            self.committed,
            name="packet_targets.committed",
            shape=(batch,),
            dtype=torch.bool,
        )
        halted = _tensor(
            self.halted,
            name="packet_targets.halted",
            shape=(batch,),
            dtype=torch.bool,
        )
        slot_mask = _tensor(
            self.slot_mask,
            name="packet_targets.slot_mask",
            shape=(batch, slots),
            dtype=torch.bool,
        )
        relations = _tensor(
            self.relations,
            name="packet_targets.relations",
            ndim=4,
            dtype=torch.bool,
        )
        if relations.shape[0] != batch or relations.shape[2:] != (slots, slots):
            raise ETTRObjectiveError("packet target relation geometry differs")
        relation_mask = _tensor(
            self.relation_mask,
            name="packet_targets.relation_mask",
            shape=relations.shape,
            dtype=torch.bool,
        )
        _same_device(
            (
                value_code,
                type_index,
                relations,
                active,
                root,
                committed,
                halted,
                slot_mask,
                relation_mask,
            ),
            name="packet targets",
        )
        _nonnegative(value_code, name="packet_targets.value_code")
        _nonnegative(type_index, name="packet_targets.type_index")
        _async_assert(
            (root <= active).all(),
            "packet target root leaves active slots",
        )
        _async_assert(
            root.sum(-1).le(1).all(),
            "packet target has multiple roots",
        )
        pair_active = active[:, None, :, None] & active[:, None, None, :]
        _async_assert(
            (relations <= pair_active).all(),
            "packet target relation touches an inactive slot",
        )
        _async_assert(
            slot_mask.any(),
            "packet target slot mask contains no support",
        )
        _async_assert(
            relation_mask.any(),
            "packet target relation mask contains no support",
        )


@dataclass(frozen=True, slots=True)
class ETTRTransactionTargets:
    """Generic policy labels.

    Every tensor is ``[B,K]``. Categorical labels are int64; committed,
    halted, and step_mask are boolean.
    """

    opcode: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    relation: torch.Tensor
    type_index: torch.Tensor
    value_code: torch.Tensor
    committed: torch.Tensor
    halted: torch.Tensor
    step_mask: torch.Tensor

    def __post_init__(self) -> None:
        opcode = _tensor(
            self.opcode,
            name="transaction_targets.opcode",
            ndim=2,
            dtype=torch.long,
        )
        shape = opcode.shape
        categorical = (opcode,)
        for name in (
            "source",
            "target",
            "relation",
            "type_index",
            "value_code",
        ):
            categorical += (
                _tensor(
                    getattr(self, name),
                    name=f"transaction_targets.{name}",
                    shape=shape,
                    dtype=torch.long,
                ),
            )
        booleans: tuple[torch.Tensor, ...] = ()
        for name in ("committed", "halted", "step_mask"):
            booleans += (
                _tensor(
                    getattr(self, name),
                    name=f"transaction_targets.{name}",
                    shape=shape,
                    dtype=torch.bool,
                ),
            )
        _same_device(categorical + booleans, name="transaction targets")
        for name, value in zip(
            (
                "opcode",
                "source",
                "target",
                "relation",
                "type_index",
                "value_code",
            ),
            categorical,
            strict=True,
        ):
            _nonnegative(value, name=f"transaction_targets.{name}")
        _async_assert(
            self.step_mask.any(),
            "transaction target mask contains no support",
        )
        _async_assert(
            self.step_mask.any(dim=1).all(),
            "every transaction target row must contain support",
        )
        _async_assert(
            (self.opcode < TRANSACTION_COUNT).all(),
            "transaction target opcode leaves the frozen range",
        )
        _async_assert(
            (~self.step_mask[:, 1:] | self.step_mask[:, :-1]).all(),
            "transaction target mask must be right padded",
        )
        for name in ("committed", "halted"):
            status = getattr(self, name)
            _async_assert(
                (~status[:, :-1] | status[:, 1:] | ~self.step_mask[:, 1:]).all(),
                f"transaction target {name} is not monotone",
            )


@dataclass(frozen=True, slots=True)
class ETTRTransactionPredictions:
    """Policy distributions and post-step traces emitted by ETTR.

    Categorical tensors are ``[B,K,C]``; active is ``[B,K,S]`` and status
    tensors are ``[B,K]``.
    """

    opcode: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    relation: torch.Tensor
    type_index: torch.Tensor
    value_code: torch.Tensor
    active: torch.Tensor
    committed: torch.Tensor
    halted: torch.Tensor

    def __post_init__(self) -> None:
        opcode = _tensor(
            self.opcode,
            name="transactions.opcode",
            ndim=3,
            floating=True,
        )
        batch, steps, _ = opcode.shape
        tensors = (opcode,)
        for name in (
            "source",
            "target",
            "relation",
            "type_index",
            "value_code",
        ):
            value = _tensor(
                getattr(self, name),
                name=f"transactions.{name}",
                ndim=3,
                floating=True,
            )
            if value.shape[:2] != (batch, steps):
                raise ETTRObjectiveError(f"transactions.{name} prefix shape differs")
            tensors += (value,)
        active = _tensor(
            self.active,
            name="transactions.active",
            ndim=3,
            floating=True,
        )
        if active.shape[:2] != (batch, steps):
            raise ETTRObjectiveError("transactions.active prefix shape differs")
        tensors += (active,)
        for name in ("committed", "halted"):
            tensors += (
                _tensor(
                    getattr(self, name),
                    name=f"transactions.{name}",
                    shape=(batch, steps),
                    floating=True,
                ),
            )
        _same_device(tensors, name="transaction predictions")
        for name, value in zip(
            (
                "opcode",
                "source",
                "target",
                "relation",
                "type_index",
                "value_code",
            ),
            tensors[:6],
            strict=True,
        ):
            _simplex(value, name=f"transactions.{name}")
        for name, value in zip(
            ("active", "committed", "halted"),
            tensors[6:],
            strict=True,
        ):
            _probability(value, name=f"transactions.{name}")

    @classmethod
    def from_reactor_trace(
        cls,
        trace: ReactorTrace,
    ) -> ETTRTransactionPredictions:
        """Preserve every generic policy head from a native reactor trace."""

        if not isinstance(trace, ReactorTrace):
            raise ETTRObjectiveError("reactor trace type differs")
        return cls(
            opcode=trace.opcode,
            source=trace.source,
            target=trace.target,
            relation=trace.relation,
            type_index=trace.type_index,
            value_code=trace.value_code,
            active=trace.active,
            committed=trace.committed,
            halted=trace.halted,
        )


@dataclass(frozen=True, slots=True)
class ETTRVariantAlignment:
    """Right-to-left coordinate maps for invariant same-world variants."""

    left_index: torch.Tensor
    right_index: torch.Tensor
    slot_permutation: torch.Tensor
    type_permutation: torch.Tensor
    relation_permutation: torch.Tensor
    value_permutation: torch.Tensor
    slot_mask: torch.Tensor
    relation_mask: torch.Tensor
    step_mask: torch.Tensor

    def __post_init__(self) -> None:
        left = _tensor(
            self.left_index,
            name="alignment.left_index",
            ndim=1,
            dtype=torch.long,
        )
        pairs = left.shape[0]
        right = _tensor(
            self.right_index,
            name="alignment.right_index",
            shape=(pairs,),
            dtype=torch.long,
        )
        matrices: tuple[torch.Tensor, ...] = ()
        for name in (
            "slot_permutation",
            "type_permutation",
            "relation_permutation",
            "value_permutation",
        ):
            value = _tensor(
                getattr(self, name),
                name=f"alignment.{name}",
                ndim=2,
                dtype=torch.long,
            )
            if value.shape[0] != pairs:
                raise ETTRObjectiveError(f"alignment.{name} pair count differs")
            expected = torch.arange(
                value.shape[1],
                device=value.device,
            ).expand_as(value)
            _async_assert(
                (value.sort(-1).values == expected).all(),
                f"alignment.{name} is not a row-wise permutation",
            )
            matrices += (value,)
        slots = matrices[0].shape[1]
        relations = matrices[2].shape[1]
        slot_mask = _tensor(
            self.slot_mask,
            name="alignment.slot_mask",
            shape=(pairs, slots),
            dtype=torch.bool,
        )
        relation_mask = _tensor(
            self.relation_mask,
            name="alignment.relation_mask",
            shape=(pairs, relations, slots, slots),
            dtype=torch.bool,
        )
        step_mask = _tensor(
            self.step_mask,
            name="alignment.step_mask",
            ndim=2,
            dtype=torch.bool,
        )
        if step_mask.shape[0] != pairs:
            raise ETTRObjectiveError("alignment.step_mask pair count differs")
        _same_device(
            (left, right, *matrices, slot_mask, relation_mask, step_mask),
            name="variant alignment",
        )
        _nonnegative(left, name="alignment.left_index")
        _nonnegative(right, name="alignment.right_index")
        _async_assert(
            slot_mask.any() & relation_mask.any() & step_mask.any(),
            "alignment masks must each contain support",
        )


@dataclass(frozen=True, slots=True)
class ETTRCausalQueryPair:
    """Matched query logits with distinct immutable factual targets."""

    correct_logits: torch.Tensor
    foil_logits: torch.Tensor
    correct_target: torch.Tensor
    foil_target: torch.Tensor

    def __post_init__(self) -> None:
        correct = _tensor(
            self.correct_logits,
            name="causal_query.correct_logits",
            ndim=2,
            floating=True,
        )
        foil = _tensor(
            self.foil_logits,
            name="causal_query.foil_logits",
            shape=correct.shape,
            floating=True,
        )
        pairs, vocab = correct.shape
        if pairs < 1 or vocab < 2:
            raise ETTRObjectiveError("causal-query pair geometry differs")
        correct_target = _tensor(
            self.correct_target,
            name="causal_query.correct_target",
            shape=(pairs,),
            dtype=torch.long,
        )
        foil_target = _tensor(
            self.foil_target,
            name="causal_query.foil_target",
            shape=(pairs,),
            dtype=torch.long,
        )
        _same_device(
            (correct, foil, correct_target, foil_target),
            name="causal-query pair",
        )
        _async_assert(
            (
                (correct_target >= 0)
                & (correct_target < vocab)
                & (foil_target >= 0)
                & (foil_target < vocab)
            ).all(),
            "causal-query target leaves vocabulary range",
        )
        _async_assert(
            correct_target.ne(foil_target).all(),
            "causal-query targets must be distinct",
        )


@dataclass(frozen=True, slots=True)
class ETTRObjectiveBatch:
    """All candidate-visible predictions and offline architecture labels."""

    token_logits: torch.Tensor
    token_targets: ETTRTokenTargets
    packet_prediction: TypedTheoryState
    packet_targets: ETTRPacketTargets
    terminal_packet_prediction: TypedTheoryState
    terminal_packet_targets: ETTRPacketTargets
    world_intervention_prediction: TypedTheoryState
    world_intervention_targets: ETTRPacketTargets
    world_intervention_transactions: ETTRTransactionPredictions
    world_intervention_transaction_targets: ETTRTransactionTargets
    command_intervention_prediction: TypedTheoryState
    command_intervention_targets: ETTRPacketTargets
    command_intervention_transactions: ETTRTransactionPredictions
    command_intervention_transaction_targets: ETTRTransactionTargets
    world_query_binding: ETTRCausalQueryPair
    command_query_binding: ETTRCausalQueryPair
    transactions: ETTRTransactionPredictions
    transaction_targets: ETTRTransactionTargets
    initial_committed: torch.Tensor
    initial_halted: torch.Tensor
    equivariance: ETTRVariantAlignment | None


@dataclass(frozen=True, slots=True)
class ETTRObjectiveReceipt:
    """Device-resident support counts; reading them is explicitly optional."""

    schema: str
    batch_size: int
    sequence_tokens: int
    lm_target_tokens: torch.Tensor
    supervised_packet_slots: torch.Tensor
    supervised_relation_cells: torch.Tensor
    supervised_world_intervention_slots: torch.Tensor
    supervised_world_intervention_relation_cells: torch.Tensor
    supervised_world_intervention_transaction_decisions: torch.Tensor
    supervised_command_intervention_slots: torch.Tensor
    supervised_command_intervention_relation_cells: torch.Tensor
    supervised_command_intervention_transaction_decisions: torch.Tensor
    supervised_world_query_pairs: torch.Tensor
    supervised_command_query_pairs: torch.Tensor
    world_query_margin_satisfied: torch.Tensor
    command_query_margin_satisfied: torch.Tensor
    supervised_transaction_steps: torch.Tensor
    supervised_transaction_decisions: torch.Tensor
    supervised_opcode_decisions: torch.Tensor
    supervised_source_decisions: torch.Tensor
    supervised_target_decisions: torch.Tensor
    supervised_relation_decisions: torch.Tensor
    supervised_type_decisions: torch.Tensor
    supervised_value_code_decisions: torch.Tensor
    equivariance_pairs: int
    equivariance_packet_cells: torch.Tensor
    equivariance_transaction_cells: torch.Tensor
    causal_lm_shift: int
    weights: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if self.schema != OBJECTIVE_SCHEMA:
            raise ETTRObjectiveError("receipt schema differs")
        if (
            not isinstance(self.batch_size, int)
            or self.batch_size <= 0
            or not isinstance(self.sequence_tokens, int)
            or self.sequence_tokens <= 1
            or not isinstance(self.equivariance_pairs, int)
            or self.equivariance_pairs < 0
            or self.causal_lm_shift != 1
            or not self.weights
        ):
            raise ETTRObjectiveError("receipt static metadata differs")
        counts = tuple(
            getattr(self, name)
            for name in (
                "lm_target_tokens",
                "supervised_packet_slots",
                "supervised_relation_cells",
                "supervised_world_intervention_slots",
                "supervised_world_intervention_relation_cells",
                "supervised_world_intervention_transaction_decisions",
                "supervised_command_intervention_slots",
                "supervised_command_intervention_relation_cells",
                "supervised_command_intervention_transaction_decisions",
                "supervised_world_query_pairs",
                "supervised_command_query_pairs",
                "world_query_margin_satisfied",
                "command_query_margin_satisfied",
                "supervised_transaction_steps",
                "supervised_transaction_decisions",
                "supervised_opcode_decisions",
                "supervised_source_decisions",
                "supervised_target_decisions",
                "supervised_relation_decisions",
                "supervised_type_decisions",
                "supervised_value_code_decisions",
                "equivariance_packet_cells",
                "equivariance_transaction_cells",
            )
        )
        for name, value in zip(
            (
                "lm_target_tokens",
                "supervised_packet_slots",
                "supervised_relation_cells",
                "supervised_world_intervention_slots",
                "supervised_world_intervention_relation_cells",
                "supervised_world_intervention_transaction_decisions",
                "supervised_command_intervention_slots",
                "supervised_command_intervention_relation_cells",
                "supervised_command_intervention_transaction_decisions",
                "supervised_world_query_pairs",
                "supervised_command_query_pairs",
                "world_query_margin_satisfied",
                "command_query_margin_satisfied",
                "supervised_transaction_steps",
                "supervised_transaction_decisions",
                "supervised_opcode_decisions",
                "supervised_source_decisions",
                "supervised_target_decisions",
                "supervised_relation_decisions",
                "supervised_type_decisions",
                "supervised_value_code_decisions",
                "equivariance_packet_cells",
                "equivariance_transaction_cells",
            ),
            counts,
            strict=True,
        ):
            _tensor(
                value,
                name=f"receipt.{name}",
                shape=(),
                dtype=torch.int64,
            )
            _async_assert(value >= 0, f"receipt.{name} is negative")
        _same_device(counts, name="receipt counts")
        _async_assert(
            self.supervised_world_query_pairs.eq(self.batch_size)
            & self.supervised_command_query_pairs.eq(self.batch_size),
            "causal-query receipt support must equal factual batch size",
        )


@dataclass(frozen=True, slots=True)
class ETTRCompositeLoss:
    """Differentiable breakdown and device-resident structural receipt."""

    total: torch.Tensor
    token_lm: torch.Tensor
    packet: torch.Tensor
    world_intervention: torch.Tensor
    command_intervention: torch.Tensor
    world_query_binding: torch.Tensor
    command_query_binding: torch.Tensor
    transaction: torch.Tensor
    equivariance: torch.Tensor
    commit_halt: torch.Tensor
    sparsity: torch.Tensor
    anti_bypass: torch.Tensor
    receipt: ETTRObjectiveReceipt

    def __post_init__(self) -> None:
        losses: tuple[torch.Tensor, ...] = ()
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "receipt":
                if not isinstance(value, ETTRObjectiveReceipt):
                    raise ETTRObjectiveError("loss receipt differs")
                continue
            tensor = _tensor(
                value,
                name=f"loss.{field.name}",
                shape=(),
                floating=True,
            )
            losses += (tensor,)
        _same_device(losses, name="objective losses")


@dataclass(frozen=True, slots=True)
class _Geometry:
    batch: int
    tokens: int
    steps: int
    lm_mask: torch.Tensor
    operand_masks: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class _LossCount:
    numerator: torch.Tensor
    count: torch.Tensor

    @property
    def mean(self) -> torch.Tensor:
        return self.numerator / self.count.to(self.numerator.dtype).clamp_min(1)


def _validate_state(
    state: TypedTheoryState,
    *,
    config: ETTRObjectiveConfig,
    batch: int,
    name: str,
) -> tuple[torch.Tensor, ...]:
    if not isinstance(state, TypedTheoryState):
        raise ETTRObjectiveError(f"{name} must be a TypedTheoryState")
    expected = {
        "value_probabilities": (
            batch,
            config.num_slots,
            config.num_value_codes,
        ),
        "type_probabilities": (
            batch,
            config.num_slots,
            config.num_types,
        ),
        "relations": (
            batch,
            config.num_relations,
            config.num_slots,
            config.num_slots,
        ),
        "active": (batch, config.num_slots),
        "root": (batch, config.num_slots),
        "committed": (batch,),
        "halted": (batch,),
    }
    tensors: tuple[torch.Tensor, ...] = ()
    for field in fields(state):
        value = getattr(state, field.name)
        if field.name == "step":
            if not isinstance(value, int) or value < 0:
                raise ETTRObjectiveError("packet prediction step differs")
            continue
        tensor = _tensor(
            value,
            name=f"{name}.{field.name}",
            shape=expected[field.name],
            floating=True,
        )
        _probability(
            tensor,
            name=f"{name}.{field.name}",
        )
        tensors += (tensor,)
    _same_device(tensors, name=name)
    return tensors


def _range_check(
    value: torch.Tensor,
    upper: int,
    *,
    name: str,
) -> None:
    _async_assert(
        ((value >= 0) & (value < upper)).all(),
        f"{name} leaves its class range",
    )


def _operand_masks(
    targets: ETTRTransactionTargets,
) -> tuple[torch.Tensor, ...]:
    valid = targets.step_mask
    opcode = targets.opcode
    source = valid & (
        (opcode == 0)
        | (opcode == 1)
        | (opcode == 2)
        | (opcode == 3)
        | (opcode == 4)
        | (opcode == 5)
    )
    target = valid & ((opcode == 3) | (opcode == 4))
    relation = target
    type_index = valid & (opcode == 0)
    value_code = valid & ((opcode == 0) | (opcode == 1))
    return valid, source, target, relation, type_index, value_code


def _validate_batch(
    batch: ETTRObjectiveBatch,
    config: ETTRObjectiveConfig,
) -> _Geometry:
    if not isinstance(batch, ETTRObjectiveBatch):
        raise ETTRObjectiveError("objective batch type differs")
    logits = _tensor(
        batch.token_logits,
        name="token_logits",
        ndim=3,
        floating=True,
    )
    rows, tokens, vocab = logits.shape
    if tokens < 2 or vocab != config.vocab_size:
        raise ETTRObjectiveError("token logit geometry differs")
    if not isinstance(batch.token_targets, ETTRTokenTargets):
        raise ETTRObjectiveError("token target type differs")
    if batch.token_targets.token_ids.shape != (rows, tokens):
        raise ETTRObjectiveError("token target geometry differs")
    token_valid = (batch.token_targets.token_ids == config.ignore_index) | (
        (batch.token_targets.token_ids >= 0)
        & (batch.token_targets.token_ids < config.vocab_size)
    )
    _async_assert(
        token_valid.all(),
        "token target leaves vocabulary range",
    )
    lm_mask = (
        batch.token_targets.mask[:, :-1]
        & batch.token_targets.mask[:, 1:]
        & ~batch.token_targets.reset_mask[:, 1:]
        & batch.token_targets.token_ids[:, 1:].ne(config.ignore_index)
    )
    _async_assert(
        lm_mask.any(),
        "causal token target mask contains no support",
    )

    state_tensors: tuple[torch.Tensor, ...] = ()
    for name, state in (
        ("packet_prediction", batch.packet_prediction),
        ("terminal_packet_prediction", batch.terminal_packet_prediction),
        (
            "world_intervention_prediction",
            batch.world_intervention_prediction,
        ),
        (
            "command_intervention_prediction",
            batch.command_intervention_prediction,
        ),
    ):
        state_tensors += _validate_state(
            state,
            config=config,
            batch=rows,
            name=name,
        )
    for name, packet in (
        ("packet_targets", batch.packet_targets),
        ("terminal_packet_targets", batch.terminal_packet_targets),
        ("world_intervention_targets", batch.world_intervention_targets),
        ("command_intervention_targets", batch.command_intervention_targets),
    ):
        if not isinstance(packet, ETTRPacketTargets):
            raise ETTRObjectiveError(f"{name} type differs")
        if packet.active.shape != (
            rows,
            config.num_slots,
        ) or packet.relations.shape != (
            rows,
            config.num_relations,
            config.num_slots,
            config.num_slots,
        ):
            raise ETTRObjectiveError(f"{name} config geometry differs")
        _range_check(
            packet.value_code,
            config.num_value_codes,
            name=f"{name}.value_code",
        )
        _range_check(
            packet.type_index,
            config.num_types,
            name=f"{name}.type_index",
        )

    if not isinstance(
        batch.transactions,
        ETTRTransactionPredictions,
    ):
        raise ETTRObjectiveError("transaction prediction type differs")
    transactions = batch.transactions
    steps = transactions.opcode.shape[1]
    expected_prediction_shapes = {
        "opcode": (rows, steps, config.transaction_count),
        "source": (rows, steps, config.num_slots),
        "target": (rows, steps, config.num_slots),
        "relation": (rows, steps, config.num_relations),
        "type_index": (rows, steps, config.num_types),
        "value_code": (rows, steps, config.num_value_codes),
        "active": (rows, steps, config.num_slots),
        "committed": (rows, steps),
        "halted": (rows, steps),
    }
    for name, shape in expected_prediction_shapes.items():
        if getattr(transactions, name).shape != shape:
            raise ETTRObjectiveError(f"transactions.{name} config geometry differs")
    if not isinstance(
        batch.transaction_targets,
        ETTRTransactionTargets,
    ):
        raise ETTRObjectiveError("transaction target type differs")
    targets = batch.transaction_targets
    if targets.opcode.shape != (rows, steps):
        raise ETTRObjectiveError("transaction target sequence geometry differs")
    for name, upper in (
        ("opcode", config.transaction_count),
        ("source", config.num_slots),
        ("target", config.num_slots),
        ("relation", config.num_relations),
        ("type_index", config.num_types),
        ("value_code", config.num_value_codes),
    ):
        _range_check(
            getattr(targets, name),
            upper,
            name=f"transaction_targets.{name}",
        )
    intervention_transaction_tensors: tuple[torch.Tensor, ...] = ()
    for prefix, prediction, intervention_targets in (
        (
            "world_intervention",
            batch.world_intervention_transactions,
            batch.world_intervention_transaction_targets,
        ),
        (
            "command_intervention",
            batch.command_intervention_transactions,
            batch.command_intervention_transaction_targets,
        ),
    ):
        if not isinstance(prediction, ETTRTransactionPredictions):
            raise ETTRObjectiveError(f"{prefix} transaction prediction differs")
        if not isinstance(intervention_targets, ETTRTransactionTargets):
            raise ETTRObjectiveError(f"{prefix} transaction target differs")
        for name, shape in expected_prediction_shapes.items():
            if getattr(prediction, name).shape != shape:
                raise ETTRObjectiveError(
                    f"{prefix} transactions.{name} geometry differs"
                )
        if intervention_targets.opcode.shape != (rows, steps):
            raise ETTRObjectiveError(
                f"{prefix} transaction target sequence geometry differs"
            )
        for name, upper in (
            ("opcode", config.transaction_count),
            ("source", config.num_slots),
            ("target", config.num_slots),
            ("relation", config.num_relations),
            ("type_index", config.num_types),
            ("value_code", config.num_value_codes),
        ):
            _range_check(
                getattr(intervention_targets, name),
                upper,
                name=f"{prefix}_transaction_targets.{name}",
            )
        intervention_transaction_tensors += tuple(
            getattr(prediction, field.name)
            for field in fields(prediction)
        )
        intervention_transaction_tensors += tuple(
            getattr(intervention_targets, field.name)
            for field in fields(intervention_targets)
        )
    query_pair_tensors: tuple[torch.Tensor, ...] = ()
    for name, pair in (
        ("world_query_binding", batch.world_query_binding),
        ("command_query_binding", batch.command_query_binding),
    ):
        if not isinstance(pair, ETTRCausalQueryPair):
            raise ETTRObjectiveError(f"{name} pair type differs")
        if pair.correct_logits.shape != (rows, config.vocab_size):
            raise ETTRObjectiveError(
                f"{name} pair count or vocabulary geometry differs"
            )
        query_pair_tensors += (
            pair.correct_logits,
            pair.foil_logits,
            pair.correct_target,
            pair.foil_target,
        )
    initial: tuple[torch.Tensor, ...] = ()
    for name in ("initial_committed", "initial_halted"):
        value = _tensor(
            getattr(batch, name),
            name=name,
            shape=(rows,),
            dtype=torch.bool,
        )
        initial += (value,)

    alignment_tensors: tuple[torch.Tensor, ...] = ()
    if batch.equivariance is None:
        if config.require_equivariance_pairs:
            raise ETTRObjectiveError("equivariance pairs are required by config")
    else:
        alignment = batch.equivariance
        if not isinstance(alignment, ETTRVariantAlignment):
            raise ETTRObjectiveError("alignment type differs")
        pairs = alignment.left_index.shape[0]
        expected_alignment_shapes = {
            "slot_permutation": (pairs, config.num_slots),
            "type_permutation": (pairs, config.num_types),
            "relation_permutation": (pairs, config.num_relations),
            "value_permutation": (pairs, config.num_value_codes),
            "slot_mask": (pairs, config.num_slots),
            "relation_mask": (
                pairs,
                config.num_relations,
                config.num_slots,
                config.num_slots,
            ),
            "step_mask": (pairs, steps),
        }
        for name, shape in expected_alignment_shapes.items():
            if getattr(alignment, name).shape != shape:
                raise ETTRObjectiveError(f"alignment.{name} config geometry differs")
        _async_assert(
            ((alignment.left_index < rows) & (alignment.right_index < rows)).all(),
            "alignment row index leaves the batch",
        )
        alignment_tensors = tuple(
            getattr(alignment, field.name) for field in fields(alignment)
        )

    packet_target_tensors = tuple(
        getattr(packet_target, field.name)
        for packet_target in (
            batch.packet_targets,
            batch.terminal_packet_targets,
            batch.world_intervention_targets,
            batch.command_intervention_targets,
        )
        for field in fields(packet_target)
    )
    target_tensors = (
        batch.token_targets.token_ids,
        batch.token_targets.mask,
        batch.token_targets.reset_mask,
        *packet_target_tensors,
        *(getattr(transactions, field.name) for field in fields(transactions)),
        *(getattr(targets, field.name) for field in fields(targets)),
        *intervention_transaction_tensors,
        *query_pair_tensors,
        *initial,
        *alignment_tensors,
    )
    _same_device(
        (logits, *state_tensors, *target_tensors),
        name="objective batch",
    )
    return _Geometry(
        batch=rows,
        tokens=tokens,
        steps=steps,
        lm_mask=lm_mask,
        operand_masks=_operand_masks(targets),
    )


def _count(mask: torch.Tensor) -> torch.Tensor:
    return mask.sum(dtype=torch.int64)


def _categorical_nll(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon: float,
) -> _LossCount:
    selected = probabilities.gather(
        -1,
        target.unsqueeze(-1),
    ).squeeze(-1).float()
    return _LossCount(
        (-(selected + epsilon).log() * mask).sum(),
        _count(mask),
    )


def _binary_nll(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon: float,
) -> _LossCount:
    probabilities = probabilities.float()
    target = target.float()
    values = -(
        target * (probabilities + epsilon).log()
        + (1.0 - target) * (1.0 - probabilities + epsilon).log()
    )
    return _LossCount((values * mask).sum(), _count(mask))


def _combine(
    losses: tuple[_LossCount, ...],
    reference: torch.Tensor,
) -> _LossCount:
    numerator = reference.sum() * 0.0
    count = torch.zeros(
        (),
        dtype=torch.int64,
        device=reference.device,
    )
    for loss in losses:
        numerator = numerator + loss.numerator
        count = count + loss.count
    return _LossCount(numerator, count)


def _token_loss(
    batch: ETTRObjectiveBatch,
    geometry: _Geometry,
) -> torch.Tensor:
    return F.cross_entropy(
        batch.token_logits[:, :-1][geometry.lm_mask],
        batch.token_targets.token_ids[:, 1:][geometry.lm_mask],
    )


def _packet_loss(
    batch: ETTRObjectiveBatch,
    config: ETTRObjectiveConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    initial, initial_slots, initial_relations = _packet_state_loss(
        batch.packet_prediction,
        batch.packet_targets,
        config,
    )
    terminal, terminal_slots, terminal_relations = _packet_state_loss(
        batch.terminal_packet_prediction,
        batch.terminal_packet_targets,
        config,
    )
    return (
        torch.stack((initial, terminal)).mean(),
        initial_slots + terminal_slots,
        initial_relations + terminal_relations,
    )


def _packet_state_loss(
    prediction: TypedTheoryState,
    target: ETTRPacketTargets,
    config: ETTRObjectiveConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    categorical_mask = target.slot_mask & target.active
    losses = (
        _categorical_nll(
            prediction.value_probabilities,
            target.value_code,
            categorical_mask,
            epsilon=config.probability_epsilon,
        ),
        _categorical_nll(
            prediction.type_probabilities,
            target.type_index,
            categorical_mask,
            epsilon=config.probability_epsilon,
        ),
        _binary_nll(
            prediction.relations,
            target.relations,
            target.relation_mask,
            epsilon=config.probability_epsilon,
        ),
        _binary_nll(
            prediction.active,
            target.active,
            target.slot_mask,
            epsilon=config.probability_epsilon,
        ),
        _binary_nll(
            prediction.root,
            target.root,
            target.slot_mask,
            epsilon=config.probability_epsilon,
        ),
        _binary_nll(
            prediction.committed,
            target.committed,
            torch.ones_like(target.committed),
            epsilon=config.probability_epsilon,
        ),
        _binary_nll(
            prediction.halted,
            target.halted,
            torch.ones_like(target.halted),
            epsilon=config.probability_epsilon,
        ),
    )
    return (
        torch.stack(tuple(loss.mean for loss in losses)).mean(),
        _count(target.slot_mask),
        _count(target.relation_mask),
    )


def _transaction_loss(
    batch: ETTRObjectiveBatch,
    config: ETTRObjectiveConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _transaction_prediction_loss(
        batch.transactions,
        batch.transaction_targets,
        config,
    )


def _transaction_prediction_loss(
    prediction: ETTRTransactionPredictions,
    targets: ETTRTransactionTargets,
    config: ETTRObjectiveConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    names = (
        "opcode",
        "source",
        "target",
        "relation",
        "type_index",
        "value_code",
    )
    losses = tuple(
        _categorical_nll(
            getattr(prediction, name),
            getattr(targets, name),
            mask,
            epsilon=config.probability_epsilon,
        )
        for name, mask in zip(
            names,
            _operand_masks(targets),
            strict=True,
        )
    )
    combined = _combine(losses, prediction.opcode)
    return combined.mean, combined.count


def _intervention_arm_loss(
    prediction: TypedTheoryState,
    targets: ETTRPacketTargets,
    transactions: ETTRTransactionPredictions,
    transaction_targets: ETTRTransactionTargets,
    config: ETTRObjectiveConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    packet, packet_slots, packet_relations = _packet_state_loss(
        prediction,
        targets,
        config,
    )
    transaction, transaction_decisions = _transaction_prediction_loss(
        transactions,
        transaction_targets,
        config,
    )
    status = _combine(
        (
            _binary_nll(
                transactions.committed,
                transaction_targets.committed,
                transaction_targets.step_mask,
                epsilon=config.probability_epsilon,
            ),
            _binary_nll(
                transactions.halted,
                transaction_targets.halted,
                transaction_targets.step_mask,
                epsilon=config.probability_epsilon,
            ),
        ),
        transactions.committed,
    ).mean
    return (
        torch.stack((packet, transaction, status)).mean(),
        packet_slots,
        packet_relations,
        transaction_decisions,
    )


def _causal_query_binding_loss(
    pair: ETTRCausalQueryPair,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    classification = 0.5 * (
        F.cross_entropy(pair.correct_logits, pair.correct_target)
        + F.cross_entropy(pair.foil_logits, pair.foil_target)
    )
    correct_for_correct = pair.correct_logits.gather(
        1,
        pair.correct_target[:, None],
    ).squeeze(1)
    correct_for_foil = pair.correct_logits.gather(
        1,
        pair.foil_target[:, None],
    ).squeeze(1)
    foil_for_correct = pair.foil_logits.gather(
        1,
        pair.correct_target[:, None],
    ).squeeze(1)
    foil_for_foil = pair.foil_logits.gather(
        1,
        pair.foil_target[:, None],
    ).squeeze(1)
    difference_in_differences = (
        correct_for_correct
        - correct_for_foil
        - foil_for_correct
        + foil_for_foil
    )
    contrast = F.softplus(margin - difference_in_differences).mean()
    support = torch.ones_like(pair.correct_target, dtype=torch.int64).sum()
    margin_satisfied = _count(difference_in_differences.ge(margin))
    return classification + contrast, support, margin_satisfied


def _commit_halt_loss(
    batch: ETTRObjectiveBatch,
    geometry: _Geometry,
    config: ETTRObjectiveConfig,
) -> torch.Tensor:
    prediction = batch.transactions
    target = batch.transaction_targets
    supervised = _combine(
        (
            _binary_nll(
                prediction.committed,
                target.committed,
                target.step_mask,
                epsilon=config.probability_epsilon,
            ),
            _binary_nll(
                prediction.halted,
                target.halted,
                target.step_mask,
                epsilon=config.probability_epsilon,
            ),
        ),
        prediction.committed,
    ).mean
    previous_commit = batch.initial_committed.to(prediction.committed.dtype)
    previous_halt = batch.initial_halted.to(prediction.halted.dtype)
    numerator = prediction.committed.sum() * 0.0
    count = torch.zeros(
        (),
        dtype=torch.int64,
        device=prediction.committed.device,
    )
    for step in range(geometry.steps):
        expected_commit = previous_commit + (
            (1.0 - previous_commit).square()
            * (1.0 - previous_halt)
            * prediction.opcode[:, step, 6]
        )
        expected_halt = previous_halt + (
            (1.0 - previous_halt).square() * prediction.opcode[:, step, 7]
        )
        valid = target.step_mask[:, step]
        numerator = (
            numerator
            + ((prediction.committed[:, step] - expected_commit).square() * valid).sum()
        )
        numerator = (
            numerator
            + ((prediction.halted[:, step] - expected_halt).square() * valid).sum()
        )
        count = count + 2 * _count(valid)
        previous_commit = prediction.committed[:, step]
        previous_halt = prediction.halted[:, step]
    return supervised + _LossCount(numerator, count).mean


def _gather_axis(
    value: torch.Tensor,
    indices: torch.Tensor,
    axis: int,
) -> torch.Tensor:
    output_shape = list(value.shape)
    output_shape[axis] = indices.shape[1]
    index_shape = [indices.shape[0]] + [1] * (value.ndim - 1)
    index_shape[axis] = indices.shape[1]
    return value.gather(
        axis,
        indices.view(index_shape).expand(output_shape),
    )


def _masked_square(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor,
) -> _LossCount:
    expanded = mask
    while expanded.ndim < left.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(left)
    return _LossCount(
        ((left - right).square() * expanded).sum(),
        _count(expanded),
    )


def _identity_maps(
    alignment: ETTRVariantAlignment,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.arange(
            value.shape[1],
            device=value.device,
        ).expand_as(value)
        for value in (
            alignment.slot_permutation,
            alignment.type_permutation,
            alignment.relation_permutation,
            alignment.value_permutation,
        )
    )


def _aligned_packet(
    state: TypedTheoryState,
    rows: torch.Tensor,
    maps: tuple[torch.Tensor, ...],
) -> dict[str, torch.Tensor]:
    slot, type_index, relation, value_code = maps
    values = _gather_axis(
        state.value_probabilities[rows],
        slot,
        1,
    )
    values = _gather_axis(values, value_code, 2)
    types = _gather_axis(
        state.type_probabilities[rows],
        slot,
        1,
    )
    types = _gather_axis(types, type_index, 2)
    relations = _gather_axis(state.relations[rows], relation, 1)
    relations = _gather_axis(relations, slot, 2)
    relations = _gather_axis(relations, slot, 3)
    return {
        "value": values,
        "type": types,
        "relations": relations,
        "active": _gather_axis(state.active[rows], slot, 1),
        "root": _gather_axis(state.root[rows], slot, 1),
        "committed": state.committed[rows],
        "halted": state.halted[rows],
    }


def _aligned_transactions(
    prediction: ETTRTransactionPredictions,
    rows: torch.Tensor,
    maps: tuple[torch.Tensor, ...],
) -> dict[str, torch.Tensor]:
    slot, type_index, relation, value_code = maps
    return {
        "opcode": prediction.opcode[rows],
        "source": _gather_axis(prediction.source[rows], slot, 2),
        "target": _gather_axis(prediction.target[rows], slot, 2),
        "relation": _gather_axis(
            prediction.relation[rows],
            relation,
            2,
        ),
        "type": _gather_axis(
            prediction.type_index[rows],
            type_index,
            2,
        ),
        "value": _gather_axis(
            prediction.value_code[rows],
            value_code,
            2,
        ),
        "active": _gather_axis(prediction.active[rows], slot, 2),
        "committed": prediction.committed[rows],
        "halted": prediction.halted[rows],
    }


def _equivariance_loss(
    batch: ETTRObjectiveBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    alignment = batch.equivariance
    if alignment is None:
        zero_loss = batch.token_logits.sum() * 0.0
        zero_count = torch.zeros(
            (),
            dtype=torch.int64,
            device=batch.token_logits.device,
        )
        return zero_loss, zero_count, zero_count
    left_maps = _identity_maps(alignment)
    right_maps = (
        alignment.slot_permutation,
        alignment.type_permutation,
        alignment.relation_permutation,
        alignment.value_permutation,
    )
    pair_mask = torch.ones_like(
        alignment.left_index,
        dtype=torch.bool,
    )
    packet = _combine(
        tuple(
            _equivariant_packet_loss(
                state,
                alignment,
                left_maps,
                right_maps,
                pair_mask,
                batch.token_logits,
            )
            for state in (
                batch.packet_prediction,
                batch.terminal_packet_prediction,
            )
        ),
        batch.token_logits,
    )
    left_policy = _aligned_transactions(
        batch.transactions,
        alignment.left_index,
        left_maps,
    )
    right_policy = _aligned_transactions(
        batch.transactions,
        alignment.right_index,
        right_maps,
    )
    step_slot_mask = alignment.step_mask[:, :, None] & alignment.slot_mask[:, None, :]
    transaction = _combine(
        (
            _masked_square(
                left_policy["opcode"],
                right_policy["opcode"],
                alignment.step_mask,
            ),
            _masked_square(
                left_policy["source"],
                right_policy["source"],
                step_slot_mask,
            ),
            _masked_square(
                left_policy["target"],
                right_policy["target"],
                step_slot_mask,
            ),
            _masked_square(
                left_policy["relation"],
                right_policy["relation"],
                alignment.step_mask,
            ),
            _masked_square(
                left_policy["type"],
                right_policy["type"],
                alignment.step_mask,
            ),
            _masked_square(
                left_policy["value"],
                right_policy["value"],
                alignment.step_mask,
            ),
            _masked_square(
                left_policy["active"],
                right_policy["active"],
                step_slot_mask,
            ),
            _masked_square(
                left_policy["committed"],
                right_policy["committed"],
                alignment.step_mask,
            ),
            _masked_square(
                left_policy["halted"],
                right_policy["halted"],
                alignment.step_mask,
            ),
        ),
        batch.token_logits,
    )
    combined = _combine((packet, transaction), batch.token_logits)
    return combined.mean, packet.count, transaction.count


def _equivariant_packet_loss(
    state: TypedTheoryState,
    alignment: ETTRVariantAlignment,
    left_maps: tuple[torch.Tensor, ...],
    right_maps: tuple[torch.Tensor, ...],
    pair_mask: torch.Tensor,
    reference: torch.Tensor,
) -> _LossCount:
    left = _aligned_packet(state, alignment.left_index, left_maps)
    right = _aligned_packet(state, alignment.right_index, right_maps)
    return _combine(
        (
            _masked_square(left["value"], right["value"], alignment.slot_mask),
            _masked_square(left["type"], right["type"], alignment.slot_mask),
            _masked_square(
                left["relations"],
                right["relations"],
                alignment.relation_mask,
            ),
            _masked_square(left["active"], right["active"], alignment.slot_mask),
            _masked_square(left["root"], right["root"], alignment.slot_mask),
            _masked_square(left["committed"], right["committed"], pair_mask),
            _masked_square(left["halted"], right["halted"], pair_mask),
        ),
        reference,
    )


def _sparsity_loss(
    state: TypedTheoryState,
    config: ETTRObjectiveConfig,
) -> torch.Tensor:
    active_count = state.active.sum(-1)
    edge_count = state.relations.sum(dim=(1, 2, 3))
    edge_capacity = config.num_relations * config.num_slots * config.num_slots
    return (
        active_count / config.num_slots
        + edge_count / edge_capacity
        + (F.relu(active_count - config.active_slot_budget) / config.num_slots).square()
        + (F.relu(edge_count - config.relation_edge_budget) / edge_capacity).square()
    ).mean()


def _anti_bypass_loss(
    state: TypedTheoryState,
    config: ETTRObjectiveConfig,
) -> torch.Tensor:
    active = state.active
    pair_active = active[:, None, :, None] * active[:, None, None, :]
    any_active = active.sum(-1).gt(0).to(active.dtype)
    edge_capacity = config.num_relations * config.num_slots * config.num_slots
    edge_count = state.relations.sum(dim=(1, 2, 3))
    discrete_fields = (
        state.value_probabilities,
        state.type_probabilities,
        state.relations,
        state.active,
        state.root,
        state.committed,
        state.halted,
    )
    discreteness = torch.stack(
        tuple((value * (1.0 - value)).mean() for value in discrete_fields)
    ).mean()
    structural = torch.stack(
        (
            (state.value_probabilities.sum(-1) - active).square().mean(),
            (state.type_probabilities.sum(-1) - active).square().mean(),
            (state.relations * (1.0 - pair_active)).square().mean(),
            (state.root * (1.0 - active)).square().mean(),
            (state.root.sum(-1) - any_active).square().mean(),
            (F.relu(edge_count - config.relation_edge_budget) / edge_capacity)
            .square()
            .mean(),
        )
    ).mean()
    return discreteness + structural


class ETTRCompositeObjective(nn.Module):
    """Frozen composite objective over ETTR-native tensors."""

    def __init__(
        self,
        config: ETTRObjectiveConfig,
        *,
        weights: ETTRObjectiveWeights | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, ETTRObjectiveConfig):
            raise ETTRObjectiveError("objective config type differs")
        self.config = config
        self.weights = ETTRObjectiveWeights() if weights is None else weights
        if not isinstance(self.weights, ETTRObjectiveWeights):
            raise ETTRObjectiveError("objective weights type differs")

    def forward(self, batch: ETTRObjectiveBatch) -> ETTRCompositeLoss:
        geometry = _validate_batch(batch, self.config)
        token_lm = _token_loss(batch, geometry)
        packet, packet_slots, relation_cells = _packet_loss(
            batch,
            self.config,
        )
        (
            world_intervention,
            world_intervention_slots,
            world_intervention_relation_cells,
            world_intervention_transaction_decisions,
        ) = _intervention_arm_loss(
            batch.world_intervention_prediction,
            batch.world_intervention_targets,
            batch.world_intervention_transactions,
            batch.world_intervention_transaction_targets,
            self.config,
        )
        (
            command_intervention,
            command_intervention_slots,
            command_intervention_relation_cells,
            command_intervention_transaction_decisions,
        ) = _intervention_arm_loss(
            batch.command_intervention_prediction,
            batch.command_intervention_targets,
            batch.command_intervention_transactions,
            batch.command_intervention_transaction_targets,
            self.config,
        )
        (
            world_query_binding,
            supervised_world_query_pairs,
            world_query_margin_satisfied,
        ) = _causal_query_binding_loss(
            batch.world_query_binding,
            margin=self.config.query_binding_margin,
        )
        (
            command_query_binding,
            supervised_command_query_pairs,
            command_query_margin_satisfied,
        ) = _causal_query_binding_loss(
            batch.command_query_binding,
            margin=self.config.query_binding_margin,
        )
        transaction, transaction_decisions = _transaction_loss(
            batch,
            self.config,
        )
        commit_halt = _commit_halt_loss(
            batch,
            geometry,
            self.config,
        )
        equivariance, equiv_packet, equiv_transaction = _equivariance_loss(batch)
        sparsity = torch.stack(
            tuple(
                _sparsity_loss(state, self.config)
                for state in (
                    batch.packet_prediction,
                    batch.terminal_packet_prediction,
                    batch.world_intervention_prediction,
                    batch.command_intervention_prediction,
                )
            )
        ).mean()
        anti_bypass = torch.stack(
            tuple(
                _anti_bypass_loss(state, self.config)
                for state in (
                    batch.packet_prediction,
                    batch.terminal_packet_prediction,
                    batch.world_intervention_prediction,
                    batch.command_intervention_prediction,
                )
            )
        ).mean()
        breakdown = {
            "token_lm": token_lm,
            "packet": packet,
            "world_intervention": world_intervention,
            "command_intervention": command_intervention,
            "world_query_binding": world_query_binding,
            "command_query_binding": command_query_binding,
            "transaction": transaction,
            "equivariance": equivariance,
            "commit_halt": commit_halt,
            "sparsity": sparsity,
            "anti_bypass": anti_bypass,
        }
        total = sum(
            getattr(self.weights, name) * loss for name, loss in breakdown.items()
        )
        receipt = ETTRObjectiveReceipt(
            schema=OBJECTIVE_SCHEMA,
            batch_size=geometry.batch,
            sequence_tokens=geometry.tokens,
            lm_target_tokens=_count(geometry.lm_mask),
            supervised_packet_slots=packet_slots,
            supervised_relation_cells=relation_cells,
            supervised_world_intervention_slots=world_intervention_slots,
            supervised_world_intervention_relation_cells=(
                world_intervention_relation_cells
            ),
            supervised_world_intervention_transaction_decisions=(
                world_intervention_transaction_decisions
            ),
            supervised_command_intervention_slots=(
                command_intervention_slots
            ),
            supervised_command_intervention_relation_cells=(
                command_intervention_relation_cells
            ),
            supervised_command_intervention_transaction_decisions=(
                command_intervention_transaction_decisions
            ),
            supervised_world_query_pairs=supervised_world_query_pairs,
            supervised_command_query_pairs=supervised_command_query_pairs,
            world_query_margin_satisfied=world_query_margin_satisfied,
            command_query_margin_satisfied=command_query_margin_satisfied,
            supervised_transaction_steps=_count(batch.transaction_targets.step_mask),
            supervised_transaction_decisions=transaction_decisions,
            supervised_opcode_decisions=_count(geometry.operand_masks[0]),
            supervised_source_decisions=_count(geometry.operand_masks[1]),
            supervised_target_decisions=_count(geometry.operand_masks[2]),
            supervised_relation_decisions=_count(geometry.operand_masks[3]),
            supervised_type_decisions=_count(geometry.operand_masks[4]),
            supervised_value_code_decisions=_count(geometry.operand_masks[5]),
            equivariance_pairs=(
                0
                if batch.equivariance is None
                else batch.equivariance.left_index.shape[0]
            ),
            equivariance_packet_cells=equiv_packet,
            equivariance_transaction_cells=equiv_transaction,
            causal_lm_shift=self.config.causal_lm_shift,
            weights=self.weights.items(),
        )
        return ETTRCompositeLoss(
            total=total,
            receipt=receipt,
            **breakdown,
        )


ETTRObjective = ETTRCompositeObjective


__all__ = [
    "ETTRCausalQueryPair",
    "ETTRCompositeLoss",
    "ETTRCompositeObjective",
    "ETTRObjective",
    "ETTRObjectiveBatch",
    "ETTRObjectiveConfig",
    "ETTRObjectiveError",
    "ETTRObjectiveReceipt",
    "ETTRObjectiveWeights",
    "ETTRPacketTargets",
    "ETTRTokenTargets",
    "ETTRTransactionPredictions",
    "ETTRTransactionTargets",
    "ETTRVariantAlignment",
    "OBJECTIVE_SCHEMA",
]
