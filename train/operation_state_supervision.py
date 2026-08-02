"""Derive exact operation-boundary states from offline ETTR trace labels."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from endogenous_typed_theory_reactor import TypedTheoryState
from ettr_objectives import ETTRTransactionTargets
from parallel_terminal_state_compiler import AtomicTypedEdits
from probe_ettr_oracle_interfaces import target_policy


_WRITE_OPCODE = 1
_COMMAND_SLOT_START = 48
_COMMAND_SLOT_STOP = 54
_TERMINAL_SUFFIX_STEPS = 2
_MAXIMUM_OPERATIONS = _COMMAND_SLOT_STOP - _COMMAND_SLOT_START


class OperationStateSupervisionError(ValueError):
    """The offline trace does not expose the frozen operation geometry."""


@dataclass(frozen=True, slots=True)
class OperationBoundaryTargets:
    states: tuple[TypedTheoryState, ...]
    mask: torch.Tensor
    last_state: TypedTheoryState


def index_typed_state(
    state: TypedTheoryState,
    index: torch.Tensor,
) -> TypedTheoryState:
    if index.ndim != 1 or index.dtype != torch.long:
        raise OperationStateSupervisionError("state index differs")
    return TypedTheoryState(
        value_probabilities=state.value_probabilities.index_select(0, index),
        type_probabilities=state.type_probabilities.index_select(0, index),
        relations=state.relations.index_select(0, index),
        active=state.active.index_select(0, index),
        root=state.root.index_select(0, index),
        committed=state.committed.index_select(0, index),
        halted=state.halted.index_select(0, index),
        step=state.step,
    )


def index_atomic_edits(
    edits: AtomicTypedEdits,
    index: torch.Tensor,
) -> AtomicTypedEdits:
    if index.ndim != 1 or index.dtype != torch.long:
        raise OperationStateSupervisionError("edit index differs")
    def optional(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else value.index_select(0, index)

    return AtomicTypedEdits(
        node_action=edits.node_action.index_select(0, index),
        value_code=edits.value_code.index_select(0, index),
        type_index=edits.type_index.index_select(0, index),
        relation_action=edits.relation_action.index_select(0, index),
        root_action=edits.root_action.index_select(0, index),
        disposition_action=edits.disposition_action.index_select(0, index),
        node_edit_count=optional(edits.node_edit_count),
        relation_link_count=optional(edits.relation_link_count),
        relation_unlink_count=optional(edits.relation_unlink_count),
        effect_kind=optional(edits.effect_kind),
        effect_node_pointer=optional(edits.effect_node_pointer),
        effect_value_code=optional(edits.effect_value_code),
        effect_type_index=optional(edits.effect_type_index),
        effect_relation_link=optional(edits.effect_relation_link),
        effect_relation_unlink=optional(edits.effect_relation_unlink),
        effect_root_pointer=optional(edits.effect_root_pointer),
    )


def operation_boundary_indices(
    targets: ETTRTransactionTargets,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return inclusive low-level step indices after each semantic operation."""

    valid = targets.step_mask
    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise OperationStateSupervisionError("transaction mask differs")
    positions = torch.arange(valid.shape[1], device=valid.device)
    starts = (
        valid
        & targets.opcode.eq(_WRITE_OPCODE)
        & targets.source.ge(_COMMAND_SLOT_START)
        & targets.source.lt(_COMMAND_SLOT_STOP)
    )
    count = starts.sum(-1)
    if not bool(
        (count.ge(1) & count.le(_MAXIMUM_OPERATIONS)).all()
    ):
        raise OperationStateSupervisionError("operation start count differs")
    sentinel = torch.full_like(targets.source, valid.shape[1])
    ordered = torch.where(starts, positions[None, :], sentinel).sort(-1).values
    ordered = ordered[:, :_MAXIMUM_OPERATIONS]
    ranks = torch.arange(_MAXIMUM_OPERATIONS, device=valid.device)
    mask = ranks[None, :].lt(count[:, None])
    gathered_source = targets.source.gather(1, ordered.clamp_max(valid.shape[1] - 1))
    expected_source = _COMMAND_SLOT_START + ranks
    if not bool(
        (
            gathered_source.eq(expected_source[None, :])
            | ~mask
        ).all()
    ):
        raise OperationStateSupervisionError("operation start order differs")

    valid_count = valid.sum(-1)
    last_boundary = valid_count - _TERMINAL_SUFFIX_STEPS - 1
    if not bool(last_boundary.ge(ordered[:, 0]).all()):
        raise OperationStateSupervisionError("terminal suffix geometry differs")
    shifted_start = torch.cat(
        (
            ordered[:, 1:],
            torch.full_like(ordered[:, :1], valid.shape[1]),
        ),
        dim=1,
    )
    next_start = torch.where(
        (ranks[None, :] + 1).lt(count[:, None]),
        shifted_start,
        last_boundary[:, None] + 1,
    )
    boundaries = torch.where(mask, next_start - 1, torch.zeros_like(next_start))
    if not bool(
        (
            (boundaries.ge(ordered) & boundaries.lt(valid_count[:, None]))
            | ~mask
        ).all()
    ):
        raise OperationStateSupervisionError("operation boundary differs")
    return boundaries, mask


def _gather_trace_tensor(
    values: tuple[torch.Tensor, ...],
    index: torch.Tensor,
) -> torch.Tensor:
    stacked = torch.stack(values, dim=1)
    gather = index.reshape(
        index.shape[0],
        1,
        *((1,) * (stacked.ndim - 2)),
    ).expand(index.shape[0], 1, *stacked.shape[2:])
    return stacked.gather(1, gather).squeeze(1)


def oracle_operation_boundary_states(
    executor,
    initial: TypedTheoryState,
    targets: ETTRTransactionTargets,
) -> OperationBoundaryTargets:
    """Replay assessor labels and gather cumulative state at each operation."""

    boundaries, mask = operation_boundary_indices(targets)
    state = initial
    trace: list[TypedTheoryState] = []
    with torch.no_grad():
        for step in range(targets.opcode.shape[1]):
            state = executor.apply(
                state,
                target_policy(
                    targets,
                    executor.config,
                    step,
                    dtype=initial.active.dtype,
                ),
                hard=True,
                validate=False,
            )
            trace.append(state)

    def gathered(index: torch.Tensor, *, step: int) -> TypedTheoryState:
        return TypedTheoryState(
            value_probabilities=_gather_trace_tensor(
                tuple(item.value_probabilities for item in trace), index
            ),
            type_probabilities=_gather_trace_tensor(
                tuple(item.type_probabilities for item in trace), index
            ),
            relations=_gather_trace_tensor(
                tuple(item.relations for item in trace), index
            ),
            active=_gather_trace_tensor(
                tuple(item.active for item in trace), index
            ),
            root=_gather_trace_tensor(tuple(item.root for item in trace), index),
            committed=_gather_trace_tensor(
                tuple(item.committed for item in trace), index
            ),
            halted=_gather_trace_tensor(
                tuple(item.halted for item in trace), index
            ),
            step=step,
        )

    states = tuple(
        gathered(boundaries[:, rank], step=rank + 1)
        for rank in range(_MAXIMUM_OPERATIONS)
    )
    last_rank = mask.sum(-1) - 1

    def gather_last(field: str) -> torch.Tensor:
        values = tuple(getattr(item, field) for item in states)
        return _gather_trace_tensor(values, last_rank)

    last_state = TypedTheoryState(
        value_probabilities=gather_last("value_probabilities"),
        type_probabilities=gather_last("type_probabilities"),
        relations=gather_last("relations"),
        active=gather_last("active"),
        root=gather_last("root"),
        committed=gather_last("committed"),
        halted=gather_last("halted"),
        step=_MAXIMUM_OPERATIONS,
    )
    return OperationBoundaryTargets(states=states, mask=mask, last_state=last_state)


__all__ = [
    "OperationBoundaryTargets",
    "OperationStateSupervisionError",
    "index_atomic_edits",
    "index_typed_state",
    "operation_boundary_indices",
    "oracle_operation_boundary_states",
]
