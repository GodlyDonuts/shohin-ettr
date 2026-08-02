"""Apply one supervised typed-state transition per public COMMAND operation.

Terminal-only recurrence can carry an opaque latent across operations while
receiving no credit for where composition first diverges.  This compiler
instead decodes and applies a typed edit after every explicit public
operation.  The resulting state is fed into the next tied transition.  A
final tied transition writes outcome and disposition fields not owned by an
individual operation.

Only WORLD-derived initial state and public COMMAND tokens/residuals are read
at inference.  Operation-boundary states and mutation traces are training
labels, never runtime inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from endogenous_typed_theory_reactor import (
    TheoryReactorError,
    TypedTheoryState,
    validate_state,
)
from parallel_terminal_state_compiler import (
    AtomicTypedEdits,
    ParallelTerminalStateCompiler,
)


@dataclass(frozen=True, slots=True)
class OperationStateTransitionTrace:
    """Differentiable state/edit sequence emitted by the public executor."""

    operation_states: tuple[TypedTheoryState, ...]
    operation_edits: tuple[AtomicTypedEdits, ...]
    operation_mask: torch.Tensor
    final_edits: AtomicTypedEdits


def _select_state(
    previous: TypedTheoryState,
    candidate: TypedTheoryState,
    selected: torch.Tensor,
) -> TypedTheoryState:
    if (
        selected.ndim != 1
        or selected.dtype != torch.bool
        or selected.shape[0] != previous.active.shape[0]
        or previous.active.shape != candidate.active.shape
    ):
        raise TheoryReactorError("operation state selection differs")

    def choose(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        mask = selected.reshape(
            selected.shape[0],
            *((1,) * (left.ndim - 1)),
        )
        return torch.where(mask, right, left)

    return TypedTheoryState(
        value_probabilities=choose(
            previous.value_probabilities,
            candidate.value_probabilities,
        ),
        type_probabilities=choose(
            previous.type_probabilities,
            candidate.type_probabilities,
        ),
        relations=choose(previous.relations, candidate.relations),
        active=choose(previous.active, candidate.active),
        root=choose(previous.root, candidate.root),
        committed=choose(previous.committed, candidate.committed),
        halted=choose(previous.halted, candidate.halted),
        step=candidate.step,
    )


class OperationStateTransitionCompiler(ParallelTerminalStateCompiler):
    """Tied operation-level state machine over declaration-resolved syntax."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        kwargs["atomic_edits"] = True
        kwargs["lexical_command"] = True
        kwargs["token_native_command_mask"] = True
        kwargs["cover_verified_command_mask"] = True
        kwargs["token_native_syntax_graph_command"] = True
        kwargs["token_native_declaration_binding_command"] = True
        kwargs["token_native_operation_recurrence_command"] = True
        super().__init__(*args, **kwargs)

    def _prepare_public_context(
        self,
        state: TypedTheoryState,
        *,
        command_hidden: torch.Tensor,
        command_lexical: torch.Tensor,
        command_tokens: torch.Tensor,
        command_attention_mask: torch.Tensor,
        steps: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        validate_state(state, self.config)
        batch = state.active.shape[0]
        if (
            not 1 <= steps <= self.config.max_steps - state.step
            or command_hidden.ndim != 3
            or command_hidden.shape[0] != batch
            or command_hidden.shape[-1] != self.config.d_model
            or command_lexical.shape != command_hidden.shape
            or command_tokens.shape != command_hidden.shape[:2]
            or command_tokens.dtype != torch.long
            or command_attention_mask.shape != command_hidden.shape[:2]
            or command_attention_mask.dtype != torch.bool
            or self.command_document_mask is None
            or self.command_syntax_graph_encoder is None
            or self.command_operation_router is None
            or self.operation_recurrence is None
            or self.command_lexical_projection is None
        ):
            raise TheoryReactorError("operation state compiler input differs")

        document_mask = self.command_document_mask(
            command_tokens,
            command_attention_mask,
        )
        command = self.command_projection(command_hidden)
        command = command + self.command_lexical_projection(command_lexical)
        command = self.command_norm(command)
        command = self.command_syntax_graph_encoder(
            command,
            command_tokens,
            document_mask,
        )
        initial_memory = self._initial_memory(state)
        memory = torch.cat((command, initial_memory), dim=1)
        memory_padding = torch.cat(
            (
                ~document_mask,
                torch.zeros(
                    batch,
                    self.config.num_slots,
                    dtype=torch.bool,
                    device=command.device,
                ),
            ),
            dim=1,
        )
        slots = self.terminal_slot_queries.to(command.dtype)
        slots = slots.unsqueeze(0).expand(batch, -1, -1) + initial_memory
        for layer in self.layers:
            slots = layer(slots, memory, memory_padding)
        operation_masks, operation_count = self.command_operation_router(
            command_tokens,
            document_mask,
        )
        return (
            command,
            document_mask,
            slots,
            initial_memory,
            operation_masks,
            operation_count,
        )

    def forward_with_operation_states(
        self,
        state: TypedTheoryState,
        *,
        command_hidden: torch.Tensor,
        command_lexical: torch.Tensor,
        command_tokens: torch.Tensor,
        command_attention_mask: torch.Tensor,
        steps: int,
        hard: bool,
    ) -> tuple[TypedTheoryState, OperationStateTransitionTrace]:
        (
            command,
            document_mask,
            slots,
            _initial_memory,
            operation_masks,
            operation_count,
        ) = self._prepare_public_context(
            state,
            command_hidden=command_hidden,
            command_lexical=command_lexical,
            command_tokens=command_tokens,
            command_attention_mask=command_attention_mask,
            steps=steps,
        )
        if self.operation_recurrence is None:
            raise TheoryReactorError("operation state recurrence is absent")
        batch = state.active.shape[0]
        operation_padding = torch.zeros(
            batch,
            1 + self.config.num_slots,
            dtype=torch.bool,
            device=command.device,
        )
        current = state
        states: list[TypedTheoryState] = []
        edits: list[AtomicTypedEdits] = []
        for operation_index in range(operation_masks.shape[1]):
            operation = torch.bmm(
                operation_masks[:, operation_index]
                .unsqueeze(1)
                .to(command.dtype),
                command,
            )
            updated_slots = self.operation_recurrence(
                slots,
                torch.cat((operation, self._initial_memory(current)), dim=1),
                operation_padding,
            )
            selected = operation_count.gt(operation_index)
            operation_edits = self._atomic_edits_from_slots(
                current,
                updated_slots,
                hard=hard,
            )
            candidate = self.apply_atomic_edits(
                current,
                operation_edits,
                steps=1,
                hard=hard,
            )
            current = _select_state(current, candidate, selected)
            slots = torch.where(
                selected[:, None, None],
                updated_slots,
                slots,
            )
            states.append(current)
            edits.append(operation_edits)

        final_slots = self.operation_recurrence(
            slots,
            torch.cat((command, self._initial_memory(current)), dim=1),
            torch.cat(
                (
                    ~document_mask,
                    torch.zeros(
                        batch,
                        self.config.num_slots,
                        dtype=torch.bool,
                        device=command.device,
                    ),
                ),
                dim=1,
            ),
        )
        final_edits = self._atomic_edits_from_slots(
            current,
            final_slots,
            hard=hard,
        )
        terminal = self.apply_atomic_edits(
            current,
            final_edits,
            steps=1,
            hard=hard,
        )
        terminal = TypedTheoryState(
            value_probabilities=terminal.value_probabilities,
            type_probabilities=terminal.type_probabilities,
            relations=terminal.relations,
            active=terminal.active,
            root=terminal.root,
            committed=terminal.committed,
            halted=terminal.halted,
            step=state.step + steps,
        )
        return terminal, OperationStateTransitionTrace(
            operation_states=tuple(states),
            operation_edits=tuple(edits),
            operation_mask=torch.arange(
                operation_masks.shape[1],
                device=command.device,
            )[None, :].lt(operation_count[:, None]),
            final_edits=final_edits,
        )

    def forward(
        self,
        state: TypedTheoryState,
        *,
        command_hidden: torch.Tensor,
        command_lexical: torch.Tensor | None = None,
        command_tokens: torch.Tensor | None = None,
        command_attention_mask: torch.Tensor,
        steps: int,
        hard: bool,
    ) -> TypedTheoryState:
        if command_lexical is None or command_tokens is None:
            raise TheoryReactorError("operation state public inputs are absent")
        terminal, _trace = self.forward_with_operation_states(
            state,
            command_hidden=command_hidden,
            command_lexical=command_lexical,
            command_tokens=command_tokens,
            command_attention_mask=command_attention_mask,
            steps=steps,
            hard=hard,
        )
        return terminal


__all__ = [
    "OperationStateTransitionCompiler",
    "OperationStateTransitionTrace",
]
