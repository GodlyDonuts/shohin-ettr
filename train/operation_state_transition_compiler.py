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
from torch import nn

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


def _hard_categories(probabilities: torch.Tensor) -> torch.Tensor:
    indices = probabilities.argmax(dim=-1, keepdim=True)
    return torch.zeros_like(probabilities).scatter_(-1, indices, 1.0)


def _count_constrained_selection(
    scores: torch.Tensor,
    valid: torch.Tensor,
    count_probabilities: torch.Tensor,
    *,
    capacity: torch.Tensor,
    hard: bool,
) -> torch.Tensor:
    """Select the highest-scoring valid coordinates under a learned count."""

    if (
        scores.shape != valid.shape
        or scores.ndim < 2
        or valid.dtype != torch.bool
        or count_probabilities.ndim != 2
        or count_probabilities.shape[0] != scores.shape[0]
        or capacity.shape != (scores.shape[0],)
        or capacity.dtype != torch.long
    ):
        raise TheoryReactorError("operation effect selection differs")
    batch = scores.shape[0]
    flat_scores = scores.reshape(batch, -1)
    flat_valid = valid.reshape(batch, -1)
    order = flat_scores.masked_fill(~flat_valid, -torch.inf).argsort(
        dim=-1,
        descending=True,
    )
    ordered_valid = flat_valid.gather(1, order)
    ranks = torch.arange(flat_scores.shape[1], device=scores.device)[None, :]
    permitted = ranks.lt(capacity[:, None]) & ordered_valid
    if hard:
        count = count_probabilities.argmax(-1).minimum(capacity)
        ordered_selection = ranks.lt(count[:, None]) & permitted
        selection = torch.zeros_like(flat_scores)
        selection.scatter_(1, order, ordered_selection.to(flat_scores.dtype))
        return selection.reshape_as(scores)

    # P(count > rank) is a differentiable exact-cardinality relaxation.  The
    # ordering is discrete, while gradients still reach both count logits and
    # the selected address/payload heads through their supervised objectives.
    survival = 1.0 - count_probabilities.cumsum(-1)[:, :-1]
    if survival.shape[1] < flat_scores.shape[1]:
        survival = torch.cat(
            (
                survival,
                torch.zeros(
                    batch,
                    flat_scores.shape[1] - survival.shape[1],
                    dtype=survival.dtype,
                    device=survival.device,
                ),
            ),
            dim=1,
        )
    ordered_selection = survival[:, : flat_scores.shape[1]]
    ordered_selection = ordered_selection * permitted.to(ordered_selection.dtype)
    ranked_selection = torch.zeros_like(flat_scores)
    ranked_selection.scatter_(
        1,
        order,
        ordered_selection.to(ranked_selection.dtype),
    )
    masked_scores = flat_scores.masked_fill(~flat_valid, -1e9)
    address_weights = masked_scores.softmax(-1) * flat_valid.to(flat_scores.dtype)
    address_weights = address_weights / address_weights.sum(
        -1,
        keepdim=True,
    ).clamp_min(1e-7)
    soft_selection = (
        address_weights * ranked_selection.sum(-1, keepdim=True).detach()
    ).clamp(max=1.0)
    # Preserve sparse count-constrained values in the forward pass while
    # giving address scores a straight-through gradient.
    selection = ranked_selection + soft_selection - soft_selection.detach()
    return selection.reshape_as(scores)


class FactorizedOperationStateTransitionCompiler(
    OperationStateTransitionCompiler
):
    """Compile sparse operation effects before binding state operands.

    Dense atomic heads decide KEEP independently at every coordinate.  This
    variant first predicts three global effect cardinalities, then binds only
    that many node edits, relation links, and relation unlinks to the current
    state.  The same fixed typed algebra applies the resulting edit.
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.node_edit_count_head = nn.Linear(
            self.width,
            self.config.num_slots + 1,
        )
        self.relation_link_count_head = nn.Linear(
            self.width,
            self.config.max_edges + 1,
        )
        self.relation_unlink_count_head = nn.Linear(
            self.width,
            self.config.max_edges + 1,
        )
        for head in (
            self.node_edit_count_head,
            self.relation_link_count_head,
            self.relation_unlink_count_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, -8.0)
            with torch.no_grad():
                head.bias[0] = 4.0

    def _atomic_edits_from_slots(
        self,
        state: TypedTheoryState,
        slots: torch.Tensor,
        *,
        hard: bool,
    ) -> AtomicTypedEdits:
        base = super()._atomic_edits_from_slots(state, slots, hard=False)
        pooled = slots.mean(1)
        node_count = self.node_edit_count_head(pooled).float().softmax(-1)
        link_count = self.relation_link_count_head(pooled).float().softmax(-1)
        unlink_count = (
            self.relation_unlink_count_head(pooled).float().softmax(-1)
        )

        active = state.active.gt(0.5)
        node_allowed = torch.stack(
            (
                ~active,
                active,
                active,
                active,
            ),
            dim=-1,
        )
        conditional_node = base.node_action[..., 1:] * node_allowed
        conditional_node = conditional_node / conditional_node.sum(
            -1,
            keepdim=True,
        ).clamp_min(1e-7)
        node_score = conditional_node.max(-1).values.clamp_min(1e-7).log()
        node_score = node_score - base.node_action[..., 0].clamp_min(1e-7).log()
        node_selection = _count_constrained_selection(
            node_score,
            node_allowed.any(-1),
            node_count,
            capacity=node_allowed.any(-1).sum(-1),
            hard=hard,
        )
        if hard:
            conditional_node = _hard_categories(conditional_node)
        node_action = torch.cat(
            (
                1.0 - node_selection.unsqueeze(-1),
                node_selection.unsqueeze(-1) * conditional_node,
            ),
            dim=-1,
        )

        relations = state.relations.gt(0.5)
        pair_active = active[:, None, :, None] & active[:, None, None, :]
        link_valid = ~relations & pair_active
        unlink_valid = relations & pair_active
        existing = relations.sum(dim=(1, 2, 3)).to(torch.long)
        link_capacity = (self.config.max_edges - existing).clamp_min(0)
        unlink_capacity = existing
        link_selection = _count_constrained_selection(
            base.relation_action[..., 1],
            link_valid,
            link_count,
            capacity=link_capacity,
            hard=hard,
        )
        unlink_selection = _count_constrained_selection(
            base.relation_action[..., 2],
            unlink_valid,
            unlink_count,
            capacity=unlink_capacity,
            hard=hard,
        )
        relation_action = torch.stack(
            (
                (1.0 - link_selection - unlink_selection).clamp_min(0.0),
                link_selection,
                unlink_selection,
            ),
            dim=-1,
        )

        value_code = base.value_code
        type_index = base.type_index
        root_action = base.root_action
        disposition_action = base.disposition_action
        if hard:
            value_code = _hard_categories(value_code)
            type_index = _hard_categories(type_index)
            root_action = _hard_categories(root_action)
            disposition_action = _hard_categories(disposition_action)
            node_count = _hard_categories(node_count)
            link_count = _hard_categories(link_count)
            unlink_count = _hard_categories(unlink_count)
        return AtomicTypedEdits(
            node_action=node_action,
            value_code=value_code,
            type_index=type_index,
            relation_action=relation_action,
            root_action=root_action,
            disposition_action=disposition_action,
            node_edit_count=node_count,
            relation_link_count=link_count,
            relation_unlink_count=unlink_count,
        )


EFFECT_KIND_COUNT = 12
EFFECT_NOOP = 0
EFFECT_ALLOCATE = 1
EFFECT_WRITE = 2
EFFECT_CLEAR = 3
EFFECT_REPLACE = 4
EFFECT_LINK = 5
EFFECT_UNLINK = 6
EFFECT_ROOT_CLEAR = 7
EFFECT_ROOT_SET = 8
EFFECT_COMMIT = 9
EFFECT_HALT = 10
EFFECT_REJECT = 11


def _masked_flat_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != mask.shape or mask.dtype != torch.bool:
        raise TheoryReactorError("operation effect pointer mask differs")
    batch, effects = logits.shape[:2]
    flat_logits = logits.reshape(batch, effects, -1)
    flat_mask = mask.reshape(batch, effects, -1)
    probabilities = flat_logits.masked_fill(~flat_mask, -1e9).softmax(-1)
    probabilities = probabilities * flat_mask.to(probabilities.dtype)
    probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(
        1e-7
    )
    return probabilities.reshape_as(logits)


def _bernoulli_count_distribution(probabilities: torch.Tensor) -> torch.Tensor:
    """Exact differentiable count law for independent effect slots."""

    if probabilities.ndim != 2:
        raise TheoryReactorError("operation effect count probabilities differ")
    batch, effects = probabilities.shape
    distribution = torch.zeros(
        batch,
        effects + 1,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    distribution[:, 0] = 1.0
    for index in range(effects):
        value = probabilities[:, index : index + 1]
        stayed = distribution * (1.0 - value)
        advanced = torch.cat(
            (
                torch.zeros_like(distribution[:, :1]),
                distribution[:, :-1] * value,
            ),
            dim=1,
        )
        distribution = stayed + advanced
    return distribution


class OperationEffectSetCompiler(OperationStateTransitionCompiler):
    """Emit an unordered bounded set of state-grounded typed effects."""

    def __init__(
        self,
        *args,
        maximum_effects: int = 16,
        **kwargs,
    ) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        if not isinstance(maximum_effects, int) or not 1 <= maximum_effects <= 64:
            raise TheoryReactorError("operation effect set geometry differs")
        self.maximum_effects = maximum_effects
        self.effect_queries = nn.Parameter(torch.empty(maximum_effects, self.width))
        self.effect_input_norm = nn.LayerNorm(self.width)
        self.effect_self_attention = nn.MultiheadAttention(
            self.width,
            self.num_heads,
            batch_first=True,
        )
        self.effect_memory_norm = nn.LayerNorm(self.width)
        self.effect_cross_attention = nn.MultiheadAttention(
            self.width,
            self.num_heads,
            batch_first=True,
        )
        self.effect_ff_norm = nn.LayerNorm(self.width)
        self.effect_ff = nn.Sequential(
            nn.Linear(self.width, 4 * self.width),
            nn.GELU(),
            nn.Linear(4 * self.width, self.width),
        )
        self.effect_output_norm = nn.LayerNorm(self.width)
        self.effect_kind_head = nn.Linear(self.width, EFFECT_KIND_COUNT)
        self.effect_node_query = nn.Linear(self.width, self.width, bias=False)
        self.effect_node_key = nn.Linear(self.width, self.width, bias=False)
        self.effect_value_head = nn.Linear(
            self.width,
            self.config.num_value_codes,
        )
        self.effect_type_head = nn.Linear(self.width, self.config.num_types)
        self.effect_relation_source = nn.Linear(
            self.width,
            self.config.num_slots,
        )
        self.effect_relation_target = nn.Linear(
            self.width,
            self.config.num_slots,
        )
        self.effect_relation_type = nn.Linear(
            self.width,
            self.config.num_relations,
        )
        self.effect_root_head = nn.Linear(self.width, self.config.num_slots)
        nn.init.normal_(self.effect_queries, std=0.02)

    def _effect_slots(self, slots: torch.Tensor) -> torch.Tensor:
        batch = slots.shape[0]
        effects = self.effect_queries.to(slots.dtype).unsqueeze(0).expand(
            batch,
            -1,
            -1,
        )
        effects = effects + slots.mean(1, keepdim=True)
        normalized = self.effect_input_norm(effects)
        attended, _ = self.effect_self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        effects = effects + attended
        attended, _ = self.effect_cross_attention(
            self.effect_input_norm(effects),
            self.effect_memory_norm(slots),
            self.effect_memory_norm(slots),
            need_weights=False,
        )
        effects = effects + attended
        effects = effects + self.effect_ff(self.effect_ff_norm(effects))
        return self.effect_output_norm(effects)

    @staticmethod
    def _dense_actions(*masses: torch.Tensor) -> torch.Tensor:
        edited = torch.stack(masses, dim=-1)
        total = edited.sum(-1, keepdim=True)
        normalizer = total.clamp_min(1.0)
        edited = edited / normalizer
        keep = (1.0 - total).clamp_min(0.0) / normalizer
        return torch.cat((keep, edited), dim=-1)

    def _atomic_edits_from_slots(
        self,
        state: TypedTheoryState,
        slots: torch.Tensor,
        *,
        hard: bool,
    ) -> AtomicTypedEdits:
        base = super()._atomic_edits_from_slots(state, slots, hard=False)
        effects = self._effect_slots(slots)
        kind = self.effect_kind_head(effects).float().softmax(-1)
        value_code = self.effect_value_head(effects).float().softmax(-1)
        type_index = self.effect_type_head(effects).float().softmax(-1)

        node_query = self.effect_node_query(effects)
        node_key = self.effect_node_key(slots)
        node_logits = torch.einsum("bkd,bsd->bks", node_query, node_key)
        node_logits = node_logits / self.width**0.5
        active = state.active.gt(0.5)
        inactive_pointer = _masked_flat_softmax(
            node_logits,
            ~active[:, None, :].expand_as(node_logits),
        )
        active_pointer = _masked_flat_softmax(
            node_logits,
            active[:, None, :].expand_as(node_logits),
        )
        node_pointer = torch.stack((inactive_pointer, active_pointer), dim=2)

        relation_source = self.effect_relation_source(effects).float()
        relation_target = self.effect_relation_target(effects).float()
        relation_type = self.effect_relation_type(effects).float()
        relation_logits = (
            relation_type[:, :, :, None, None]
            + relation_source[:, :, None, :, None]
            + relation_target[:, :, None, None, :]
        )
        relations = state.relations.gt(0.5)
        relation_link = _masked_flat_softmax(
            relation_logits,
            (~relations)[:, None].expand_as(relation_logits),
        )
        pair_active = active[:, None, :, None] & active[:, None, None, :]
        relation_unlink = _masked_flat_softmax(
            relation_logits,
            (relations & pair_active)[:, None].expand_as(relation_logits),
        )
        root_pointer = self.effect_root_head(effects).float()
        root_pointer = _masked_flat_softmax(
            root_pointer,
            torch.ones_like(root_pointer, dtype=torch.bool),
        )

        if hard:
            kind = _hard_categories(kind)
            value_code = _hard_categories(value_code)
            type_index = _hard_categories(type_index)
            inactive_pointer = _hard_categories(inactive_pointer)
            active_pointer = _hard_categories(active_pointer)
            node_pointer = torch.stack((inactive_pointer, active_pointer), dim=2)
            relation_link = _hard_categories(
                relation_link.reshape(
                    relation_link.shape[0],
                    relation_link.shape[1],
                    -1,
                )
            ).reshape_as(relation_link)
            relation_unlink = _hard_categories(
                relation_unlink.reshape(
                    relation_unlink.shape[0],
                    relation_unlink.shape[1],
                    -1,
                )
            ).reshape_as(relation_unlink)
            root_pointer = _hard_categories(root_pointer)

        allocate = torch.einsum(
            "bk,bks->bs",
            kind[..., EFFECT_ALLOCATE],
            inactive_pointer,
        )
        write = torch.einsum(
            "bk,bks->bs",
            kind[..., EFFECT_WRITE],
            active_pointer,
        )
        clear = torch.einsum(
            "bk,bks->bs",
            kind[..., EFFECT_CLEAR],
            active_pointer,
        )
        replace = torch.einsum(
            "bk,bks->bs",
            kind[..., EFFECT_REPLACE],
            active_pointer,
        )
        node_action = self._dense_actions(allocate, write, clear, replace)

        value_mass = (
            kind[..., EFFECT_ALLOCATE, None] * inactive_pointer
            + (
                kind[..., EFFECT_WRITE, None]
                + kind[..., EFFECT_REPLACE, None]
            )
            * active_pointer
        )
        dense_value = torch.einsum("bks,bkv->bsv", value_mass, value_code)
        dense_value = dense_value / value_mass.sum(1).unsqueeze(-1).clamp_min(1e-7)
        dense_value = torch.where(
            value_mass.sum(1).unsqueeze(-1).gt(0),
            dense_value,
            base.value_code,
        )
        type_mass = (
            kind[..., EFFECT_ALLOCATE, None] * inactive_pointer
            + kind[..., EFFECT_REPLACE, None] * active_pointer
        )
        dense_type = torch.einsum("bks,bkt->bst", type_mass, type_index)
        dense_type = dense_type / type_mass.sum(1).unsqueeze(-1).clamp_min(1e-7)
        dense_type = torch.where(
            type_mass.sum(1).unsqueeze(-1).gt(0),
            dense_type,
            base.type_index,
        )

        link = torch.einsum(
            "bk,bkrst->brst",
            kind[..., EFFECT_LINK],
            relation_link,
        )
        unlink = torch.einsum(
            "bk,bkrst->brst",
            kind[..., EFFECT_UNLINK],
            relation_unlink,
        )
        relation_action = self._dense_actions(link, unlink)

        root_clear = kind[..., EFFECT_ROOT_CLEAR].sum(-1)
        root_set = torch.einsum(
            "bk,bks->bs",
            kind[..., EFFECT_ROOT_SET],
            root_pointer,
        )
        root_total = root_clear + root_set.sum(-1)
        root_normalizer = root_total.clamp_min(1.0).unsqueeze(-1)
        root_action = torch.cat(
            (
                (1.0 - root_total).clamp_min(0.0).unsqueeze(-1),
                root_clear.unsqueeze(-1),
                root_set,
            ),
            dim=-1,
        ) / root_normalizer
        disposition_action = self._dense_actions(
            kind[..., EFFECT_COMMIT].sum(-1),
            kind[..., EFFECT_HALT].sum(-1),
            kind[..., EFFECT_REJECT].sum(-1),
        )

        if hard:
            node_action = _hard_categories(node_action)
            dense_value = _hard_categories(dense_value)
            dense_type = _hard_categories(dense_type)
            relation_action = _hard_categories(relation_action)
            root_action = _hard_categories(root_action)
            disposition_action = _hard_categories(disposition_action)

        node_count = _bernoulli_count_distribution(
            kind[..., EFFECT_ALLOCATE : EFFECT_REPLACE + 1].sum(-1)
        )
        link_count = _bernoulli_count_distribution(kind[..., EFFECT_LINK])
        unlink_count = _bernoulli_count_distribution(kind[..., EFFECT_UNLINK])
        return AtomicTypedEdits(
            node_action=node_action,
            value_code=dense_value,
            type_index=dense_type,
            relation_action=relation_action,
            root_action=root_action,
            disposition_action=disposition_action,
            node_edit_count=node_count,
            relation_link_count=link_count,
            relation_unlink_count=unlink_count,
            effect_kind=kind,
            effect_node_pointer=node_pointer,
            effect_value_code=value_code,
            effect_type_index=type_index,
            effect_relation_link=relation_link,
            effect_relation_unlink=relation_unlink,
            effect_root_pointer=root_pointer,
        )


__all__ = [
    "FactorizedOperationStateTransitionCompiler",
    "OperationEffectSetCompiler",
    "OperationStateTransitionCompiler",
    "OperationStateTransitionTrace",
]
