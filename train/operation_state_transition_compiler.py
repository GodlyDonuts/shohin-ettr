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


ROLE_ANCHORED_EFFECT_ROLES = 4
ROLE_ANCHORED_EFFECT_MOTORS_PER_ROLE = 5
ROLE_ANCHORED_EFFECT_SLOTS = (
    ROLE_ANCHORED_EFFECT_ROLES * ROLE_ANCHORED_EFFECT_MOTORS_PER_ROLE
)
WRITE_RAIL_EFFECT_SLOTS = 3
LINK_RAIL_EFFECT_SLOTS = 10
WRITE_LINK_RAIL_EFFECT_SLOTS = WRITE_RAIL_EFFECT_SLOTS + LINK_RAIL_EFFECT_SLOTS


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
        tuple[torch.Tensor, torch.Tensor] | None,
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
        effect_anchors = None
        effect_role_count = int(getattr(self, "effect_role_count", 0))
        if effect_role_count:
            role_masks, role_valid = self.command_operation_router.effect_role_masks(
                command_tokens,
                document_mask,
                operation_masks,
                maximum_roles=effect_role_count,
            )
            effect_anchors = (
                torch.einsum(
                    "borl,bld->bord",
                    role_masks.to(command.dtype),
                    command,
                ),
                role_valid,
            )
        return (
            command,
            document_mask,
            slots,
            initial_memory,
            operation_masks,
            operation_count,
            effect_anchors,
        )

    def _operation_edits_from_slots(
        self,
        state: TypedTheoryState,
        slots: torch.Tensor,
        *,
        hard: bool,
        effect_anchors: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> AtomicTypedEdits:
        return self._atomic_edits_from_slots(
            state,
            slots,
            hard=hard,
            effect_anchors=effect_anchors,
        )

    def _final_edits_from_slots(
        self,
        state: TypedTheoryState,
        slots: torch.Tensor,
        *,
        hard: bool,
        effect_anchors: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> AtomicTypedEdits:
        return self._atomic_edits_from_slots(
            state,
            slots,
            hard=hard,
            effect_anchors=effect_anchors,
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
            effect_anchors,
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
                operation_masks[:, operation_index].unsqueeze(1).to(command.dtype),
                command,
            )
            updated_slots = self.operation_recurrence(
                slots,
                torch.cat((operation, self._initial_memory(current)), dim=1),
                operation_padding,
            )
            selected = operation_count.gt(operation_index)
            operation_edits = self._operation_edits_from_slots(
                current,
                updated_slots,
                hard=hard,
                effect_anchors=(
                    (
                        effect_anchors[0][:, operation_index],
                        effect_anchors[1][:, operation_index],
                    )
                    if effect_anchors is not None
                    else None
                ),
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
        final_edits = self._final_edits_from_slots(
            current,
            final_slots,
            hard=hard,
            effect_anchors=(
                (
                    (
                        effect_anchors[0]
                        * effect_anchors[1][..., None].to(effect_anchors[0].dtype)
                    ).sum(1)
                    / effect_anchors[1]
                    .sum(1, keepdim=False)[..., None]
                    .clamp_min(1)
                    .to(effect_anchors[0].dtype),
                    effect_anchors[1].any(1),
                )
                if effect_anchors is not None
                else None
            ),
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


class FactorizedOperationStateTransitionCompiler(OperationStateTransitionCompiler):
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
        effect_anchors: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> AtomicTypedEdits:
        del effect_anchors
        base = super()._atomic_edits_from_slots(state, slots, hard=False)
        pooled = slots.mean(1)
        node_count = self.node_edit_count_head(pooled).float().softmax(-1)
        link_count = self.relation_link_count_head(pooled).float().softmax(-1)
        unlink_count = self.relation_unlink_count_head(pooled).float().softmax(-1)

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
    flat_logits = logits.float().reshape(batch, effects, -1)
    flat_mask = mask.reshape(batch, effects, -1)
    probabilities = flat_logits.masked_fill(~flat_mask, -1e9).softmax(-1)
    probabilities = probabilities * flat_mask.to(probabilities.dtype)
    probabilities = probabilities / probabilities.sum(-1, keepdim=True).clamp_min(1e-7)
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
        public_role_anchors: bool = False,
        maximum_effect_roles: int = ROLE_ANCHORED_EFFECT_ROLES,
        explicit_effect_cardinality: bool = False,
        **kwargs,
    ) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        if (
            not isinstance(maximum_effects, int)
            or not 1 <= maximum_effects <= 64
            or not isinstance(public_role_anchors, bool)
            or not isinstance(explicit_effect_cardinality, bool)
            or not isinstance(maximum_effect_roles, int)
            or (
                public_role_anchors
                and (
                    not 2 <= maximum_effect_roles <= maximum_effects
                    or maximum_effects % maximum_effect_roles
                )
            )
        ):
            raise TheoryReactorError("operation effect set geometry differs")
        self.maximum_effects = maximum_effects
        self.public_role_anchors = public_role_anchors
        self.explicit_effect_cardinality = explicit_effect_cardinality
        self.effect_role_count = maximum_effect_roles if public_role_anchors else 0
        self.effect_motors_per_role = (
            maximum_effects // maximum_effect_roles if public_role_anchors else 0
        )
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
        self.effect_activity_head = (
            nn.Linear(self.width, 1) if explicit_effect_cardinality else None
        )
        self.effect_count_head = (
            nn.Linear(self.width, maximum_effects + 1)
            if explicit_effect_cardinality
            else None
        )
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

    def _effect_slots(
        self,
        slots: torch.Tensor,
        effect_anchors: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = slots.shape[0]
        effects = (
            self.effect_queries.to(slots.dtype)
            .unsqueeze(0)
            .expand(
                batch,
                -1,
                -1,
            )
        )
        effects = effects + slots.mean(1, keepdim=True)
        valid = torch.ones(
            batch,
            self.maximum_effects,
            dtype=torch.bool,
            device=slots.device,
        )
        if self.public_role_anchors:
            if effect_anchors is None:
                raise TheoryReactorError("public effect-role anchors are absent")
            anchors, role_valid = effect_anchors
            if (
                anchors.shape != (batch, self.effect_role_count, self.width)
                or role_valid.shape != (batch, self.effect_role_count)
                or role_valid.dtype != torch.bool
            ):
                raise TheoryReactorError("public effect-role anchors differ")
            effect_role = torch.arange(
                self.maximum_effects,
                device=slots.device,
            ).div(self.effect_motors_per_role, rounding_mode="floor")
            effects = effects + anchors.index_select(1, effect_role).to(slots.dtype)
            valid = role_valid.index_select(1, effect_role)
        elif effect_anchors is not None:
            raise TheoryReactorError("unexpected public effect-role anchors")
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
        return self.effect_output_norm(effects), valid

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
        effect_anchors: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> AtomicTypedEdits:
        base = super()._atomic_edits_from_slots(state, slots, hard=False)
        effects, effect_valid = self._effect_slots(slots, effect_anchors)
        kind_logits = self.effect_kind_head(effects).float()
        effect_count = None
        if self.explicit_effect_cardinality:
            if self.effect_activity_head is None or self.effect_count_head is None:
                raise TheoryReactorError("explicit effect-cardinality heads are absent")
            valid_weight = effect_valid[..., None].to(effects.dtype)
            pooled = (effects * valid_weight).sum(1) / valid_weight.sum(1).clamp_min(
                1.0
            )
            effect_count = self.effect_count_head(pooled).float().softmax(-1)
            activity_logits = self.effect_activity_head(effects).float().squeeze(-1)
            selected = (
                _count_constrained_selection(
                    activity_logits,
                    effect_valid,
                    effect_count,
                    capacity=effect_valid.sum(-1),
                    hard=True,
                )
                if hard
                else activity_logits.sigmoid() * effect_valid.to(torch.float32)
            )
            nonnoop = kind_logits[..., 1:].softmax(-1)
            kind = torch.cat(
                (
                    (1.0 - selected).unsqueeze(-1),
                    selected.unsqueeze(-1) * nonnoop,
                ),
                dim=-1,
            )
        else:
            forced_noop = torch.full_like(kind_logits, -1e9)
            forced_noop[..., EFFECT_NOOP] = 0.0
            kind = torch.where(
                effect_valid[..., None],
                kind_logits,
                forced_noop,
            ).softmax(-1)
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
            + (kind[..., EFFECT_WRITE, None] + kind[..., EFFECT_REPLACE, None])
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
        root_action = (
            torch.cat(
                (
                    (1.0 - root_total).clamp_min(0.0).unsqueeze(-1),
                    root_clear.unsqueeze(-1),
                    root_set,
                ),
                dim=-1,
            )
            / root_normalizer
        )
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
            effect_count=effect_count,
        )


class _OperationTypedEffectRail(nn.Module):
    """A typed motor bank conditioned on state slots and public AST roles."""

    def __init__(self, width: int, num_heads: int, motors: int) -> None:
        super().__init__()
        self.motors = motors
        self.queries = nn.Parameter(torch.empty(motors, width))
        self.input_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(
            width,
            num_heads,
            batch_first=True,
        )
        self.memory_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(
            width,
            num_heads,
            batch_first=True,
        )
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )
        self.output_norm = nn.LayerNorm(width)
        nn.init.normal_(self.queries, std=0.02)

    def forward(
        self,
        slots: torch.Tensor,
        effect_anchors: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        anchors, role_valid = effect_anchors
        batch, roles, width = anchors.shape
        if (
            slots.shape[0] != batch
            or slots.shape[-1] != width
            or role_valid.shape != (batch, roles)
            or role_valid.dtype != torch.bool
        ):
            raise TheoryReactorError("typed effect rail anchors differ")
        valid_weight = role_valid[..., None].to(anchors.dtype)
        pooled_anchor = (anchors * valid_weight).sum(1, keepdim=True)
        pooled_anchor = pooled_anchor / valid_weight.sum(1, keepdim=True).clamp_min(1.0)
        effects = self.queries.to(slots.dtype).unsqueeze(0).expand(batch, -1, -1)
        effects = effects + slots.mean(1, keepdim=True) + pooled_anchor.to(slots.dtype)
        normalized = self.input_norm(effects)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        effects = effects + attended
        memory = torch.cat((slots, anchors.to(slots.dtype)), dim=1)
        memory_padding = torch.cat(
            (
                torch.zeros(
                    batch,
                    slots.shape[1],
                    dtype=torch.bool,
                    device=slots.device,
                ),
                ~role_valid,
            ),
            dim=1,
        )
        attended, _ = self.cross_attention(
            self.input_norm(effects),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=memory_padding,
            need_weights=False,
        )
        effects = effects + attended
        effects = effects + self.ff(self.ff_norm(effects))
        return self.output_norm(effects)


class OperationWriteLinkRailCompiler(OperationStateTransitionCompiler):
    """Compile corpus-exact WRITE and LINK rails without a kind classifier."""

    def __init__(
        self,
        *args,
        maximum_effect_roles: int = ROLE_ANCHORED_EFFECT_ROLES,
        **kwargs,
    ) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        if maximum_effect_roles != ROLE_ANCHORED_EFFECT_ROLES:
            raise TheoryReactorError("write/link rail role geometry differs")
        self.maximum_effects = WRITE_LINK_RAIL_EFFECT_SLOTS
        self.effect_role_count = maximum_effect_roles
        self.effect_motors_per_role = 0
        self.write_rail = _OperationTypedEffectRail(
            self.width,
            self.num_heads,
            WRITE_RAIL_EFFECT_SLOTS,
        )
        self.link_rail = _OperationTypedEffectRail(
            self.width,
            self.num_heads,
            LINK_RAIL_EFFECT_SLOTS,
        )
        self.write_count_head = nn.Linear(self.width, WRITE_RAIL_EFFECT_SLOTS + 1)
        self.link_count_head = nn.Linear(self.width, LINK_RAIL_EFFECT_SLOTS + 1)
        self.write_activity_head = nn.Linear(self.width, 1)
        self.link_activity_head = nn.Linear(self.width, 1)
        self.write_node_query = nn.Linear(self.width, self.width, bias=False)
        self.write_node_key = nn.Linear(self.width, self.width, bias=False)
        self.write_value_head = nn.Linear(self.width, self.config.num_value_codes)
        self.link_relation_source = nn.Linear(self.width, self.config.num_slots)
        self.link_relation_target = nn.Linear(self.width, self.config.num_slots)
        self.link_relation_type = nn.Linear(self.width, self.config.num_relations)
        for head in (self.write_count_head, self.link_count_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _link_pointer(
        self,
        state: TypedTheoryState,
        slots: torch.Tensor,
        link_features: torch.Tensor,
        *,
        write_activity: torch.Tensor,
        write_pointer: torch.Tensor,
        write_value: torch.Tensor,
    ) -> torch.Tensor:
        """Bind LINK tuples against the pre-operation state."""

        del slots, write_activity, write_pointer, write_value
        if self.link_relation_source is None or self.link_relation_target is None:
            raise TheoryReactorError("write/link rail endpoint heads are absent")
        relation_logits = (
            self.link_relation_type(link_features).float()[:, :, :, None, None]
            + self.link_relation_source(link_features).float()[:, :, None, :, None]
            + self.link_relation_target(link_features).float()[:, :, None, None, :]
        )
        active = state.active.gt(0.5)
        relations = state.relations.gt(0.5)
        pair_active = active[:, None, :, None] & active[:, None, None, :]
        return _masked_flat_softmax(
            relation_logits,
            ((~relations) & pair_active)[:, None].expand_as(relation_logits),
        )

    def _operation_edits_from_slots(
        self,
        state: TypedTheoryState,
        slots: torch.Tensor,
        *,
        hard: bool,
        effect_anchors: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> AtomicTypedEdits:
        if effect_anchors is None:
            raise TheoryReactorError("write/link rail public anchors are absent")
        base = super()._atomic_edits_from_slots(state, slots, hard=False)
        batch = slots.shape[0]
        write_features = self.write_rail(slots, effect_anchors)
        link_features = self.link_rail(slots, effect_anchors)
        write_count = self.write_count_head(write_features.mean(1)).float().softmax(-1)
        link_count = self.link_count_head(link_features.mean(1)).float().softmax(-1)
        write_valid = torch.ones(
            batch,
            WRITE_RAIL_EFFECT_SLOTS,
            dtype=torch.bool,
            device=slots.device,
        )
        link_valid = torch.ones(
            batch,
            LINK_RAIL_EFFECT_SLOTS,
            dtype=torch.bool,
            device=slots.device,
        )
        write_activity = _count_constrained_selection(
            self.write_activity_head(write_features).float().squeeze(-1),
            write_valid,
            write_count,
            capacity=torch.full(
                (batch,),
                WRITE_RAIL_EFFECT_SLOTS,
                dtype=torch.long,
                device=slots.device,
            ),
            hard=hard,
        )
        link_activity = _count_constrained_selection(
            self.link_activity_head(link_features).float().squeeze(-1),
            link_valid,
            link_count,
            capacity=torch.full(
                (batch,),
                LINK_RAIL_EFFECT_SLOTS,
                dtype=torch.long,
                device=slots.device,
            ),
            hard=hard,
        )

        active = state.active.gt(0.5)
        write_logits = (
            torch.einsum(
                "bkd,bsd->bks",
                self.write_node_query(write_features),
                self.write_node_key(slots),
            )
            / self.width**0.5
        )
        write_pointer = _masked_flat_softmax(
            write_logits,
            active[:, None, :].expand_as(write_logits),
        )
        write_value = self.write_value_head(write_features).float().softmax(-1)

        link_pointer = self._link_pointer(
            state,
            slots,
            link_features,
            write_activity=write_activity,
            write_pointer=write_pointer,
            write_value=write_value,
        )
        if hard:
            write_pointer = _hard_categories(write_pointer)
            write_value = _hard_categories(write_value)
            link_pointer = _hard_categories(
                link_pointer.reshape(batch, LINK_RAIL_EFFECT_SLOTS, -1)
            ).reshape_as(link_pointer)

        write_mass = torch.einsum(
            "bk,bks->bs",
            write_activity,
            write_pointer,
        )
        zeros_node = torch.zeros_like(write_mass)
        node_action = OperationEffectSetCompiler._dense_actions(
            zeros_node,
            write_mass,
            zeros_node,
            zeros_node,
        )
        dense_value = torch.einsum(
            "bk,bks,bkv->bsv",
            write_activity,
            write_pointer,
            write_value,
        )
        dense_value = dense_value / write_mass.unsqueeze(-1).clamp_min(1e-7)
        dense_value = torch.where(
            write_mass.unsqueeze(-1).gt(0),
            dense_value,
            base.value_code,
        )
        link_mass = torch.einsum(
            "bk,bkrst->brst",
            link_activity,
            link_pointer,
        )
        relation_action = OperationEffectSetCompiler._dense_actions(
            link_mass,
            torch.zeros_like(link_mass),
        )
        root_action = torch.zeros_like(base.root_action)
        root_action[..., 0] = 1.0
        disposition_action = torch.zeros_like(base.disposition_action)
        disposition_action[..., 0] = 1.0
        type_index = base.type_index
        if hard:
            node_action = _hard_categories(node_action)
            dense_value = _hard_categories(dense_value)
            type_index = _hard_categories(type_index)
            relation_action = _hard_categories(relation_action)
            write_count = _hard_categories(write_count)
            link_count = _hard_categories(link_count)

        kinds = torch.zeros(
            batch,
            WRITE_LINK_RAIL_EFFECT_SLOTS,
            EFFECT_KIND_COUNT,
            dtype=write_activity.dtype,
            device=slots.device,
        )
        kinds[:, :WRITE_RAIL_EFFECT_SLOTS, EFFECT_NOOP] = 1.0 - write_activity
        kinds[:, :WRITE_RAIL_EFFECT_SLOTS, EFFECT_WRITE] = write_activity
        kinds[:, WRITE_RAIL_EFFECT_SLOTS:, EFFECT_NOOP] = 1.0 - link_activity
        kinds[:, WRITE_RAIL_EFFECT_SLOTS:, EFFECT_LINK] = link_activity
        generic_node_pointer = torch.zeros(
            batch,
            WRITE_LINK_RAIL_EFFECT_SLOTS,
            2,
            self.config.num_slots,
            dtype=write_pointer.dtype,
            device=slots.device,
        )
        generic_node_pointer[:, :WRITE_RAIL_EFFECT_SLOTS, 0] = write_pointer
        generic_node_pointer[:, :WRITE_RAIL_EFFECT_SLOTS, 1] = write_pointer
        generic_value = torch.full(
            (
                batch,
                WRITE_LINK_RAIL_EFFECT_SLOTS,
                self.config.num_value_codes,
            ),
            1.0 / self.config.num_value_codes,
            dtype=write_value.dtype,
            device=slots.device,
        )
        generic_value[:, :WRITE_RAIL_EFFECT_SLOTS] = write_value
        generic_type = torch.full(
            (batch, WRITE_LINK_RAIL_EFFECT_SLOTS, self.config.num_types),
            1.0 / self.config.num_types,
            dtype=write_value.dtype,
            device=slots.device,
        )
        generic_relation = torch.zeros(
            batch,
            WRITE_LINK_RAIL_EFFECT_SLOTS,
            self.config.num_relations,
            self.config.num_slots,
            self.config.num_slots,
            dtype=link_pointer.dtype,
            device=slots.device,
        )
        generic_relation[:, WRITE_RAIL_EFFECT_SLOTS:] = link_pointer
        generic_root = torch.full(
            (batch, WRITE_LINK_RAIL_EFFECT_SLOTS, self.config.num_slots),
            1.0 / self.config.num_slots,
            dtype=write_value.dtype,
            device=slots.device,
        )
        unlink_count = torch.zeros(
            batch,
            self.config.max_edges + 1,
            dtype=link_count.dtype,
            device=slots.device,
        )
        unlink_count[:, 0] = 1.0
        return AtomicTypedEdits(
            node_action=node_action,
            value_code=dense_value,
            type_index=type_index,
            relation_action=relation_action,
            root_action=root_action,
            disposition_action=disposition_action,
            node_edit_count=write_count,
            relation_link_count=link_count,
            relation_unlink_count=unlink_count,
            effect_kind=kinds,
            effect_node_pointer=generic_node_pointer,
            effect_value_code=generic_value,
            effect_type_index=generic_type,
            effect_relation_link=generic_relation,
            effect_relation_unlink=generic_relation,
            effect_root_pointer=generic_root,
        )


class OperationPostWriteLinkRailCompiler(OperationWriteLinkRailCompiler):
    """Bind LINK tuples against the differentiable state after WRITE release."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.link_relation_source = None
        self.link_relation_target = None
        self.link_source_query = nn.Linear(self.width, self.width, bias=False)
        self.link_source_key = nn.Linear(self.width, self.width, bias=False)
        self.link_target_query = nn.Linear(self.width, self.width, bias=False)
        self.link_target_key = nn.Linear(self.width, self.width, bias=False)

    def _link_pointer(
        self,
        state: TypedTheoryState,
        slots: torch.Tensor,
        link_features: torch.Tensor,
        *,
        write_activity: torch.Tensor,
        write_pointer: torch.Tensor,
        write_value: torch.Tensor,
    ) -> torch.Tensor:
        del slots
        write_mass = torch.einsum(
            "bk,bks->bs",
            write_activity,
            write_pointer,
        )
        write_probability = write_mass.clamp(max=1.0).unsqueeze(-1)
        dense_value = torch.einsum(
            "bk,bks,bkv->bsv",
            write_activity,
            write_pointer,
            write_value,
        )
        dense_value = dense_value / write_mass.unsqueeze(-1).clamp_min(1e-7)
        post_value = (
            state.value_probabilities.float() * (1.0 - write_probability)
            + dense_value * write_probability
        )
        post_write_state = TypedTheoryState(
            value_probabilities=post_value.to(state.value_probabilities.dtype),
            type_probabilities=state.type_probabilities,
            relations=state.relations,
            active=state.active,
            root=state.root,
            committed=state.committed,
            halted=state.halted,
            step=state.step,
        )
        post_write_slots = self._initial_memory(post_write_state).to(
            link_features.dtype
        )
        source_logits = torch.einsum(
            "bkd,bsd->bks",
            self.link_source_query(link_features),
            self.link_source_key(post_write_slots),
        )
        target_logits = torch.einsum(
            "bkd,bsd->bks",
            self.link_target_query(link_features),
            self.link_target_key(post_write_slots),
        )
        relation_logits = (
            self.link_relation_type(link_features).float()[:, :, :, None, None]
            + source_logits.float()[:, :, None, :, None]
            + target_logits.float()[:, :, None, None, :]
        )
        active = post_write_state.active.gt(0.5)
        relations = post_write_state.relations.gt(0.5)
        pair_active = active[:, None, :, None] & active[:, None, None, :]
        return _masked_flat_softmax(
            relation_logits,
            ((~relations) & pair_active)[:, None].expand_as(relation_logits),
        )


__all__ = [
    "FactorizedOperationStateTransitionCompiler",
    "OperationEffectSetCompiler",
    "OperationPostWriteLinkRailCompiler",
    "OperationStateTransitionCompiler",
    "OperationStateTransitionTrace",
    "OperationWriteLinkRailCompiler",
]
