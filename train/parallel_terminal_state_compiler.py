"""Compile the query-independent terminal-state quotient in one pass.

Multiple transaction schedules can denote the same terminal typed state.  A
categorical schedule objective therefore spends capacity selecting an
arbitrary serialization and can splice incompatible local decisions.  This
module instead maps the initial typed state plus COMMAND residuals directly
to the complete terminal state consumed by the source-deleted query reader.

The runtime accepts no QUERY bytes, answer labels, oracle program, or host
solver.  It predicts one sticky semantic result and enforces the deployed
typed-state constraints when ``hard=True``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from endogenous_typed_theory_reactor import (
    ReactorTrace,
    TRANSACTION_COUNT,
    TheoryReactorConfig,
    TheoryReactorError,
    TypedTheoryState,
    validate_deployed_state,
    validate_state,
)
from token_native_syntax_router import (
    CoverVerifiedTokenNativeDocumentMask,
    TokenNativeDocumentMask,
    TokenNativeOccurrenceEncoder,
    TokenNativeOperationRouter,
    TokenNativeSyntaxGraphEncoder,
)


def _hard_one_hot(probabilities: torch.Tensor) -> torch.Tensor:
    return F.one_hot(
        probabilities.argmax(-1),
        probabilities.shape[-1],
    ).to(probabilities.dtype)


def _hard_binary(probabilities: torch.Tensor) -> torch.Tensor:
    return probabilities.ge(0.5).to(probabilities.dtype)


def _hard_capped_relations(
    probabilities: torch.Tensor,
    *,
    maximum: int,
) -> torch.Tensor:
    """Threshold sparse relations, retaining only the strongest allowed edges."""

    batch = probabilities.shape[0]
    flat = probabilities.reshape(batch, -1)
    binary = flat.ge(0.5)
    if flat.shape[1] > maximum:
        indices = flat.topk(maximum, dim=-1).indices
        retained = torch.zeros_like(binary)
        retained.scatter_(1, indices, True)
        binary &= retained
    return binary.reshape_as(probabilities).to(probabilities.dtype)


NODE_EDIT_COUNT = 5
RELATION_EDIT_COUNT = 3
ROOT_EDIT_PREFIX_COUNT = 2
DISPOSITION_EDIT_COUNT = 4


@dataclass(frozen=True, slots=True)
class AtomicTypedEdits:
    """One parallel, coherent state difference consumed by fixed algebra."""

    node_action: torch.Tensor
    value_code: torch.Tensor
    type_index: torch.Tensor
    relation_action: torch.Tensor
    root_action: torch.Tensor
    disposition_action: torch.Tensor
    node_edit_count: torch.Tensor | None = None
    relation_link_count: torch.Tensor | None = None
    relation_unlink_count: torch.Tensor | None = None
    effect_kind: torch.Tensor | None = None
    effect_node_pointer: torch.Tensor | None = None
    effect_value_code: torch.Tensor | None = None
    effect_type_index: torch.Tensor | None = None
    effect_relation_link: torch.Tensor | None = None
    effect_relation_unlink: torch.Tensor | None = None
    effect_root_pointer: torch.Tensor | None = None


class _TerminalStateLayer(nn.Module):
    """Jointly update terminal slots from slots, initial state, and COMMAND."""

    def __init__(self, width: int, num_heads: int) -> None:
        super().__init__()
        self.slot_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(
            width,
            num_heads,
            batch_first=True,
        )
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

    def forward(
        self,
        slots: torch.Tensor,
        memory: torch.Tensor,
        memory_padding: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.slot_norm(slots)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        slots = slots + attended
        attended, _ = self.cross_attention(
            self.slot_norm(slots),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=memory_padding,
            need_weights=False,
        )
        slots = slots + attended
        return slots + self.ff(self.ff_norm(slots))


class ParallelTerminalStateCompiler(nn.Module):
    """Predict an absolute terminal typed state without serial transactions."""

    def __init__(
        self,
        config: TheoryReactorConfig,
        *,
        width: int = 512,
        layers: int = 4,
        num_heads: int = 8,
        relation_width: int = 64,
        residual_edits: bool = False,
        atomic_edits: bool = False,
        lexical_command: bool = False,
        token_native_command_mask: bool = False,
        cover_verified_command_mask: bool = False,
        token_native_occurrence_command: bool = False,
        token_native_syntax_graph_command: bool = False,
        token_native_declaration_binding_command: bool = False,
        token_native_operation_recurrence_command: bool = False,
        token_native_codebook_ids: Sequence[int] | None = None,
        token_native_codebook_atoms: Sequence[str] | None = None,
        token_native_vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        if (
            not isinstance(width, int)
            or width < 64
            or not isinstance(num_heads, int)
            or num_heads < 1
            or width % num_heads
            or not isinstance(layers, int)
            or not 1 <= layers <= 8
            or not isinstance(relation_width, int)
            or relation_width < 8
            or not isinstance(residual_edits, bool)
            or not isinstance(atomic_edits, bool)
            or not isinstance(lexical_command, bool)
            or not isinstance(token_native_command_mask, bool)
            or not isinstance(cover_verified_command_mask, bool)
            or not isinstance(token_native_occurrence_command, bool)
            or not isinstance(token_native_syntax_graph_command, bool)
            or not isinstance(token_native_declaration_binding_command, bool)
            or not isinstance(token_native_operation_recurrence_command, bool)
            or (residual_edits and atomic_edits)
            or (
                token_native_occurrence_command
                and not token_native_command_mask
            )
            or (cover_verified_command_mask and not token_native_command_mask)
            or (
                token_native_syntax_graph_command
                and not token_native_command_mask
            )
            or (
                token_native_occurrence_command
                and token_native_syntax_graph_command
            )
            or (
                token_native_declaration_binding_command
                and not token_native_syntax_graph_command
            )
            or (
                token_native_operation_recurrence_command
                and not token_native_declaration_binding_command
            )
            or (
                token_native_command_mask
                and (
                    not atomic_edits
                    or not lexical_command
                    or token_native_codebook_ids is None
                    or token_native_vocab_size is None
                    or (
                        cover_verified_command_mask
                        and token_native_codebook_atoms is None
                    )
                )
            )
            or (
                not token_native_command_mask
                and (
                    token_native_codebook_ids is not None
                    or token_native_codebook_atoms is not None
                    or token_native_vocab_size is not None
                )
            )
        ):
            raise TheoryReactorError("terminal-state compiler geometry differs")
        self.config = config
        self.width = width
        self.layers_count = layers
        self.num_heads = num_heads
        self.relation_width = relation_width
        self.residual_edits = residual_edits
        self.atomic_edits = atomic_edits
        self.lexical_command = lexical_command
        self.token_native_command_mask = token_native_command_mask
        self.cover_verified_command_mask = cover_verified_command_mask
        self.token_native_occurrence_command = (
            token_native_occurrence_command
        )
        self.token_native_syntax_graph_command = token_native_syntax_graph_command
        self.token_native_declaration_binding_command = (
            token_native_declaration_binding_command
        )
        self.token_native_operation_recurrence_command = (
            token_native_operation_recurrence_command
        )

        self.command_projection = nn.Linear(config.d_model, width)
        self.command_lexical_projection = (
            nn.Linear(config.d_model, width, bias=False)
            if lexical_command
            else None
        )
        self.command_norm = nn.LayerNorm(width)
        self.command_document_mask = (
            CoverVerifiedTokenNativeDocumentMask(
                token_native_codebook_ids,
                token_native_codebook_atoms,
                vocab_size=token_native_vocab_size,
            )
            if cover_verified_command_mask
            else TokenNativeDocumentMask(
                token_native_codebook_ids,
                vocab_size=token_native_vocab_size,
            )
            if token_native_command_mask
            else None
        )
        self.command_occurrence_encoder = (
            TokenNativeOccurrenceEncoder(
                token_native_codebook_ids,
                vocab_size=token_native_vocab_size,
                width=width,
                num_heads=num_heads,
                maximum_positions=96,
                maximum_identifier_codes=96,
            )
            if token_native_occurrence_command
            else None
        )
        self.command_syntax_graph_encoder = (
            TokenNativeSyntaxGraphEncoder(
                token_native_codebook_ids,
                vocab_size=token_native_vocab_size,
                width=width,
                layers=layers,
                maximum_positions=96,
                maximum_identifier_codes=96,
                resolve_declarations=token_native_declaration_binding_command,
            )
            if token_native_syntax_graph_command
            else None
        )
        self.command_operation_router = (
            TokenNativeOperationRouter(
                token_native_codebook_ids,
                vocab_size=token_native_vocab_size,
                maximum_positions=96,
                maximum_operations=6,
            )
            if token_native_operation_recurrence_command
            else None
        )
        self.value_embedding = nn.Parameter(
            torch.empty(config.num_value_codes, width)
        )
        self.type_embedding = nn.Parameter(torch.empty(config.num_types, width))
        self.initial_slot_embedding = nn.Parameter(
            torch.empty(config.num_slots, width)
        )
        self.terminal_slot_queries = nn.Parameter(
            torch.empty(config.num_slots, width)
        )
        self.active_projection = nn.Linear(1, width, bias=False)
        self.root_projection = nn.Linear(1, width, bias=False)
        self.status_projection = nn.Linear(2, width, bias=False)
        self.relation_summary_projection = nn.Linear(
            2 * config.num_relations,
            width,
            bias=False,
        )
        self.initial_state_norm = nn.LayerNorm(width)
        self.layers = nn.ModuleList(
            _TerminalStateLayer(width, num_heads) for _ in range(layers)
        )
        self.operation_recurrence = (
            _TerminalStateLayer(width, num_heads)
            if token_native_operation_recurrence_command
            else None
        )
        self.output_norm = nn.LayerNorm(width)
        self.value_head = nn.Linear(width, config.num_value_codes)
        self.type_head = nn.Linear(width, config.num_types)
        self.active_head = nn.Linear(width, 1)
        self.root_head = nn.Linear(width, 1)
        self.no_root_head = nn.Linear(width, 1)
        self.relation_left = nn.Linear(
            width,
            config.num_relations * relation_width,
            bias=False,
        )
        self.relation_right = nn.Linear(
            width,
            config.num_relations * relation_width,
            bias=False,
        )
        self.relation_bias = nn.Parameter(torch.zeros(config.num_relations))
        self.status_head = nn.Linear(width, 2)
        if residual_edits:
            self.value_edit_head = nn.Linear(width, 1)
            self.type_edit_head = nn.Linear(width, 1)
            self.active_edit_head = nn.Linear(width, 1)
            self.root_edit_head = nn.Linear(width, 1)
            self.relation_edit_left = nn.Linear(
                width,
                config.num_relations * relation_width,
                bias=False,
            )
            self.relation_edit_right = nn.Linear(
                width,
                config.num_relations * relation_width,
                bias=False,
            )
            self.relation_edit_bias = nn.Parameter(
                torch.full((config.num_relations,), -2.0)
            )
            self.status_edit_head = nn.Linear(width, 2)
        else:
            self.value_edit_head = None
            self.type_edit_head = None
            self.active_edit_head = None
            self.root_edit_head = None
            self.relation_edit_left = None
            self.relation_edit_right = None
            self.register_parameter("relation_edit_bias", None)
            self.status_edit_head = None
        if atomic_edits:
            self.node_action_head = nn.Linear(width, NODE_EDIT_COUNT)
            self.relation_unlink_left = nn.Linear(
                width,
                config.num_relations * relation_width,
                bias=False,
            )
            self.relation_unlink_right = nn.Linear(
                width,
                config.num_relations * relation_width,
                bias=False,
            )
            self.relation_action_bias = nn.Parameter(
                torch.empty(config.num_relations, RELATION_EDIT_COUNT)
            )
            self.root_control_head = nn.Linear(width, ROOT_EDIT_PREFIX_COUNT)
            self.disposition_action_head = nn.Linear(
                width,
                DISPOSITION_EDIT_COUNT,
            )
        else:
            self.node_action_head = None
            self.relation_unlink_left = None
            self.relation_unlink_right = None
            self.register_parameter("relation_action_bias", None)
            self.root_control_head = None
            self.disposition_action_head = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in (
            self.value_embedding,
            self.type_embedding,
            self.initial_slot_embedding,
            self.terminal_slot_queries,
        ):
            nn.init.normal_(parameter, std=0.02)
        for head in (
            self.value_edit_head,
            self.type_edit_head,
            self.active_edit_head,
            self.root_edit_head,
            self.status_edit_head,
        ):
            if head is not None:
                nn.init.constant_(head.bias, -2.0)
        if self.node_action_head is not None:
            nn.init.zeros_(self.node_action_head.bias)
            self.node_action_head.bias.data[0] = 3.0
            self.node_action_head.bias.data[1:] = -3.0
        if self.relation_action_bias is not None:
            nn.init.constant_(self.relation_action_bias, -3.0)
            self.relation_action_bias.data[:, 0] = 3.0
        if self.root_control_head is not None:
            nn.init.zeros_(self.root_control_head.bias)
            self.root_control_head.bias.data[0] = 3.0
            self.root_control_head.bias.data[1] = -3.0
        if self.disposition_action_head is not None:
            nn.init.zeros_(self.disposition_action_head.bias)
            self.disposition_action_head.bias.data[0] = 3.0
            self.disposition_action_head.bias.data[1:] = -3.0

    def _initial_memory(self, state: TypedTheoryState) -> torch.Tensor:
        values = torch.einsum(
            "bsc,cw->bsw",
            state.value_probabilities,
            self.value_embedding,
        )
        types = torch.einsum(
            "bst,tw->bsw",
            state.type_probabilities,
            self.type_embedding,
        )
        outgoing = state.relations.sum(-1).transpose(1, 2)
        incoming = state.relations.sum(-2).transpose(1, 2)
        relations = self.relation_summary_projection(
            torch.cat((incoming, outgoing), dim=-1)
        )
        status = torch.stack((state.committed, state.halted), dim=-1)
        memory = (
            values
            + types
            + relations
            + self.active_projection(state.active.unsqueeze(-1))
            + self.root_projection(state.root.unsqueeze(-1))
            + self.status_projection(status).unsqueeze(1)
            + self.initial_slot_embedding.unsqueeze(0)
        )
        return self.initial_state_norm(memory)

    def _encode_slots(
        self,
        state: TypedTheoryState,
        *,
        command_hidden: torch.Tensor,
        command_lexical: torch.Tensor | None,
        command_tokens: torch.Tensor | None,
        command_attention_mask: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        validate_state(state, self.config)
        batch = state.value_probabilities.shape[0]
        if (
            not 1 <= steps <= self.config.max_steps - state.step
            or command_hidden.ndim != 3
            or command_hidden.shape[0] != batch
            or command_hidden.shape[-1] != self.config.d_model
            or (
                self.lexical_command
                and (
                    command_lexical is None
                    or command_lexical.shape != command_hidden.shape
                )
            )
            or (
                not self.lexical_command
                and command_lexical is not None
            )
            or (
                self.token_native_command_mask
                and (
                    command_tokens is None
                    or command_tokens.shape != command_hidden.shape[:2]
                    or command_tokens.dtype != torch.long
                )
            )
            or (
                not self.token_native_command_mask
                and command_tokens is not None
            )
            or command_attention_mask.shape != command_hidden.shape[:2]
            or command_attention_mask.dtype != torch.bool
        ):
            raise TheoryReactorError("terminal-state compiler input differs")

        if self.command_document_mask is not None:
            if command_tokens is None:
                raise TheoryReactorError(
                    "terminal-state compiler COMMAND tokens are absent"
                )
            command_attention_mask = self.command_document_mask(
                command_tokens,
                command_attention_mask,
            )

        command = self.command_projection(command_hidden)
        if self.command_lexical_projection is not None:
            if command_lexical is None:
                raise TheoryReactorError(
                    "terminal-state compiler lexical COMMAND is absent"
                )
            command = command + self.command_lexical_projection(command_lexical)
        command = self.command_norm(command)
        if self.command_occurrence_encoder is not None:
            if command_tokens is None:
                raise TheoryReactorError(
                    "terminal-state compiler COMMAND tokens are absent"
                )
            command = self.command_occurrence_encoder(
                command,
                command_tokens,
                command_attention_mask,
            )
        if self.command_syntax_graph_encoder is not None:
            if command_tokens is None:
                raise TheoryReactorError(
                    "terminal-state compiler COMMAND tokens are absent"
                )
            command = self.command_syntax_graph_encoder(
                command,
                command_tokens,
                command_attention_mask,
            )
        initial = self._initial_memory(state)
        memory = torch.cat((command, initial), dim=1)
        memory_padding = torch.cat(
            (
                ~command_attention_mask,
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
        slots = slots.unsqueeze(0).expand(batch, -1, -1)
        if self.residual_edits or self.atomic_edits:
            slots = slots + initial
        for layer in self.layers:
            slots = layer(slots, memory, memory_padding)
        if self.command_operation_router is not None:
            if command_tokens is None or self.operation_recurrence is None:
                raise TheoryReactorError(
                    "terminal-state operation recurrence input differs"
                )
            operation_masks, operation_count = self.command_operation_router(
                command_tokens,
                command_attention_mask,
            )
            operation_padding = torch.zeros(
                batch,
                1 + self.config.num_slots,
                dtype=torch.bool,
                device=command.device,
            )
            for operation_index in range(operation_masks.shape[1]):
                operation = torch.bmm(
                    operation_masks[:, operation_index]
                    .unsqueeze(1)
                    .to(command.dtype),
                    command,
                )
                updated = self.operation_recurrence(
                    slots,
                    torch.cat((operation, initial), dim=1),
                    operation_padding,
                )
                active = operation_count.gt(operation_index).view(batch, 1, 1)
                slots = torch.where(active, updated, slots)
        return self.output_norm(slots)

    def _atomic_edits_from_slots(
        self,
        state: TypedTheoryState,
        slots: torch.Tensor,
        *,
        hard: bool,
        effect_anchors: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> AtomicTypedEdits:
        if (
            not self.atomic_edits
            or self.node_action_head is None
            or self.relation_unlink_left is None
            or self.relation_unlink_right is None
            or self.relation_action_bias is None
            or self.root_control_head is None
            or self.disposition_action_head is None
        ):
            raise TheoryReactorError("atomic typed-edit path differs")
        del effect_anchors
        batch = slots.shape[0]
        node_action = self.node_action_head(slots).float().softmax(-1)
        value_code = self.value_head(slots).float().softmax(-1)
        type_index = self.type_head(slots).float().softmax(-1)

        left = self.relation_left(slots).view(
            batch,
            self.config.num_slots,
            self.config.num_relations,
            self.relation_width,
        )
        right = self.relation_right(slots).view_as(left)
        link_logits = torch.einsum(
            "bsrd,btrd->brst",
            left,
            right,
        ) / math.sqrt(self.relation_width)
        unlink_left = self.relation_unlink_left(slots).view_as(left)
        unlink_right = self.relation_unlink_right(slots).view_as(right)
        unlink_logits = torch.einsum(
            "bsrd,btrd->brst",
            unlink_left,
            unlink_right,
        ) / math.sqrt(self.relation_width)
        keep_logits = self.relation_action_bias[:, 0].view(1, -1, 1, 1)
        keep_logits = keep_logits.expand_as(link_logits)
        relation_logits = torch.stack(
            (
                keep_logits,
                link_logits
                + self.relation_action_bias[:, 1].view(1, -1, 1, 1),
                unlink_logits
                + self.relation_action_bias[:, 2].view(1, -1, 1, 1),
            ),
            dim=-1,
        )
        relation_action = relation_logits.float().softmax(-1)

        keep, allocate, _write, clear, _replace = node_action.unbind(-1)
        del keep
        initial_active = state.active.float()
        projected_active = (
            initial_active + allocate * (1.0 - initial_active)
        ) * (1.0 - clear * initial_active)
        pooled = slots.mean(1)
        root_prefix = self.root_control_head(pooled).float()
        root_set = self.root_head(slots).float().squeeze(-1)
        root_set = root_set + projected_active.clamp_min(1e-4).log()
        root_action = torch.cat((root_prefix, root_set), dim=-1).softmax(-1)
        disposition_action = self.disposition_action_head(pooled).float().softmax(-1)

        if hard:
            node_action = _hard_one_hot(node_action)
            value_code = _hard_one_hot(value_code)
            type_index = _hard_one_hot(type_index)
            relation_action = _hard_one_hot(relation_action)
            root_action = _hard_one_hot(root_action)
            disposition_action = _hard_one_hot(disposition_action)
        return AtomicTypedEdits(
            node_action=node_action,
            value_code=value_code,
            type_index=type_index,
            relation_action=relation_action,
            root_action=root_action,
            disposition_action=disposition_action,
        )

    def apply_atomic_edits(
        self,
        state: TypedTheoryState,
        edits: AtomicTypedEdits,
        *,
        steps: int,
        hard: bool,
    ) -> TypedTheoryState:
        """Apply one parallel typed difference with no learned executor."""

        keep, allocate, write, clear, replace = edits.node_action.float().unbind(-1)
        del keep
        initial_active = state.active.float()
        allocated = allocate * (1.0 - initial_active)
        cleared = clear * initial_active
        replaced = replace * initial_active
        active = (initial_active + allocated) * (1.0 - cleared)

        type_write = (allocated + replaced).clamp(max=1.0).unsqueeze(-1)
        type_probability = (
            state.type_probabilities.float() * (1.0 - type_write)
            + edits.type_index.float() * type_write
        )
        type_probability = type_probability * (1.0 - cleared.unsqueeze(-1))
        value_write = (
            write * initial_active + allocated + replaced
        ).clamp(max=1.0).unsqueeze(-1)
        value = (
            state.value_probabilities.float() * (1.0 - value_write)
            + edits.value_code.float() * value_write
        )
        value = value * (1.0 - cleared.unsqueeze(-1))

        _relation_keep, link, unlink = edits.relation_action.float().unbind(-1)
        relations = state.relations.float()
        relations = relations + link * (1.0 - relations)
        relations = relations * (1.0 - unlink)
        pair_active = active[:, None, :, None] * active[:, None, None, :]
        relations = relations * pair_active
        if hard:
            relations = _hard_capped_relations(
                relations,
                maximum=self.config.max_edges,
            ) * pair_active

        root_keep = edits.root_action[:, 0:1].float()
        root_set = edits.root_action[:, ROOT_EDIT_PREFIX_COUNT:].float()
        root = (root_keep * state.root.float() + root_set) * active
        status_keep, commit, halt, reject = (
            edits.disposition_action.float().unbind(-1)
        )
        del status_keep
        open_state = (1.0 - state.committed.float()) * (
            1.0 - state.halted.float()
        )
        committed = state.committed.float() + open_state * (commit + reject)
        halted = state.halted.float() + open_state * (halt + reject)

        if hard:
            active = _hard_binary(active)
            value = _hard_one_hot(value) * active.unsqueeze(-1)
            type_probability = _hard_one_hot(type_probability) * active.unsqueeze(-1)
            root = root * active
            committed = _hard_binary(committed)
            halted = _hard_binary(halted)

        terminal = TypedTheoryState(
            value_probabilities=value.to(state.value_probabilities.dtype),
            type_probabilities=type_probability.to(state.value_probabilities.dtype),
            relations=relations.to(state.value_probabilities.dtype),
            active=active.to(state.value_probabilities.dtype),
            root=root.to(state.value_probabilities.dtype),
            committed=committed.to(state.value_probabilities.dtype),
            halted=halted.to(state.value_probabilities.dtype),
            step=state.step + steps,
        )
        if hard:
            validate_deployed_state(terminal, self.config)
        else:
            validate_state(terminal, self.config)
        return terminal

    def forward_with_atomic_edits(
        self,
        state: TypedTheoryState,
        *,
        command_hidden: torch.Tensor,
        command_lexical: torch.Tensor | None = None,
        command_tokens: torch.Tensor | None = None,
        command_attention_mask: torch.Tensor,
        steps: int,
        hard: bool,
    ) -> tuple[TypedTheoryState, AtomicTypedEdits]:
        slots = self._encode_slots(
            state,
            command_hidden=command_hidden,
            command_lexical=command_lexical,
            command_tokens=command_tokens,
            command_attention_mask=command_attention_mask,
            steps=steps,
        )
        edits = self._atomic_edits_from_slots(state, slots, hard=hard)
        return (
            self.apply_atomic_edits(state, edits, steps=steps, hard=hard),
            edits,
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
        slots = self._encode_slots(
            state,
            command_hidden=command_hidden,
            command_lexical=command_lexical,
            command_tokens=command_tokens,
            command_attention_mask=command_attention_mask,
            steps=steps,
        )
        if self.atomic_edits:
            edits = self._atomic_edits_from_slots(state, slots, hard=hard)
            return self.apply_atomic_edits(
                state,
                edits,
                steps=steps,
                hard=hard,
            )
        batch = state.value_probabilities.shape[0]

        value = self.value_head(slots).float().softmax(-1)
        type_probability = self.type_head(slots).float().softmax(-1)
        active = self.active_head(slots).float().sigmoid().squeeze(-1)

        root_slot_logits = self.root_head(slots).float().squeeze(-1)
        pooled = slots.mean(1)
        no_root_logit = self.no_root_head(pooled).float()
        root_with_none = torch.cat((root_slot_logits, no_root_logit), dim=-1).softmax(-1)
        root = root_with_none[:, :-1]

        left = self.relation_left(slots).view(
            batch,
            self.config.num_slots,
            self.config.num_relations,
            self.relation_width,
        )
        right = self.relation_right(slots).view_as(left)
        relation_logits = torch.einsum(
            "bsrd,btrd->brst",
            left,
            right,
        ) / math.sqrt(self.relation_width)
        relation_logits = relation_logits + self.relation_bias.view(1, -1, 1, 1)
        relations = relation_logits.float().sigmoid()
        status = self.status_head(pooled).float().sigmoid()
        committed, halted = status.unbind(-1)

        if self.residual_edits:
            if (
                self.value_edit_head is None
                or self.type_edit_head is None
                or self.active_edit_head is None
                or self.root_edit_head is None
                or self.relation_edit_left is None
                or self.relation_edit_right is None
                or self.relation_edit_bias is None
                or self.status_edit_head is None
            ):
                raise TheoryReactorError(
                    "terminal-state residual edit path differs"
                )
            value_gate = self.value_edit_head(slots).float().sigmoid()
            type_gate = self.type_edit_head(slots).float().sigmoid()
            active_gate = (
                self.active_edit_head(slots).float().sigmoid().squeeze(-1)
            )
            root_gate = self.root_edit_head(pooled).float().sigmoid()
            edit_left = self.relation_edit_left(slots).view_as(left)
            edit_right = self.relation_edit_right(slots).view_as(right)
            relation_edit_logits = torch.einsum(
                "bsrd,btrd->brst",
                edit_left,
                edit_right,
            ) / math.sqrt(self.relation_width)
            relation_edit_logits = relation_edit_logits + (
                self.relation_edit_bias.view(1, -1, 1, 1)
            )
            relation_gate = relation_edit_logits.float().sigmoid()
            status_gate = self.status_edit_head(pooled).float().sigmoid()
            value = torch.lerp(
                state.value_probabilities.float(),
                value,
                value_gate,
            )
            type_probability = torch.lerp(
                state.type_probabilities.float(),
                type_probability,
                type_gate,
            )
            active = torch.lerp(state.active.float(), active, active_gate)
            root = torch.lerp(state.root.float(), root, root_gate)
            relations = torch.lerp(
                state.relations.float(),
                relations,
                relation_gate,
            )
            initial_status = torch.stack(
                (state.committed, state.halted),
                dim=-1,
            ).float()
            status = torch.lerp(initial_status, status, status_gate)
            committed, halted = status.unbind(-1)

        if hard:
            active = _hard_binary(active)
            value = _hard_one_hot(value)
            type_probability = _hard_one_hot(type_probability)
            no_root = (1.0 - root.sum(-1, keepdim=True)).clamp(0.0, 1.0)
            root_choice = _hard_one_hot(torch.cat((root, no_root), dim=-1))
            root = root_choice[:, :-1]

        value = value * active.unsqueeze(-1)
        type_probability = type_probability * active.unsqueeze(-1)
        root = root * active
        pair_active = active[:, None, :, None] * active[:, None, None, :]
        relations = relations * pair_active
        if hard:
            relations = _hard_capped_relations(
                relations,
                maximum=self.config.max_edges,
            ) * pair_active
            committed = _hard_binary(committed)
            halted = _hard_binary(halted)

        terminal = TypedTheoryState(
            value_probabilities=value.to(state.value_probabilities.dtype),
            type_probabilities=type_probability.to(state.value_probabilities.dtype),
            relations=relations.to(state.value_probabilities.dtype),
            active=active.to(state.value_probabilities.dtype),
            root=root.to(state.value_probabilities.dtype),
            committed=committed.to(state.value_probabilities.dtype),
            halted=halted.to(state.value_probabilities.dtype),
            step=state.step + steps,
        )
        if hard:
            validate_deployed_state(terminal, self.config)
        else:
            validate_state(terminal, self.config)
        return terminal


class ParallelTerminalStateReactor(nn.Module):
    """Model-compatible direct semantic transport with no transaction policy."""

    def __init__(
        self,
        compiler: ParallelTerminalStateCompiler,
        config: TheoryReactorConfig,
    ) -> None:
        super().__init__()
        config.validate()
        if compiler.config != config:
            raise TheoryReactorError("terminal-state reactor config differs")
        self.config = config
        self.compiler = compiler
        self.requires_command_lexical = compiler.lexical_command
        self.requires_command_tokens = compiler.token_native_command_mask

    def forward(
        self,
        state: TypedTheoryState,
        *,
        steps: int,
        hard: bool = False,
        command_hidden: torch.Tensor | None = None,
        command_lexical: torch.Tensor | None = None,
        command_tokens: torch.Tensor | None = None,
        command_attention_mask: torch.Tensor | None = None,
    ) -> tuple[TypedTheoryState, ReactorTrace]:
        if command_hidden is None or command_attention_mask is None:
            raise TheoryReactorError("terminal-state reactor requires COMMAND bytes")
        terminal = self.compiler(
            state,
            command_hidden=command_hidden,
            command_lexical=command_lexical,
            command_tokens=command_tokens,
            command_attention_mask=command_attention_mask.bool(),
            steps=steps,
            hard=hard,
        )
        batch = state.active.shape[0]
        dtype = state.active.dtype
        device = state.active.device

        def zeros(classes: int) -> torch.Tensor:
            return torch.zeros(batch, steps, classes, dtype=dtype, device=device)

        # Direct quotient transport deliberately makes no transaction claim.
        return terminal, ReactorTrace(
            opcode=zeros(TRANSACTION_COUNT),
            source=zeros(self.config.num_slots),
            target=zeros(self.config.num_slots),
            relation=zeros(self.config.num_relations),
            type_index=zeros(self.config.num_types),
            value_code=zeros(self.config.num_value_codes),
            applied_opcode=zeros(TRANSACTION_COUNT),
            applied_source=zeros(self.config.num_slots),
            applied_target=zeros(self.config.num_slots),
            applied_relation=zeros(self.config.num_relations),
            applied_type_index=zeros(self.config.num_types),
            applied_value_code=zeros(self.config.num_value_codes),
            active=terminal.active.unsqueeze(1).expand(-1, steps, -1),
            committed=terminal.committed.unsqueeze(1).expand(-1, steps),
            halted=terminal.halted.unsqueeze(1).expand(-1, steps),
        )


__all__ = [
    "AtomicTypedEdits",
    "DISPOSITION_EDIT_COUNT",
    "NODE_EDIT_COUNT",
    "ParallelTerminalStateCompiler",
    "ParallelTerminalStateReactor",
    "RELATION_EDIT_COUNT",
    "ROOT_EDIT_PREFIX_COUNT",
]
