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
        ):
            raise TheoryReactorError("terminal-state compiler geometry differs")
        self.config = config
        self.width = width
        self.layers_count = layers
        self.num_heads = num_heads
        self.relation_width = relation_width
        self.residual_edits = residual_edits

        self.command_projection = nn.Linear(config.d_model, width)
        self.command_norm = nn.LayerNorm(width)
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

    def forward(
        self,
        state: TypedTheoryState,
        *,
        command_hidden: torch.Tensor,
        command_attention_mask: torch.Tensor,
        steps: int,
        hard: bool,
    ) -> TypedTheoryState:
        validate_state(state, self.config)
        batch = state.value_probabilities.shape[0]
        if (
            not 1 <= steps <= self.config.max_steps - state.step
            or command_hidden.ndim != 3
            or command_hidden.shape[0] != batch
            or command_hidden.shape[-1] != self.config.d_model
            or command_attention_mask.shape != command_hidden.shape[:2]
            or command_attention_mask.dtype != torch.bool
        ):
            raise TheoryReactorError("terminal-state compiler input differs")

        command = self.command_norm(self.command_projection(command_hidden))
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
        if self.residual_edits:
            slots = slots + initial
        for layer in self.layers:
            slots = layer(slots, memory, memory_padding)
        slots = self.output_norm(slots)

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

    def forward(
        self,
        state: TypedTheoryState,
        *,
        steps: int,
        hard: bool = False,
        command_hidden: torch.Tensor | None = None,
        command_attention_mask: torch.Tensor | None = None,
    ) -> tuple[TypedTheoryState, ReactorTrace]:
        if command_hidden is None or command_attention_mask is None:
            raise TheoryReactorError("terminal-state reactor requires COMMAND bytes")
        terminal = self.compiler(
            state,
            command_hidden=command_hidden,
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
    "ParallelTerminalStateCompiler",
    "ParallelTerminalStateReactor",
]
