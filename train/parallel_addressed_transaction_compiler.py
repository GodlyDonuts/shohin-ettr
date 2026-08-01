"""Compile a complete addressed ETTR transaction schedule in one pass.

The recurrent ETTR controller is trained with oracle previous states but is
deployed on its own hard choices.  This module removes that exposure boundary:
it reads the initial typed state and COMMAND bytes once, emits one sticky
categorical schedule, and delegates every state mutation to the existing exact
transaction algebra.  No query bytes or answer labels enter the module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from endogenous_typed_theory_reactor import (
    GenericTransactionReactor,
    ReactorTrace,
    TRANSACTION_COUNT,
    TheoryReactorConfig,
    TheoryReactorError,
    TransactionPolicy,
    TypedTheoryState,
)


@dataclass(frozen=True, slots=True)
class AddressedSchedule:
    """Per-step categorical distributions and their applied choices."""

    opcode: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    relation: torch.Tensor
    type_index: torch.Tensor
    value_code: torch.Tensor
    applied_opcode: torch.Tensor
    applied_source: torch.Tensor
    applied_target: torch.Tensor
    applied_relation: torch.Tensor
    applied_type_index: torch.Tensor
    applied_value_code: torch.Tensor

    def policy(self, step: int) -> TransactionPolicy:
        if not 0 <= step < self.opcode.shape[1]:
            raise TheoryReactorError("addressed schedule step differs")
        values = {
            "opcode": self.applied_opcode[:, step],
            "source": self.applied_source[:, step],
            "target": self.applied_target[:, step],
            "relation": self.applied_relation[:, step],
            "type_index": self.applied_type_index[:, step],
            "value_code": self.applied_value_code[:, step],
        }
        return TransactionPolicy(
            **values,
            opcode_probabilities=self.opcode[:, step],
            source_probabilities=self.source[:, step],
            target_probabilities=self.target[:, step],
            relation_probabilities=self.relation[:, step],
            type_probabilities=self.type_index[:, step],
            value_probabilities=self.value_code[:, step],
        )


def _hard_one_hot(probabilities: torch.Tensor) -> torch.Tensor:
    indices = probabilities.argmax(-1)
    return F.one_hot(indices, probabilities.shape[-1]).to(probabilities.dtype)


class ParallelAddressedTransactionCompiler(nn.Module):
    """Compile fixed-address transactions without recurrent teacher forcing."""

    def __init__(
        self,
        config: TheoryReactorConfig,
        *,
        width: int = 384,
        layers: int = 3,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        config.validate()
        if (
            not isinstance(width, int)
            or width < 64
            or width % num_heads
            or not isinstance(layers, int)
            or not 1 <= layers <= 8
            or not isinstance(num_heads, int)
            or num_heads < 1
        ):
            raise TheoryReactorError("addressed schedule geometry differs")
        self.config = config
        self.width = width
        self.layers = layers
        self.num_heads = num_heads

        self.command_projection = nn.Linear(config.d_model, width)
        self.command_norm = nn.LayerNorm(width)
        self.value_embedding = nn.Parameter(
            torch.empty(config.num_value_codes, width)
        )
        self.type_embedding = nn.Parameter(torch.empty(config.num_types, width))
        self.slot_embedding = nn.Parameter(torch.empty(config.num_slots, width))
        self.active_projection = nn.Linear(1, width, bias=False)
        self.root_projection = nn.Linear(1, width, bias=False)
        self.relation_summary_projection = nn.Linear(
            2 * config.num_relations,
            width,
            bias=False,
        )
        self.state_norm = nn.LayerNorm(width)
        self.step_queries = nn.Parameter(torch.empty(config.max_steps, width))
        self.cross_attention = nn.MultiheadAttention(
            width,
            num_heads,
            batch_first=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=num_heads,
            dim_feedforward=4 * width,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.schedule_core = nn.TransformerEncoder(
            layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(width)
        self.opcode_head = nn.Linear(width, TRANSACTION_COUNT)
        self.source_head = nn.Linear(width, config.num_slots)
        self.target_head = nn.Linear(width, config.num_slots)
        self.relation_head = nn.Linear(width, config.num_relations)
        self.type_head = nn.Linear(width, config.num_types)
        self.value_head = nn.Linear(width, config.num_value_codes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in (
            self.value_embedding,
            self.type_embedding,
            self.slot_embedding,
            self.step_queries,
        ):
            nn.init.normal_(parameter, std=0.02)

    def _state_memory(self, state: TypedTheoryState) -> torch.Tensor:
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
        relation_summary = self.relation_summary_projection(
            torch.cat((incoming, outgoing), dim=-1)
        )
        memory = (
            values
            + types
            + relation_summary
            + self.active_projection(state.active.unsqueeze(-1))
            + self.root_projection(state.root.unsqueeze(-1))
            + self.slot_embedding.unsqueeze(0)
        )
        return self.state_norm(memory)

    def forward(
        self,
        state: TypedTheoryState,
        *,
        command_hidden: torch.Tensor,
        command_attention_mask: torch.Tensor,
        steps: int,
        hard: bool,
    ) -> AddressedSchedule:
        batch = state.value_probabilities.shape[0]
        if (
            not 1 <= steps <= self.config.max_steps
            or command_hidden.ndim != 3
            or command_hidden.shape[0] != batch
            or command_hidden.shape[-1] != self.config.d_model
            or command_attention_mask.shape != command_hidden.shape[:2]
            or command_attention_mask.dtype != torch.bool
        ):
            raise TheoryReactorError("addressed schedule input differs")
        command = self.command_norm(self.command_projection(command_hidden))
        state_memory = self._state_memory(state)
        memory = torch.cat((command, state_memory), dim=1)
        padding = torch.cat(
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
        queries = self.step_queries[:steps].to(command.dtype)
        queries = queries.unsqueeze(0).expand(batch, -1, -1)
        read, _ = self.cross_attention(
            queries,
            memory,
            memory,
            key_padding_mask=padding,
            need_weights=False,
        )
        hidden = self.output_norm(self.schedule_core(queries + read))
        probabilities = {
            "opcode": self.opcode_head(hidden).float().softmax(-1),
            "source": self.source_head(hidden).float().softmax(-1),
            "target": self.target_head(hidden).float().softmax(-1),
            "relation": self.relation_head(hidden).float().softmax(-1),
            "type_index": self.type_head(hidden).float().softmax(-1),
            "value_code": self.value_head(hidden).float().softmax(-1),
        }
        applied = {
            f"applied_{name}": (
                _hard_one_hot(value) if hard else value
            ).to(state.value_probabilities.dtype)
            for name, value in probabilities.items()
        }
        return AddressedSchedule(
            **{
                name: value.to(state.value_probabilities.dtype)
                for name, value in probabilities.items()
            },
            **applied,
        )


class ParallelScheduledReactor(nn.Module):
    """Model-compatible reactor that replays one sticky compiled schedule."""

    def __init__(
        self,
        compiler: ParallelAddressedTransactionCompiler,
        executor: GenericTransactionReactor,
    ) -> None:
        super().__init__()
        if compiler.config != executor.config:
            raise TheoryReactorError("parallel reactor config differs")
        self.config = compiler.config
        self.compiler = compiler
        self.executor = executor
        for parameter in self.executor.parameters():
            parameter.requires_grad_(False)

    def apply(
        self,
        state: TypedTheoryState,
        policy: TransactionPolicy,
        *,
        hard: bool = False,
        validate: bool = True,
    ) -> TypedTheoryState:
        return self.executor.apply(
            state,
            policy,
            hard=hard,
            validate=validate,
        )

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
            raise TheoryReactorError("parallel reactor requires COMMAND bytes")
        schedule = self.compiler(
            state,
            command_hidden=command_hidden,
            command_attention_mask=command_attention_mask.bool(),
            steps=steps,
            hard=hard,
        )
        states: list[TypedTheoryState] = []
        for step in range(steps):
            state = self.executor.apply(
                state,
                schedule.policy(step),
                hard=hard,
                validate=False,
            )
            states.append(state)
        return state, ReactorTrace(
            opcode=schedule.opcode,
            source=schedule.source,
            target=schedule.target,
            relation=schedule.relation,
            type_index=schedule.type_index,
            value_code=schedule.value_code,
            applied_opcode=schedule.applied_opcode,
            applied_source=schedule.applied_source,
            applied_target=schedule.applied_target,
            applied_relation=schedule.applied_relation,
            applied_type_index=schedule.applied_type_index,
            applied_value_code=schedule.applied_value_code,
            active=torch.stack([item.active for item in states], dim=1),
            committed=torch.stack([item.committed for item in states], dim=1),
            halted=torch.stack([item.halted for item in states], dim=1),
        )
