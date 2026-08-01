"""Learned query compiler plus cardinality-preserving typed-state executor.

Unlike the residual-injection reader, this interface gives query parsing and
state execution separate, measurable responsibilities.  The compiler sees
only source query tokens up to ``R=``.  The executor sees only addressed
initial/terminal typed state and the compiler's predicted soft program.  No
answer or assessor label enters autonomous inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from endogenous_typed_theory_reactor import (
    DISPOSITION_COUNT,
    TheoryReactorConfig,
    TheoryReactorError,
    TypedTheoryState,
    _disposition_probabilities,
)
from ettr_query_supervision import (
    ETTRQuerySpecBatch,
    MAX_QUERY_ARGUMENTS,
    MAX_QUERY_ARGUMENT_VALUE,
    QUERY_OPERATIONS,
)


@dataclass(frozen=True, slots=True)
class TypedQueryReaderOutput:
    """Autonomous answer logits and auditable compiler predictions."""

    vocab_logits: torch.Tensor
    class_logits: torch.Tensor
    operation_logits: torch.Tensor
    argument_logits: torch.Tensor
    argument_present_logits: torch.Tensor


class TypedQueryStateReader(nn.Module):
    """Compile token-native queries and execute them over typed graph state."""

    def __init__(
        self,
        config: TheoryReactorConfig,
        *,
        source_vocab_size: int,
        target_vocab_size: int,
        answer_token_ids: tuple[int, int, int, int],
        width: int = 224,
        query_layers: int = 3,
        state_layers: int = 2,
        num_heads: int = 8,
        max_query_tokens: int = 48,
    ) -> None:
        super().__init__()
        config.validate()
        if (
            not isinstance(source_vocab_size, int)
            or source_vocab_size < 2
            or not isinstance(target_vocab_size, int)
            or target_vocab_size < 5
            or not isinstance(answer_token_ids, tuple)
            or len(answer_token_ids) != 4
            or len(set(answer_token_ids)) != 4
            or any(
                not isinstance(value, int) or not 0 <= value < target_vocab_size
                for value in answer_token_ids
            )
            or not isinstance(width, int)
            or width < 64
            or width % num_heads
            or not 1 <= query_layers <= 12
            or not 1 <= state_layers <= 12
            or not isinstance(max_query_tokens, int)
            or max_query_tokens < 2
        ):
            raise TheoryReactorError("typed query-reader geometry differs")
        self.config = config
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.width = width
        self.max_query_tokens = max_query_tokens

        self.query_token_embedding = nn.Embedding(source_vocab_size, width)
        self.query_position_embedding = nn.Parameter(
            torch.empty(max_query_tokens, width)
        )
        self.query_norm = nn.LayerNorm(width)
        query_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=num_heads,
            dim_feedforward=4 * width,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.query_encoder = nn.TransformerEncoder(
            query_layer,
            num_layers=query_layers,
            enable_nested_tensor=False,
        )

        self.value_embedding = nn.Parameter(
            torch.empty(config.num_value_codes, width)
        )
        self.type_embedding = nn.Parameter(torch.empty(config.num_types, width))
        self.relation_embedding = nn.Parameter(
            torch.empty(config.num_relations, width)
        )
        self.slot_embedding = nn.Parameter(torch.empty(config.num_slots, width))
        self.phase_embedding = nn.Parameter(torch.empty(2, width))
        self.active_projection = nn.Linear(1, width, bias=False)
        self.root_projection = nn.Linear(1, width, bias=False)
        self.status_projection = nn.Linear(DISPOSITION_COUNT, width, bias=False)
        self.state_norm = nn.LayerNorm(width)
        state_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=num_heads,
            dim_feedforward=4 * width,
            batch_first=True,
            norm_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.state_encoder = nn.TransformerEncoder(
            state_layer,
            num_layers=state_layers,
            enable_nested_tensor=False,
        )

        self.compiler_fusion = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.operation_head = nn.Linear(width, len(QUERY_OPERATIONS))
        self.argument_head = nn.Linear(
            width,
            MAX_QUERY_ARGUMENTS * (MAX_QUERY_ARGUMENT_VALUE + 1),
        )
        self.argument_present_head = nn.Linear(width, 2 * MAX_QUERY_ARGUMENTS)
        self.operation_embedding = nn.Parameter(
            torch.empty(len(QUERY_OPERATIONS), width)
        )
        self.argument_embedding = nn.Parameter(
            torch.empty(MAX_QUERY_ARGUMENTS, MAX_QUERY_ARGUMENT_VALUE + 1, width)
        )
        self.query_latent_projection = nn.Linear(width, width, bias=False)

        self.program_projection = nn.Linear(width, width)
        self.slot_gate = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.slot_select = nn.Linear(width, 1)
        self.pair_projection = nn.Sequential(
            nn.LayerNorm(2 * width),
            nn.Linear(2 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.pair_gate = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )
        self.truth_motor = nn.Sequential(
            nn.LayerNorm(8 * width),
            nn.Linear(8 * width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, 2),
        )
        self.register_buffer(
            "answer_token_ids",
            torch.tensor(answer_token_ids, dtype=torch.long),
            persistent=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_position_embedding, std=0.02)
        for parameter in (
            self.value_embedding,
            self.type_embedding,
            self.relation_embedding,
            self.slot_embedding,
            self.phase_embedding,
            self.operation_embedding,
            self.argument_embedding,
        ):
            nn.init.normal_(parameter, std=0.02)

    def _encode_query(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        read_index: torch.Tensor,
    ) -> torch.Tensor:
        if (
            tokens.ndim != 2
            or tokens.shape != attention_mask.shape
            or tokens.shape[1] > self.max_query_tokens
            or attention_mask.dtype != torch.bool
            or read_index.shape != (tokens.shape[0],)
            or read_index.dtype != torch.long
            or tokens.dtype != torch.long
            or not bool(((tokens >= 0) & (tokens < self.source_vocab_size)).all())
        ):
            raise TheoryReactorError("typed query token geometry differs")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        visible = attention_mask & (positions[None, :] <= read_index[:, None])
        if not bool(visible.gather(1, read_index[:, None]).all()):
            raise TheoryReactorError("typed query read boundary differs")
        safe_tokens = torch.where(visible, tokens, torch.zeros_like(tokens))
        hidden = self.query_token_embedding(safe_tokens)
        hidden = hidden + self.query_position_embedding[: tokens.shape[1]]
        hidden = self.query_encoder(
            self.query_norm(hidden),
            src_key_padding_mask=~visible,
        )
        return hidden.gather(
            1,
            read_index[:, None, None].expand(-1, 1, hidden.shape[-1]),
        ).squeeze(1)

    def _state_slots(
        self,
        state: TypedTheoryState,
        *,
        phase: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if phase not in (0, 1):
            raise TheoryReactorError("typed state phase differs")
        active = state.active.float()
        values = torch.einsum(
            "bsc,cw->bsw",
            state.value_probabilities.float(),
            self.value_embedding,
        )
        types = torch.einsum(
            "bst,tw->bsw",
            state.type_probabilities.float(),
            self.type_embedding,
        )
        relations = state.relations.float()
        outgoing = relations.sum(dim=-1).transpose(1, 2)
        incoming = relations.sum(dim=-2).transpose(1, 2)
        relation_context = torch.einsum(
            "bsr,rw->bsw",
            outgoing + incoming,
            self.relation_embedding,
        )
        status = self.status_projection(
            _disposition_probabilities(state).float()
        )[:, None, :]
        slots = (
            values
            + types
            + relation_context
            + self.slot_embedding[None, :, :]
            + self.phase_embedding[phase][None, None, :]
            + self.active_projection(active.unsqueeze(-1))
            + self.root_projection(state.root.float().unsqueeze(-1))
            + status
        )
        valid = active.ge(0.5)
        empty = ~valid.any(dim=1)
        valid = valid.clone()
        valid[empty, 0] = True
        return self.state_norm(slots), valid

    def _encode_state(
        self,
        initial_state: TypedTheoryState,
        terminal_state: TypedTheoryState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        initial, initial_valid = self._state_slots(initial_state, phase=0)
        terminal, terminal_valid = self._state_slots(terminal_state, phase=1)
        slots = torch.cat((initial, terminal), dim=1)
        valid = torch.cat((initial_valid, terminal_valid), dim=1)
        slots = self.state_encoder(slots, src_key_padding_mask=~valid)
        weights = valid.to(slots.dtype).unsqueeze(-1)
        count = weights.sum(dim=1).clamp_min(1.0)
        mean_pool = (slots * weights).sum(dim=1) / count
        sum_pool = (slots * weights).sum(dim=1) / count.sqrt()
        return slots, valid, torch.cat((mean_pool, sum_pool), dim=-1)

    def _program(
        self,
        compiler_hidden: torch.Tensor,
        query_hidden: torch.Tensor,
        *,
        teacher: ETTRQuerySpecBatch | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        operation_logits = self.operation_head(compiler_hidden)
        argument_logits = self.argument_head(compiler_hidden).reshape(
            -1,
            MAX_QUERY_ARGUMENTS,
            MAX_QUERY_ARGUMENT_VALUE + 1,
        )
        present_logits = self.argument_present_head(compiler_hidden).reshape(
            -1,
            MAX_QUERY_ARGUMENTS,
            2,
        )
        if teacher is None:
            operation_probabilities = operation_logits.softmax(dim=-1)
            argument_probabilities = argument_logits.softmax(dim=-1)
            present = present_logits.softmax(dim=-1)[..., 1]
        else:
            teacher.validate(batch_size=compiler_hidden.shape[0])
            operation_probabilities = F.one_hot(
                teacher.operation,
                len(QUERY_OPERATIONS),
            ).to(compiler_hidden.dtype)
            argument_probabilities = F.one_hot(
                teacher.arguments,
                MAX_QUERY_ARGUMENT_VALUE + 1,
            ).to(compiler_hidden.dtype)
            present = teacher.argument_mask.to(compiler_hidden.dtype)
        program = torch.einsum(
            "bo,ow->bw",
            operation_probabilities,
            self.operation_embedding,
        )
        argument_vectors = torch.einsum(
            "bpa,paw->bpw",
            argument_probabilities,
            self.argument_embedding,
        )
        program = (
            program
            + (argument_vectors * present.unsqueeze(-1)).sum(dim=1)
            + self.query_latent_projection(query_hidden)
        )
        return program, operation_logits, argument_logits, present_logits

    def _execute(
        self,
        program: torch.Tensor,
        state_slots: torch.Tensor,
        state_valid: torch.Tensor,
        state_global: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.program_projection(program)[:, None, :]
        conditioned = state_slots + condition
        valid = state_valid.to(state_slots.dtype).unsqueeze(-1)
        gate = torch.sigmoid(self.slot_gate(conditioned)) * valid
        slot_count = valid.sum(dim=1).clamp_min(1.0)
        gated_sum = (state_slots * gate).sum(dim=1) / slot_count.sqrt()
        masked = conditioned.masked_fill(~state_valid.unsqueeze(-1), -1.0e4)
        gated_max = masked.max(dim=1).values
        select_logits = self.slot_select(conditioned).squeeze(-1)
        select_logits = select_logits.masked_fill(~state_valid, -1.0e4)
        selected = torch.einsum(
            "bs,bsw->bw",
            select_logits.softmax(dim=-1),
            state_slots,
        )

        batch = state_slots.shape[0]
        slots = state_slots.reshape(batch, 2, self.config.num_slots, self.width)
        validity = state_valid.reshape(batch, 2, self.config.num_slots)
        pairs = self.pair_projection(
            torch.cat((slots[:, :, :-1], slots[:, :, 1:]), dim=-1)
        )
        pair_valid = validity[:, :, :-1] & validity[:, :, 1:]
        pair_conditioned = pairs + condition[:, None, :, :]
        pair_gate = torch.sigmoid(self.pair_gate(pair_conditioned))
        pair_weight = pair_valid.to(pairs.dtype).unsqueeze(-1)
        pair_count = pair_weight.sum(dim=(1, 2)).clamp_min(1.0)
        pair_sum = (pairs * pair_gate * pair_weight).sum(dim=(1, 2)) / pair_count.sqrt()
        pair_max = pair_conditioned.masked_fill(
            ~pair_valid.unsqueeze(-1),
            -1.0e4,
        ).flatten(1, 2).max(dim=1).values

        return self.truth_motor(
            torch.cat(
                (
                    program,
                    state_global,
                    gated_sum,
                    gated_max,
                    selected,
                    pair_sum,
                    pair_max,
                ),
                dim=-1,
            )
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        query_attention_mask: torch.Tensor,
        query_read_index: torch.Tensor,
        initial_state: TypedTheoryState,
        terminal_state: TypedTheoryState,
        *,
        teacher_program: ETTRQuerySpecBatch | None = None,
    ) -> TypedQueryReaderOutput:
        query_hidden = self._encode_query(
            query_tokens,
            query_attention_mask,
            query_read_index,
        )
        state_slots, state_valid, state_global = self._encode_state(
            initial_state,
            terminal_state,
        )
        mean_state, sum_state = state_global.chunk(2, dim=-1)
        compiler_hidden = self.compiler_fusion(
            torch.cat((query_hidden, mean_state, sum_state), dim=-1)
        )
        program, operation_logits, argument_logits, present_logits = self._program(
            compiler_hidden,
            query_hidden,
            teacher=teacher_program,
        )
        truth = F.log_softmax(
            self._execute(program, state_slots, state_valid, state_global),
            dim=-1,
        )
        disposition = _disposition_probabilities(terminal_state).to(truth.dtype)
        tiny = torch.finfo(truth.dtype).tiny
        class_logits = torch.cat(
            (
                truth + disposition[:, 1:2].clamp_min(tiny).log(),
                disposition[:, 2:3].clamp_min(tiny).log(),
                disposition[:, 3:4].clamp_min(tiny).log(),
            ),
            dim=-1,
        )
        floor = -min(1.0e4, math.sqrt(torch.finfo(class_logits.dtype).max))
        vocab_logits = torch.full(
            (query_tokens.shape[0], self.target_vocab_size),
            floor,
            dtype=class_logits.dtype,
            device=class_logits.device,
        )
        token_ids = self.answer_token_ids.to(query_tokens.device)
        vocab_logits.scatter_(
            -1,
            token_ids[None, :].expand(query_tokens.shape[0], -1),
            class_logits,
        )
        return TypedQueryReaderOutput(
            vocab_logits=vocab_logits,
            class_logits=class_logits,
            operation_logits=operation_logits,
            argument_logits=argument_logits,
            argument_present_logits=present_logits,
        )


__all__ = ["TypedQueryReaderOutput", "TypedQueryStateReader"]
