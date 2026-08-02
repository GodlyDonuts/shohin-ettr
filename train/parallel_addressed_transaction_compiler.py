"""Compile a complete addressed ETTR transaction schedule in one pass.

The recurrent ETTR controller is trained with oracle previous states but is
deployed on its own hard choices.  This module removes that exposure boundary:
it reads the initial typed state and COMMAND bytes once, emits one sticky
categorical schedule, and delegates every state mutation to the existing exact
transaction algebra.  No query bytes or answer labels enter the module.
"""

from __future__ import annotations

from collections.abc import Sequence
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
from token_native_syntax_router import (
    TokenNativeDocumentMask,
    TokenNativeOccurrenceEncoder,
    TokenNativeSyntaxGraphEncoder,
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
    program_probabilities: torch.Tensor | None = None

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


def _opcode_programs(
    value: Sequence[Sequence[int]] | None,
    config: TheoryReactorConfig,
) -> tuple[tuple[int, ...], ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or len(value) < 2:
        raise TheoryReactorError("opcode program registry differs")
    programs = []
    for raw in value:
        if isinstance(raw, (str, bytes)):
            raise TheoryReactorError("opcode program registry differs")
        program = tuple(raw)
        if (
            not 1 <= len(program) <= config.max_steps
            or any(type(opcode) is not int for opcode in program)
            or any(not 0 <= opcode < TRANSACTION_COUNT for opcode in program)
        ):
            raise TheoryReactorError("opcode program registry differs")
        programs.append(program)
    if len(programs) != len(set(programs)):
        raise TheoryReactorError("opcode program registry contains duplicates")
    return tuple(programs)


class ParallelAddressedTransactionCompiler(nn.Module):
    """Compile fixed-address transactions without recurrent teacher forcing."""

    def __init__(
        self,
        config: TheoryReactorConfig,
        *,
        width: int = 384,
        layers: int = 3,
        num_heads: int = 8,
        grounded_pointers: bool = False,
        valid_pointer_masks: bool = False,
        token_native_command_mask: bool = False,
        token_native_occurrence_command: bool = False,
        token_native_syntax_graph_command: bool = False,
        token_native_codebook_ids: Sequence[int] | None = None,
        token_native_vocab_size: int | None = None,
        opcode_program_sequences: Sequence[Sequence[int]] | None = None,
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
            or not isinstance(grounded_pointers, bool)
            or not isinstance(valid_pointer_masks, bool)
            or (valid_pointer_masks and not grounded_pointers)
            or not isinstance(token_native_command_mask, bool)
            or not isinstance(token_native_occurrence_command, bool)
            or not isinstance(token_native_syntax_graph_command, bool)
            or (token_native_occurrence_command and not token_native_command_mask)
            or (token_native_syntax_graph_command and not token_native_command_mask)
            or (token_native_occurrence_command and token_native_syntax_graph_command)
            or (
                token_native_command_mask
                and (
                    token_native_codebook_ids is None or token_native_vocab_size is None
                )
            )
            or (
                not token_native_command_mask
                and (
                    token_native_codebook_ids is not None
                    or token_native_vocab_size is not None
                )
            )
        ):
            raise TheoryReactorError("addressed schedule geometry differs")
        self.config = config
        self.width = width
        self.layers = layers
        self.num_heads = num_heads
        self.grounded_pointers = grounded_pointers
        self.valid_pointer_masks = valid_pointer_masks
        self.token_native_command_mask = token_native_command_mask
        self.token_native_occurrence_command = token_native_occurrence_command
        self.token_native_syntax_graph_command = token_native_syntax_graph_command
        self.opcode_program_sequences = _opcode_programs(
            opcode_program_sequences,
            config,
        )

        self.command_projection = nn.Linear(config.d_model, width)
        self.command_norm = nn.LayerNorm(width)
        self.command_document_mask = (
            TokenNativeDocumentMask(
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
            )
            if token_native_syntax_graph_command
            else None
        )
        self.value_embedding = nn.Parameter(torch.empty(config.num_value_codes, width))
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
        if self.opcode_program_sequences is None:
            self.opcode_head = nn.Linear(width, TRANSACTION_COUNT)
            self.program_selector_norm = None
            self.program_selector = None
            self.program_embedding = None
            self.register_buffer("opcode_program_table", None)
            self.register_buffer("opcode_program_step_mask", None)
        else:
            table = torch.zeros(
                len(self.opcode_program_sequences),
                config.max_steps,
                dtype=torch.long,
            )
            step_mask = torch.zeros(
                len(self.opcode_program_sequences),
                config.max_steps,
                dtype=torch.bool,
            )
            for index, program in enumerate(self.opcode_program_sequences):
                table[index, : len(program)] = torch.tensor(program)
                step_mask[index, : len(program)] = True
            self.register_buffer("opcode_program_table", table)
            self.register_buffer("opcode_program_step_mask", step_mask)
            self.program_selector_norm = nn.LayerNorm(width)
            self.program_selector = nn.Linear(
                width,
                len(self.opcode_program_sequences),
            )
            self.program_embedding = nn.Parameter(
                torch.empty(len(self.opcode_program_sequences), width)
            )
        if grounded_pointers:
            self.source_query = nn.Linear(width, width, bias=False)
            self.target_query = nn.Linear(width, width, bias=False)
            self.slot_key = nn.Linear(width, width, bias=False)
        else:
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
        if self.program_embedding is not None:
            nn.init.normal_(self.program_embedding, std=0.02)

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

    @staticmethod
    def _active_prefix(
        initial_active: torch.Tensor,
        opcode: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        """Replay only slot occupancy to derive per-step pointer validity."""

        active = initial_active.float().clamp(0.0, 1.0)
        values = []
        for step in range(opcode.shape[1]):
            values.append(active)
            allocation = opcode[:, step, 0:1] * source[:, step]
            clear = opcode[:, step, 2:3] * source[:, step]
            allocated = allocation * (1.0 - active)
            cleared = clear * active
            active = (active + allocated * (1.0 - active)) * (1.0 - cleared)
        return torch.stack(values, dim=1)

    @staticmethod
    def _mask_pointer_logits(
        source_logits: torch.Tensor,
        target_logits: torch.Tensor,
        opcode: torch.Tensor,
        active_prefix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        allocate = opcode[..., 0:1]
        active_source = opcode[..., 1:6].sum(-1, keepdim=True)
        source_ignored = opcode[..., 6:].sum(-1, keepdim=True)
        source_validity = (
            allocate * (1.0 - active_prefix)
            + active_source * active_prefix
            + source_ignored
        )
        relational = opcode[..., 3:5].sum(-1, keepdim=True)
        target_validity = relational * active_prefix + (1.0 - relational)
        return (
            source_logits + source_validity.clamp_min(1e-4).log(),
            target_logits + target_validity.clamp_min(1e-4).log(),
        )

    def forward(
        self,
        state: TypedTheoryState,
        *,
        command_hidden: torch.Tensor,
        command_attention_mask: torch.Tensor,
        command_tokens: torch.Tensor | None = None,
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
            or (
                self.token_native_command_mask
                and (
                    command_tokens is None
                    or command_tokens.shape != command_hidden.shape[:2]
                    or command_tokens.dtype != torch.long
                )
            )
            or (not self.token_native_command_mask and command_tokens is not None)
        ):
            raise TheoryReactorError("addressed schedule input differs")
        if self.command_document_mask is not None:
            command_attention_mask = self.command_document_mask(
                command_tokens,
                command_attention_mask,
            )
        command = self.command_norm(self.command_projection(command_hidden))
        if self.command_occurrence_encoder is not None:
            command = self.command_occurrence_encoder(
                command,
                command_tokens,
                command_attention_mask,
            )
        if self.command_syntax_graph_encoder is not None:
            command = self.command_syntax_graph_encoder(
                command,
                command_tokens,
                command_attention_mask,
            )
        state_memory = self._state_memory(state)
        selector_probabilities = None
        selected_program = None
        program_context = None
        if self.opcode_program_table is not None:
            assert self.program_selector_norm is not None
            assert self.program_selector is not None
            assert self.program_embedding is not None
            mask = command_attention_mask.unsqueeze(-1).to(command.dtype)
            command_pool = (command * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            state_pool = state_memory.mean(1)
            selector_probabilities = (
                self.program_selector(
                    self.program_selector_norm(command_pool + state_pool)
                )
                .float()
                .softmax(-1)
            )
            selected_program = _hard_one_hot(selector_probabilities)
            straight_through_program = (
                selected_program
                + selector_probabilities
                - selector_probabilities.detach()
            )
            program_context = torch.einsum(
                "bk,kw->bw",
                straight_through_program.to(self.program_embedding.dtype),
                self.program_embedding,
            )
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
        if program_context is not None:
            queries = queries + program_context.unsqueeze(1).to(queries.dtype)
        read, _ = self.cross_attention(
            queries,
            memory,
            memory,
            key_padding_mask=padding,
            need_weights=False,
        )
        hidden = self.output_norm(self.schedule_core(queries + read))
        if self.opcode_program_table is None:
            opcode = self.opcode_head(hidden).float().softmax(-1)
        else:
            assert selector_probabilities is not None
            templates = F.one_hot(
                self.opcode_program_table[:, :steps],
                TRANSACTION_COUNT,
            ).to(selector_probabilities.dtype)
            opcode = torch.einsum(
                "bk,ktc->btc",
                selector_probabilities,
                templates,
            )
        if self.grounded_pointers:
            keys = self.slot_key(state_memory)
            source_logits = torch.einsum(
                "btw,bsw->bts",
                self.source_query(hidden),
                keys,
            ).float()
            target_logits = torch.einsum(
                "btw,bsw->bts",
                self.target_query(hidden),
                keys,
            ).float()
            if self.valid_pointer_masks:
                pointer_opcode = _hard_one_hot(opcode) if hard else opcode
                source = source_logits.softmax(-1)
                prefix_source = _hard_one_hot(source) if hard else source
                active_prefix = self._active_prefix(
                    state.active,
                    pointer_opcode,
                    prefix_source,
                )
                masked_source, _ = self._mask_pointer_logits(
                    source_logits,
                    target_logits,
                    pointer_opcode,
                    active_prefix,
                )
                source = masked_source.softmax(-1)
                prefix_source = _hard_one_hot(source) if hard else source
                active_prefix = self._active_prefix(
                    state.active,
                    pointer_opcode,
                    prefix_source,
                )
                source_logits, target_logits = self._mask_pointer_logits(
                    source_logits,
                    target_logits,
                    pointer_opcode,
                    active_prefix,
                )
            source = source_logits.softmax(-1)
            target = target_logits.softmax(-1)
        else:
            source = self.source_head(hidden).float().softmax(-1)
            target = self.target_head(hidden).float().softmax(-1)
        probabilities = {
            "opcode": opcode,
            "source": source,
            "target": target,
            "relation": self.relation_head(hidden).float().softmax(-1),
            "type_index": self.type_head(hidden).float().softmax(-1),
            "value_code": self.value_head(hidden).float().softmax(-1),
        }
        applied = {
            f"applied_{name}": (_hard_one_hot(value) if hard else value).to(
                state.value_probabilities.dtype
            )
            for name, value in probabilities.items()
        }
        if hard and self.opcode_program_table is not None:
            assert selected_program is not None
            selected_opcode = torch.einsum(
                "bk,ktc->btc",
                selected_program,
                F.one_hot(
                    self.opcode_program_table[:, :steps],
                    TRANSACTION_COUNT,
                ).to(selected_program.dtype),
            )
            applied["applied_opcode"] = selected_opcode.to(
                state.value_probabilities.dtype
            )
        return AddressedSchedule(
            **{
                name: value.to(state.value_probabilities.dtype)
                for name, value in probabilities.items()
            },
            **applied,
            program_probabilities=selector_probabilities,
        )


class MeanParallelAddressedScheduleCompiler(nn.Module):
    """Average complete schedule distributions across independent basins."""

    def __init__(
        self,
        compilers: Sequence[ParallelAddressedTransactionCompiler],
    ) -> None:
        super().__init__()
        if len(compilers) < 2:
            raise TheoryReactorError(
                "parallel schedule ensemble requires multiple compilers"
            )
        config = compilers[0].config
        if any(compiler.config != config for compiler in compilers):
            raise TheoryReactorError("parallel schedule ensemble config differs")
        self.config = config
        self.compilers = nn.ModuleList(compilers)

    def forward(
        self,
        state: TypedTheoryState,
        *,
        command_hidden: torch.Tensor,
        command_attention_mask: torch.Tensor,
        command_tokens: torch.Tensor | None = None,
        steps: int,
        hard: bool,
    ) -> AddressedSchedule:
        schedules = [
            compiler(
                state,
                command_hidden=command_hidden,
                command_attention_mask=command_attention_mask,
                command_tokens=command_tokens,
                steps=steps,
                hard=False,
            )
            for compiler in self.compilers
        ]
        probabilities = {
            name: torch.stack(
                [getattr(schedule, name) for schedule in schedules],
                dim=0,
            ).mean(0)
            for name in (
                "opcode",
                "source",
                "target",
                "relation",
                "type_index",
                "value_code",
            )
        }
        applied = {
            f"applied_{name}": (_hard_one_hot(value) if hard else value).to(
                state.value_probabilities.dtype
            )
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
        config: TheoryReactorConfig,
    ) -> None:
        super().__init__()
        config.validate()
        if compiler.config != config:
            raise TheoryReactorError("parallel reactor config differs")
        self.config = config
        self.compiler = compiler
        self.requires_command_tokens = compiler.token_native_command_mask

    def apply(
        self,
        state: TypedTheoryState,
        policy: TransactionPolicy,
        *,
        hard: bool = False,
        validate: bool = True,
    ) -> TypedTheoryState:
        # GenericTransactionReactor.apply is the audited exact algebra.  It
        # depends only on ``config`` and carries no learned policy weights.
        return GenericTransactionReactor.apply(
            self,
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
        command_tokens: torch.Tensor | None = None,
    ) -> tuple[TypedTheoryState, ReactorTrace]:
        if command_hidden is None or command_attention_mask is None:
            raise TheoryReactorError("parallel reactor requires COMMAND bytes")
        schedule = self.compiler(
            state,
            command_hidden=command_hidden,
            command_attention_mask=command_attention_mask.bool(),
            command_tokens=command_tokens,
            steps=steps,
            hard=hard,
        )
        states: list[TypedTheoryState] = []
        for step in range(steps):
            state = self.apply(
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
