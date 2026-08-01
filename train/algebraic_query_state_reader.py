"""Neural query compiler with an exact differentiable ETTR query algebra.

The compiler sees only the candidate-visible source QUERY prefix.  Its soft
operation and argument distributions parameterize fixed tensor programs over
addressed initial and terminal ETTR state.  The executor contains no learned
truth network and receives no free-form query residual, so an oracle program
has an exact and separately measurable execution ceiling.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from endogenous_typed_theory_reactor import (
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
from typed_query_state_reader import TypedQueryReaderOutput


_HORN_PREDICATE_COUNT = 5
_HORN_OBJECT_COUNT = 6
_HORN_RELATION_BASE = 8
_RUNTIME_SLOT_BASE = 32
_REGISTER_COUNT = 6
_REGISTER_TYPES = (0, 1, 0, 1, 0, 1)
_SYMBOL_COUNT = 4
_RESOURCE_PLACE_COUNT = 4
_SMALL_UINT_BASE = 65
_LOCAL_ID_BASE = 33
_CURSOR_SLOT = 54
_OUTCOME_SLOT = 55
_PROCESS_HALT_CODE = 148


def _straight_through_ge(value: torch.Tensor, threshold: int) -> torch.Tensor:
    """Exact Boolean forward with a smooth bounded threshold adjoint."""

    soft = torch.sigmoid(8.0 * (value - float(threshold) + 0.5))
    hard = value.ge(float(threshold)).to(value.dtype)
    return hard + soft - soft.detach()


class AlgebraicQueryStateReader(nn.Module):
    """Compile query bytes and execute a fixed typed Boolean operator bank."""

    def __init__(
        self,
        config: TheoryReactorConfig,
        *,
        source_vocab_size: int,
        target_vocab_size: int,
        answer_token_ids: tuple[int, int, int, int],
        width: int = 224,
        query_layers: int = 3,
        num_heads: int = 8,
        max_query_tokens: int = 48,
    ) -> None:
        super().__init__()
        config.validate()
        if (
            config.num_slots < 56
            or config.num_relations < 13
            or config.num_value_codes < 150
            or not isinstance(source_vocab_size, int)
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
            or not isinstance(max_query_tokens, int)
            or max_query_tokens < 2
        ):
            raise TheoryReactorError("algebraic query-reader geometry differs")
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
        self.compiler = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.operation_head = nn.Linear(width, len(QUERY_OPERATIONS))
        self.argument_head = nn.Linear(
            width,
            MAX_QUERY_ARGUMENTS * (MAX_QUERY_ARGUMENT_VALUE + 1),
        )
        self.argument_present_head = nn.Linear(width, 2 * MAX_QUERY_ARGUMENTS)
        self.register_buffer(
            "answer_token_ids",
            torch.tensor(answer_token_ids, dtype=torch.long),
            persistent=True,
        )
        nn.init.normal_(self.query_position_embedding, std=0.02)

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
            raise TheoryReactorError("algebraic query token geometry differs")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        visible = attention_mask & (positions[None, :] <= read_index[:, None])
        if not bool(visible.gather(1, read_index[:, None]).all()):
            raise TheoryReactorError("algebraic query read boundary differs")
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

    def _program_probabilities(
        self,
        compiler_hidden: torch.Tensor,
        *,
        teacher: ETTRQuerySpecBatch | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            operation = operation_logits.softmax(dim=-1)
            arguments = argument_logits.softmax(dim=-1)
            present = present_logits.softmax(dim=-1)[..., 1]
        else:
            teacher.validate(batch_size=compiler_hidden.shape[0])
            operation = F.one_hot(
                teacher.operation,
                len(QUERY_OPERATIONS),
            ).to(compiler_hidden.dtype)
            arguments = F.one_hot(
                teacher.arguments,
                MAX_QUERY_ARGUMENT_VALUE + 1,
            ).to(compiler_hidden.dtype)
            present = teacher.argument_mask.to(compiler_hidden.dtype)
        return (
            operation,
            arguments,
            present,
            operation_logits,
            argument_logits,
            present_logits,
        )

    def _operator_truths(
        self,
        arguments: torch.Tensor,
        present: torch.Tensor,
        initial_state: TypedTheoryState,
        terminal_state: TypedTheoryState,
    ) -> torch.Tensor:
        batch = arguments.shape[0]
        dtype = arguments.dtype
        device = arguments.device
        values = terminal_state.value_probabilities.float()
        initial_values = initial_state.value_probabilities.float()
        relations = terminal_state.relations.float()
        if (
            arguments.shape
            != (batch, MAX_QUERY_ARGUMENTS, MAX_QUERY_ARGUMENT_VALUE + 1)
            or present.shape != (batch, MAX_QUERY_ARGUMENTS)
            or values.shape[:2] != (batch, self.config.num_slots)
            or initial_values.shape != values.shape
            or relations.shape
            != (
                batch,
                self.config.num_relations,
                self.config.num_slots,
                self.config.num_slots,
            )
        ):
            raise TheoryReactorError("algebraic state geometry differs")

        # horn_has(predicate, object[, object])
        horn_has = torch.zeros(batch, device=device, dtype=torch.float32)
        for predicate in range(_HORN_PREDICATE_COUNT):
            for source in range(_HORN_OBJECT_COUNT):
                unary = relations[
                    :,
                    _HORN_RELATION_BASE + predicate,
                    _RUNTIME_SLOT_BASE + source,
                    _RUNTIME_SLOT_BASE + source,
                ]
                binary = torch.zeros_like(unary)
                for target in range(_HORN_OBJECT_COUNT):
                    binary = binary + arguments[:, 2, target] * relations[
                        :,
                        _HORN_RELATION_BASE + predicate,
                        _RUNTIME_SLOT_BASE + source,
                        _RUNTIME_SLOT_BASE + target,
                    ]
                selected = (1.0 - present[:, 2]) * unary + present[:, 2] * binary
                horn_has = horn_has + (
                    arguments[:, 0, predicate]
                    * arguments[:, 1, source]
                    * selected
                )

        # horn_count_ge(threshold)
        fact_count = relations[
            :,
            _HORN_RELATION_BASE : _HORN_RELATION_BASE + _HORN_PREDICATE_COUNT,
            _RUNTIME_SLOT_BASE : _RUNTIME_SLOT_BASE + _HORN_OBJECT_COUNT,
            _RUNTIME_SLOT_BASE : _RUNTIME_SLOT_BASE + _HORN_OBJECT_COUNT,
        ].sum(dim=(1, 2, 3))
        horn_count_table = torch.zeros(
            batch,
            MAX_QUERY_ARGUMENT_VALUE + 1,
            device=device,
            dtype=torch.float32,
        )
        for threshold in range(1, MAX_QUERY_ARGUMENT_VALUE + 1):
            horn_count_table[:, threshold] = _straight_through_ge(
                fact_count,
                threshold,
            )
        horn_count = (arguments[:, 0] * horn_count_table).sum(-1)

        # resource_place_ge(place, threshold)
        resource_table = torch.zeros(
            batch,
            MAX_QUERY_ARGUMENT_VALUE + 1,
            MAX_QUERY_ARGUMENT_VALUE + 1,
            device=device,
            dtype=torch.float32,
        )
        for place in range(_RESOURCE_PLACE_COUNT):
            slot_values = values[:, _RUNTIME_SLOT_BASE + place]
            for threshold in range(1, 4):
                resource_table[:, place, threshold] = slot_values[
                    :, _SMALL_UINT_BASE + threshold : _SMALL_UINT_BASE + 16
                ].sum(-1)
        resource_place = torch.einsum(
            "bi,bj,bij->b",
            arguments[:, 0],
            arguments[:, 1],
            resource_table,
        )

        # resource_cursor_ge(threshold) and resource_halt().
        cursor_table = torch.zeros(
            batch,
            MAX_QUERY_ARGUMENT_VALUE + 1,
            device=device,
            dtype=torch.float32,
        )
        cursor_values = values[:, _CURSOR_SLOT]
        for threshold in range(1, 7):
            cursor_table[:, threshold] = cursor_values[
                :, _SMALL_UINT_BASE + threshold : _SMALL_UINT_BASE + 16
            ].sum(-1)
        resource_cursor = (arguments[:, 0] * cursor_table).sum(-1)
        resource_halt = values[:, _OUTCOME_SLOT, _PROCESS_HALT_CODE]

        # slot_is(slot, symbol)
        slot_is_table = torch.zeros(
            batch,
            MAX_QUERY_ARGUMENT_VALUE + 1,
            MAX_QUERY_ARGUMENT_VALUE + 1,
            device=device,
            dtype=torch.float32,
        )
        for slot in range(_REGISTER_COUNT):
            for symbol in range(_SYMBOL_COUNT):
                slot_is_table[:, slot, symbol] = values[
                    :, _RUNTIME_SLOT_BASE + slot, _LOCAL_ID_BASE + symbol
                ]
        slot_is = torch.einsum(
            "bi,bj,bij->b",
            arguments[:, 0],
            arguments[:, 1],
            slot_is_table,
        )

        # type_count_ge(type, symbol, threshold)
        type_count_table = torch.zeros(
            batch,
            MAX_QUERY_ARGUMENT_VALUE + 1,
            MAX_QUERY_ARGUMENT_VALUE + 1,
            MAX_QUERY_ARGUMENT_VALUE + 1,
            device=device,
            dtype=torch.float32,
        )
        for type_index in range(2):
            typed_slots = [
                slot
                for slot, register_type in enumerate(_REGISTER_TYPES)
                if register_type == type_index
            ]
            for symbol in range(_SYMBOL_COUNT):
                count = sum(
                    values[
                        :, _RUNTIME_SLOT_BASE + slot, _LOCAL_ID_BASE + symbol
                    ]
                    for slot in typed_slots
                )
                for threshold in range(1, 4):
                    type_count_table[:, type_index, symbol, threshold] = (
                        _straight_through_ge(count, threshold)
                    )
        type_count = torch.einsum(
            "bi,bj,bk,bijk->b",
            arguments[:, 0],
            arguments[:, 1],
            arguments[:, 2],
            type_count_table,
        )

        # adjacent_is(site, left, right)
        adjacent_table = torch.zeros_like(type_count_table)
        for site in range(_REGISTER_COUNT - 1):
            for left in range(_SYMBOL_COUNT):
                for right in range(_SYMBOL_COUNT):
                    adjacent_table[:, site, left, right] = (
                        values[
                            :, _RUNTIME_SLOT_BASE + site, _LOCAL_ID_BASE + left
                        ]
                        * values[
                            :,
                            _RUNTIME_SLOT_BASE + site + 1,
                            _LOCAL_ID_BASE + right,
                        ]
                    )
        adjacent = torch.einsum(
            "bi,bj,bk,bijk->b",
            arguments[:, 0],
            arguments[:, 1],
            arguments[:, 2],
            adjacent_table,
        )

        # pattern_exists(left, right)
        pattern_table = torch.zeros(
            batch,
            MAX_QUERY_ARGUMENT_VALUE + 1,
            MAX_QUERY_ARGUMENT_VALUE + 1,
            device=device,
            dtype=torch.float32,
        )
        for left in range(_SYMBOL_COUNT):
            for right in range(_SYMBOL_COUNT):
                sites = torch.stack(
                    tuple(
                        values[
                            :, _RUNTIME_SLOT_BASE + site, _LOCAL_ID_BASE + left
                        ]
                        * values[
                            :,
                            _RUNTIME_SLOT_BASE + site + 1,
                            _LOCAL_ID_BASE + right,
                        ]
                        for site in range(_REGISTER_COUNT - 1)
                    ),
                    dim=-1,
                )
                pattern_table[:, left, right] = 1.0 - (1.0 - sites).prod(-1)
        pattern = torch.einsum(
            "bi,bj,bij->b",
            arguments[:, 0],
            arguments[:, 1],
            pattern_table,
        )

        # same_type_slots_equal(left, right)
        equality_table = torch.zeros_like(pattern_table)
        for left in range(_REGISTER_COUNT):
            for right in range(_REGISTER_COUNT):
                equality_table[:, left, right] = (
                    values[
                        :,
                        _RUNTIME_SLOT_BASE + left,
                        _LOCAL_ID_BASE : _LOCAL_ID_BASE + _SYMBOL_COUNT,
                    ]
                    * values[
                        :,
                        _RUNTIME_SLOT_BASE + right,
                        _LOCAL_ID_BASE : _LOCAL_ID_BASE + _SYMBOL_COUNT,
                    ]
                ).sum(-1)
        same_type_equal = torch.einsum(
            "bi,bj,bij->b",
            arguments[:, 0],
            arguments[:, 1],
            equality_table,
        )

        # slot_changed(slot)
        changed_table = torch.zeros(
            batch,
            MAX_QUERY_ARGUMENT_VALUE + 1,
            device=device,
            dtype=torch.float32,
        )
        for slot in range(_REGISTER_COUNT):
            equal = (
                initial_values[
                    :,
                    _RUNTIME_SLOT_BASE + slot,
                    _LOCAL_ID_BASE : _LOCAL_ID_BASE + _SYMBOL_COUNT,
                ]
                * values[
                    :,
                    _RUNTIME_SLOT_BASE + slot,
                    _LOCAL_ID_BASE : _LOCAL_ID_BASE + _SYMBOL_COUNT,
                ]
            ).sum(-1)
            changed_table[:, slot] = 1.0 - equal
        slot_changed = (arguments[:, 0] * changed_table).sum(-1)

        return torch.stack(
            (
                horn_has,
                horn_count,
                resource_place,
                resource_cursor,
                resource_halt,
                slot_is,
                type_count,
                adjacent,
                pattern,
                same_type_equal,
                slot_changed,
            ),
            dim=-1,
        ).to(dtype)

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
        compiler_hidden = self.compiler(query_hidden)
        (
            operation,
            arguments,
            present,
            operation_logits,
            argument_logits,
            present_logits,
        ) = self._program_probabilities(compiler_hidden, teacher=teacher_program)
        operator_truths = self._operator_truths(
            arguments,
            present,
            initial_state,
            terminal_state,
        )
        truth_probability = (operation * operator_truths).sum(-1)
        truth_probability = truth_probability * (1.0 - 2.0e-6) + 1.0e-6
        truth = torch.stack(
            (
                torch.log1p(-truth_probability),
                truth_probability.log(),
            ),
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
            device=query_tokens.device,
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


__all__ = ["AlgebraicQueryStateReader"]
