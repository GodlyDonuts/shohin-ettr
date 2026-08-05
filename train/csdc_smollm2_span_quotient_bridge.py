#!/usr/bin/env python3
"""Compile whole lexical mentions into the frozen CSDC reasoner."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from csdc_role_gated_copy_bridge import (
    OTHER_ROLE,
    OUTCOME_ROLE,
    ROLE_COUNT,
    START_ROLE,
    WORD_ROLE,
)
from csdc_semantic_bridge import DecodedChallenges, render_semantic_source
from csdc_smollm2_lexical_bridge import (
    EXPECTED_REASONER_SHA256,
    EXPECTED_WARM_ADAPTER_SHA256,
    SHIFT_GENERATOR_ALIASES,
    SHIFT_STATE_ALIASES,
    TRAIN_GENERATOR_ALIASES,
    TRAIN_STATE_ALIASES,
    AnnotatedRecord,
    LexicalBridgeConfig,
    _alias_assignment,
    _challenge_record,
    _observation_record,
    gather_lexical_targets,
    load_warm_adapter,
    sha256_file,
)
from evaluate_counterexample_selected_closure import (
    binary_completion_candidates,
    select_with_challenges,
)
from frozen_pointer_backbone import load_frozen_pointer_backbone
from learned_pspa_language_reasoning import (
    LanguageConfig,
    LearnedPSPAGate,
    execute_word,
)
from prompt_selected_presented_algebra import PresentedAlgebraConfig
from pspa_presented_reasoning import (
    FAMILIES,
    PresentedBatch,
    PresentedReasoningError,
    batch_sha256,
    generate_batch,
)


SCHEMA = "shohin-csdc-smollm2-span-quotient-bridge-v1"
MAX_SPAN_WIDTH = 4
ERROR_NONE = 0
ERROR_EXACT = 1
ERROR_PARTIAL = 2
ERROR_SUPERSET = 3
ERROR_OVERLAP = 4
EXPECTED_FIRST_SUBTOKEN_REPORT_SHA256 = (
    "b3ae0526e9e28ef21e93f2b32bacd7845f40cdb663d8d4b48d74ed9b7cfc05c5"
)


@dataclass(frozen=True, slots=True)
class SpanLexicalSource:
    ids: torch.Tensor
    valid_mask: torch.Tensor
    token_record: torch.Tensor
    record_mask: torch.Tensor
    challenge_record: torch.Tensor
    challenge_start: torch.Tensor
    challenge_outcome: torch.Tensor
    challenge_length: torch.Tensor
    challenge_word: torch.Tensor
    candidate_batch: torch.Tensor
    candidate_record: torch.Tensor
    candidate_group_start: torch.Tensor
    candidate_group_end: torch.Tensor
    candidate_start: torch.Tensor
    candidate_end: torch.Tensor
    candidate_class: torch.Tensor
    candidate_target_role: torch.Tensor
    candidate_target_value: torch.Tensor
    candidate_error_kind: torch.Tensor
    labeled_mentions: torch.Tensor
    represented_mentions: torch.Tensor

    def to(self, device: torch.device | str) -> SpanLexicalSource:
        return SpanLexicalSource(
            **{
                name: getattr(self, name).to(device)
                for name in self.__dataclass_fields__
            }
        )


@dataclass(frozen=True, slots=True)
class SpanQuotientLogits:
    kind: torch.Tensor
    role: torch.Tensor


@dataclass(frozen=True, slots=True)
class SpanDecodeAudit:
    selected_candidate: torch.Tensor
    selected_mask: torch.Tensor
    valid: torch.Tensor
    missing_start: torch.Tensor
    duplicate_start: torch.Tensor
    missing_outcome: torch.Tensor
    duplicate_outcome: torch.Tensor
    missing_word: torch.Tensor
    excess_word: torch.Tensor
    nonexact_identity: torch.Tensor


def _positions_for_span(
    offsets: list[tuple[int, int]], start: int, end: int
) -> tuple[int, ...]:
    return tuple(
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > start and token_start < end
    )


def _candidate_text(
    text: str,
    offsets: list[tuple[int, int]],
    start: int,
    end: int,
) -> str:
    char_start = int(offsets[start][0])
    char_end = int(offsets[end][1])
    while char_start < char_end and text[char_start].isspace():
        char_start += 1
    while char_end > char_start and text[char_end - 1].isspace():
        char_end -= 1
    return text[char_start:char_end]


def _overlap_kind(
    start: int,
    end: int,
    labeled: list[tuple[int, int, int, int]],
) -> int:
    for gold_start, gold_end, _, _ in labeled:
        if start == gold_start and end == gold_end:
            return ERROR_EXACT
    result = ERROR_NONE
    for gold_start, gold_end, _, _ in labeled:
        if end < gold_start or start > gold_end:
            continue
        if gold_start <= start and end <= gold_end:
            result = max(result, ERROR_PARTIAL)
        elif start <= gold_start and gold_end <= end:
            result = max(result, ERROR_SUPERSET)
        else:
            result = max(result, ERROR_OVERLAP)
    return result


def _records_for_row(
    batch: PresentedBatch,
    row: int,
    algebra: PresentedAlgebraConfig,
    *,
    rng: random.Random,
    templates: tuple[int, ...],
    shifted_aliases: bool,
) -> list[AnnotatedRecord]:
    state_pool = SHIFT_STATE_ALIASES if shifted_aliases else TRAIN_STATE_ALIASES
    generator_pool = (
        SHIFT_GENERATOR_ALIASES if shifted_aliases else TRAIN_GENERATOR_ALIASES
    )
    states = _alias_assignment(algebra.carrier_size, state_pool, rng)
    generators = _alias_assignment(
        algebra.maximum_generators, generator_pool, rng
    )
    records: list[AnnotatedRecord] = []
    for index in range(algebra.maximum_observations):
        if bool(batch.observation_mask[row, index]):
            records.append(
                _observation_record(
                    rng.randrange(2),
                    int(batch.observation_generator[row, index]),
                    int(batch.observation_input[row, index]),
                    int(batch.observation_output[row, index]),
                    states,
                    generators,
                )
            )
    for index in range(algebra.maximum_challenges):
        if not bool(batch.challenge_mask[row, index]):
            continue
        word = [
            int(value)
            for value, keep in zip(
                batch.challenge_word[row, index],
                batch.challenge_word_mask[row, index],
                strict=True,
            )
            if bool(keep)
        ]
        records.append(
            _challenge_record(
                templates[rng.randrange(len(templates))],
                int(batch.challenge_start[row, index]),
                word,
                int(batch.challenge_outcome[row, index]),
                states,
                generators,
            )
        )
    rng.shuffle(records)
    return records


def render_span_lexical_source(
    batch: PresentedBatch,
    algebra: PresentedAlgebraConfig,
    tokenizer: Tokenizer,
    *,
    seed: int,
    templates: tuple[int, ...],
    shifted_aliases: bool,
    seq_len: int,
    max_span_width: int = MAX_SPAN_WIDTH,
) -> SpanLexicalSource:
    """Render natural records and enumerate label-agnostic whole-span candidates."""

    if batch.family.device.type != "cpu":
        raise PresentedReasoningError("span rendering requires a CPU batch")
    if max_span_width != MAX_SPAN_WIDTH:
        raise PresentedReasoningError("span width differs from the frozen protocol")
    if not templates or any(template not in range(4) for template in templates):
        raise PresentedReasoningError("invalid lexical templates")
    maximum_records = algebra.maximum_observations + algebra.maximum_challenges
    rows: list[dict[str, Any]] = []
    class_index: dict[tuple[int, bytes], int] = {}
    candidate_rows: list[int] = []
    candidate_records: list[int] = []
    candidate_starts: list[int] = []
    candidate_ends: list[int] = []
    candidate_classes: list[int] = []
    candidate_roles: list[int] = []
    candidate_values: list[int] = []
    candidate_errors: list[int] = []
    maximum_tokens = 0

    for row in range(batch.family.shape[0]):
        rng = random.Random(seed * 1_000_003 + row * 8191)
        records = _records_for_row(
            batch,
            row,
            algebra,
            rng=rng,
            templates=templates,
            shifted_aliases=shifted_aliases,
        )
        if len(records) > maximum_records:
            raise PresentedReasoningError("lexical record count exceeds geometry")
        ids: list[int] = []
        token_record: list[int] = []
        record_targets: list[AnnotatedRecord] = []
        candidate_groups: list[tuple[int, int]] = []
        labeled_count = 0
        represented_count = 0
        for record_index, record in enumerate(records):
            group_start = len(candidate_rows)
            encoding = tokenizer.encode(record.text + "\n", add_special_tokens=False)
            if not encoding.ids:
                raise PresentedReasoningError("tokenizer emitted an empty record")
            token_offset = len(ids)
            labeled: list[tuple[int, int, int, int]] = []
            exact_targets: dict[tuple[int, int], tuple[int, int]] = {}
            for span in record.spans:
                positions = _positions_for_span(
                    encoding.offsets, span.start, span.end
                )
                labeled_count += 1
                if not positions or positions != tuple(
                    range(positions[0], positions[-1] + 1)
                ):
                    raise PresentedReasoningError("mention is not a contiguous token span")
                if len(positions) > max_span_width:
                    raise PresentedReasoningError(
                        "labeled alias exceeds frozen span width"
                    )
                key = (positions[0], positions[-1])
                if key in exact_targets:
                    raise PresentedReasoningError("labeled mentions collide")
                exact_targets[key] = (span.role, span.value)
                labeled.append((positions[0], positions[-1], span.role, span.value))
            represented: set[tuple[int, int]] = set()
            for start in range(len(encoding.ids)):
                for width in range(1, max_span_width + 1):
                    end = start + width - 1
                    if end >= len(encoding.ids):
                        break
                    text = _candidate_text(
                        record.text + "\n", encoding.offsets, start, end
                    )
                    if not text:
                        continue
                    target_role, target_value = exact_targets.get(
                        (start, end), (OTHER_ROLE, -1)
                    )
                    if target_role != OTHER_ROLE:
                        represented.add((start, end))
                    surface = text.encode("utf-8")
                    class_key = (row, surface)
                    if class_key not in class_index:
                        class_index[class_key] = len(class_index)
                    candidate_rows.append(row)
                    candidate_records.append(record_index)
                    candidate_starts.append(token_offset + start)
                    candidate_ends.append(token_offset + end)
                    candidate_classes.append(class_index[class_key])
                    candidate_roles.append(target_role)
                    candidate_values.append(target_value)
                    candidate_errors.append(
                        _overlap_kind(start, end, labeled)
                    )
            if represented != set(exact_targets):
                raise PresentedReasoningError("span enumeration missed a mention")
            represented_count += len(represented)
            candidate_groups.append((group_start, len(candidate_rows)))
            ids.extend(encoding.ids)
            token_record.extend([record_index] * len(encoding.ids))
            record_targets.append(record)
        if len(ids) > seq_len:
            raise PresentedReasoningError(
                f"lexical source length {len(ids)} exceeds sequence length {seq_len}"
            )
        maximum_tokens = max(maximum_tokens, len(ids))
        rows.append(
            {
                "ids": ids,
                "token_record": token_record,
                "records": record_targets,
                "candidate_groups": candidate_groups,
                "labeled": labeled_count,
                "represented": represented_count,
            }
        )

    if not candidate_rows:
        raise PresentedReasoningError("span renderer produced no candidates")
    geometry = (len(rows), maximum_tokens)
    ids_tensor = torch.zeros(geometry, dtype=torch.long)
    valid_mask = torch.zeros(geometry, dtype=torch.bool)
    token_record_tensor = torch.full(geometry, -1, dtype=torch.long)
    record_mask = torch.zeros(len(rows), maximum_records, dtype=torch.bool)
    challenge_record = torch.zeros_like(record_mask)
    challenge_start = torch.zeros_like(record_mask, dtype=torch.long)
    challenge_outcome = torch.zeros_like(record_mask, dtype=torch.long)
    challenge_length = torch.ones_like(record_mask, dtype=torch.long)
    challenge_word = torch.zeros(
        len(rows), maximum_records, algebra.maximum_word_length, dtype=torch.long
    )
    candidate_group_start = torch.full(
        (len(rows), maximum_records), -1, dtype=torch.long
    )
    candidate_group_end = torch.full_like(candidate_group_start, -1)
    labeled_mentions = torch.zeros(len(rows), dtype=torch.long)
    represented_mentions = torch.zeros(len(rows), dtype=torch.long)
    for row_index, row in enumerate(rows):
        length = len(row["ids"])
        ids_tensor[row_index, :length] = torch.tensor(row["ids"])
        valid_mask[row_index, :length] = True
        token_record_tensor[row_index, :length] = torch.tensor(row["token_record"])
        labeled_mentions[row_index] = row["labeled"]
        represented_mentions[row_index] = row["represented"]
        for record_index, record in enumerate(row["records"]):
            group_start, group_end = row["candidate_groups"][record_index]
            candidate_group_start[row_index, record_index] = group_start
            candidate_group_end[row_index, record_index] = group_end
            record_mask[row_index, record_index] = True
            challenge_record[row_index, record_index] = record.challenge
            if record.challenge:
                challenge_start[row_index, record_index] = record.start
                challenge_outcome[row_index, record_index] = record.outcome
                challenge_length[row_index, record_index] = len(record.word)
                challenge_word[row_index, record_index, : len(record.word)] = (
                    torch.tensor(record.word)
                )
    return SpanLexicalSource(
        ids=ids_tensor,
        valid_mask=valid_mask,
        token_record=token_record_tensor,
        record_mask=record_mask,
        challenge_record=challenge_record,
        challenge_start=challenge_start,
        challenge_outcome=challenge_outcome,
        challenge_length=challenge_length,
        challenge_word=challenge_word,
        candidate_batch=torch.tensor(candidate_rows, dtype=torch.long),
        candidate_record=torch.tensor(candidate_records, dtype=torch.long),
        candidate_group_start=candidate_group_start,
        candidate_group_end=candidate_group_end,
        candidate_start=torch.tensor(candidate_starts, dtype=torch.long),
        candidate_end=torch.tensor(candidate_ends, dtype=torch.long),
        candidate_class=torch.tensor(candidate_classes, dtype=torch.long),
        candidate_target_role=torch.tensor(candidate_roles, dtype=torch.long),
        candidate_target_value=torch.tensor(candidate_values, dtype=torch.long),
        candidate_error_kind=torch.tensor(candidate_errors, dtype=torch.long),
        labeled_mentions=labeled_mentions,
        represented_mentions=represented_mentions,
    )


class SpanQuotientChallengeParser(nn.Module):
    """Frozen Smol residuals plus exact-surface whole-mention quotienting."""

    def __init__(self, model: nn.Module, config: LexicalBridgeConfig):
        super().__init__()
        if model.cfg.n_loop != 1 or not 0 <= config.layer < len(model.blocks):
            raise ValueError("invalid frozen lexical backbone")
        if config.width % config.heads or config.encoder_layers <= 0:
            raise ValueError("invalid span adapter geometry")
        self.model = model
        self.config = config
        self.model.requires_grad_(False)
        self.memory_norm = nn.LayerNorm(model.cfg.d_model)
        self.memory_projection = nn.Linear(
            model.cfg.d_model, config.width, bias=False
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=config.heads,
            dim_feedforward=config.ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory_encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.encoder_layers,
            enable_nested_tensor=False,
        )
        self.span_projection = nn.Sequential(
            nn.LayerNorm(3 * config.width),
            nn.Linear(3 * config.width, config.width),
            nn.GELU(),
        )
        self.class_projection = nn.Sequential(
            nn.LayerNorm(config.width),
            nn.Linear(config.width, config.width),
            nn.GELU(),
        )
        self.kind_head = nn.Linear(config.width, 2)
        self.role_head = nn.Sequential(
            nn.LayerNorm(2 * config.width),
            nn.Linear(2 * config.width, config.width),
            nn.GELU(),
            nn.Linear(config.width, ROLE_COUNT),
        )

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if not name.startswith("model."):
                yield parameter

    def encode_memory(self, source: SpanLexicalSource) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            hidden = self.model.tok(source.ids)
            cos = self.model.cos[: source.ids.shape[1]].to(hidden.device)
            sin = self.model.sin[: source.ids.shape[1]].to(hidden.device)
            for block in self.model.blocks[: self.config.layer + 1]:
                hidden, _ = block(hidden, cos, sin)
        memory = self.memory_projection(self.memory_norm(hidden.detach()))
        return self.memory_encoder(
            memory, src_key_padding_mask=~source.valid_mask
        )

    def score_memory(
        self,
        memory: torch.Tensor,
        source: SpanLexicalSource,
        *,
        class_messages: bool = True,
    ) -> SpanQuotientLogits:
        records = source.record_mask.shape[1]
        record_index = source.token_record.clamp_min(0)
        membership = F.one_hot(record_index, records).to(memory.dtype)
        membership = membership * source.valid_mask[..., None]
        summaries = torch.einsum("blr,blh->brh", membership, memory)
        summaries = summaries / membership.sum(1).clamp_min(1).unsqueeze(-1)

        batch = source.candidate_batch
        start = source.candidate_start
        end = source.candidate_end
        prefix = torch.cat(
            (
                torch.zeros(
                    memory.shape[0],
                    1,
                    memory.shape[2],
                    device=memory.device,
                    dtype=memory.dtype,
                ),
                memory.cumsum(1),
            ),
            dim=1,
        )
        mean = (prefix[batch, end + 1] - prefix[batch, start]) / (
            (end - start + 1).to(memory.dtype).unsqueeze(-1)
        )
        span = self.span_projection(
            torch.cat((memory[batch, start], memory[batch, end], mean), dim=-1)
        )
        classes = source.candidate_class
        class_count = int(classes.max().item()) + 1
        class_sum = torch.zeros(
            class_count, self.config.width, device=span.device, dtype=span.dtype
        ).index_add_(0, classes, span)
        class_size = torch.zeros(
            class_count, device=span.device, dtype=span.dtype
        ).index_add_(
            0,
            classes,
            torch.ones_like(classes, dtype=span.dtype),
        )
        class_mean = class_sum / class_size.clamp_min(1).unsqueeze(-1)
        class_context = self.class_projection(class_mean)[classes]
        if not class_messages:
            class_context = torch.zeros_like(class_context)
        return SpanQuotientLogits(
            kind=self.kind_head(summaries).float(),
            role=self.role_head(torch.cat((span, class_context), dim=-1)).float(),
        )

    def forward(
        self,
        source: SpanLexicalSource,
        *,
        class_messages: bool = True,
    ) -> SpanQuotientLogits:
        return self.score_memory(
            self.encode_memory(source), source, class_messages=class_messages
        )


def _decode_record_candidates(
    role_logits: torch.Tensor,
    source: SpanLexicalSource,
    indices: torch.Tensor,
    maximum_word_length: int,
) -> tuple[list[int], tuple[int, int, int]]:
    if indices.numel() == 0:
        return [], (0, 0, 0)
    local = role_logits[indices]
    predicted = local.argmax(-1)
    margin = local.gather(-1, predicted[:, None]).squeeze(-1) - local[:, OTHER_ROLE]
    active = predicted.ne(OTHER_ROLE) & margin.gt(0)
    order = indices[active][margin[active].argsort(descending=True)]
    occupied: set[int] = set()
    selected: list[int] = []
    for candidate in order.tolist():
        start = int(source.candidate_start[candidate].item())
        end = int(source.candidate_end[candidate].item())
        positions = set(range(start, end + 1))
        if occupied & positions:
            continue
        occupied.update(positions)
        selected.append(candidate)
    start_candidates = [
        index for index in selected if int(role_logits[index].argmax().item()) == START_ROLE
    ]
    outcome_candidates = [
        index
        for index in selected
        if int(role_logits[index].argmax().item()) == OUTCOME_ROLE
    ]
    word_candidates = sorted(
        (
            index
            for index in selected
            if int(role_logits[index].argmax().item()) == WORD_ROLE
        ),
        key=lambda index: int(source.candidate_start[index].item()),
    )
    return start_candidates + outcome_candidates + word_candidates, (
        len(start_candidates),
        len(outcome_candidates),
        len(word_candidates),
    )


def decode_span_logits(
    logits: SpanQuotientLogits,
    source: SpanLexicalSource,
    algebra: PresentedAlgebraConfig,
) -> tuple[DecodedChallenges, SpanDecodeAudit]:
    output_device = source.ids.device
    challenge_score = logits.kind.softmax(-1)[..., 1].masked_fill(
        ~source.record_mask, -torch.inf
    )
    record_index = challenge_score.topk(
        algebra.maximum_challenges, -1
    ).indices.detach().cpu()
    cpu_source = source if source.ids.device.type == "cpu" else source.to("cpu")
    cpu_role_logits = logits.role.detach().cpu()
    batch_size = source.ids.shape[0]
    maximum_selected = 2 + algebra.maximum_word_length
    selected_candidate = torch.full(
        (batch_size, algebra.maximum_challenges, maximum_selected),
        -1,
        dtype=torch.long,
    )
    selected_mask = torch.zeros_like(selected_candidate, dtype=torch.bool)
    valid = torch.zeros(
        batch_size,
        algebra.maximum_challenges,
        dtype=torch.bool,
    )
    missing_start = torch.zeros_like(valid)
    duplicate_start = torch.zeros_like(valid)
    missing_outcome = torch.zeros_like(valid)
    duplicate_outcome = torch.zeros_like(valid)
    missing_word = torch.zeros_like(valid)
    excess_word = torch.zeros_like(valid)
    nonexact_identity = torch.zeros_like(valid)
    start = torch.zeros_like(valid, dtype=torch.long)
    outcome = torch.zeros_like(valid, dtype=torch.long)
    length = torch.ones_like(valid, dtype=torch.long)
    word = torch.zeros(
        batch_size,
        algebra.maximum_challenges,
        algebra.maximum_word_length,
        dtype=torch.long,
    )
    for row in range(batch_size):
        for slot in range(algebra.maximum_challenges):
            record = int(record_index[row, slot].item())
            candidate_start = int(
                cpu_source.candidate_group_start[row, record].item()
            )
            candidate_end = int(cpu_source.candidate_group_end[row, record].item())
            candidates = torch.arange(
                candidate_start,
                candidate_end,
            )
            selected, counts = _decode_record_candidates(
                cpu_role_logits,
                cpu_source,
                candidates,
                algebra.maximum_word_length,
            )
            if selected:
                count = min(len(selected), maximum_selected)
                selected_candidate[row, slot, :count] = torch.tensor(selected[:count])
                selected_mask[row, slot, :count] = True
            start_count, outcome_count, word_count = counts
            missing_start[row, slot] = start_count == 0
            duplicate_start[row, slot] = start_count > 1
            missing_outcome[row, slot] = outcome_count == 0
            duplicate_outcome[row, slot] = outcome_count > 1
            missing_word[row, slot] = word_count == 0
            excess_word[row, slot] = word_count > algebra.maximum_word_length
            shape_valid = (
                start_count == 1
                and outcome_count == 1
                and 1 <= word_count <= algebra.maximum_word_length
            )
            if not shape_valid:
                continue
            start_index, outcome_index, *word_indices = selected
            selected_tensor = torch.tensor(selected)
            target_roles = cpu_source.candidate_target_role[selected_tensor]
            exact_valid = bool(target_roles[0].eq(START_ROLE))
            exact_valid = exact_valid and bool(target_roles[1].eq(OUTCOME_ROLE))
            exact_valid = exact_valid and bool(target_roles[2:].eq(WORD_ROLE).all())
            if not exact_valid:
                nonexact_identity[row, slot] = True
                continue
            start[row, slot] = cpu_source.candidate_target_value[start_index]
            outcome[row, slot] = cpu_source.candidate_target_value[outcome_index]
            length[row, slot] = len(word_indices)
            word[row, slot, : len(word_indices)] = cpu_source.candidate_target_value[
                torch.tensor(word_indices)
            ]
            valid[row, slot] = True
    word_position = torch.arange(
        algebra.maximum_word_length
    )
    word_mask = word_position[None, None] < length[..., None]
    return (
        DecodedChallenges(
            record_index=record_index.to(output_device),
            start=start.clamp(0, algebra.carrier_size - 1).to(output_device),
            outcome=outcome.clamp(0, algebra.carrier_size - 1).to(output_device),
            length=length.to(output_device),
            word=word.clamp(0, algebra.maximum_generators - 1).to(output_device),
            word_mask=word_mask.to(output_device),
        ),
        SpanDecodeAudit(
            selected_candidate=selected_candidate.to(output_device),
            selected_mask=selected_mask.to(output_device),
            valid=valid.to(output_device),
            missing_start=missing_start.to(output_device),
            duplicate_start=duplicate_start.to(output_device),
            missing_outcome=missing_outcome.to(output_device),
            duplicate_outcome=duplicate_outcome.to(output_device),
            missing_word=missing_word.to(output_device),
            excess_word=excess_word.to(output_device),
            nonexact_identity=nonexact_identity.to(output_device),
        ),
    )


def span_quotient_loss(
    logits: SpanQuotientLogits,
    source: SpanLexicalSource,
) -> tuple[torch.Tensor, dict[str, float]]:
    kind_loss = F.cross_entropy(
        logits.kind[source.record_mask],
        source.challenge_record[source.record_mask].long(),
    )
    challenge_candidate = source.challenge_record[
        source.candidate_batch, source.candidate_record
    ]
    weights = torch.tensor(
        [0.05, 1.0, 1.0, 1.0],
        device=logits.role.device,
        dtype=torch.float32,
    )
    role_loss = F.cross_entropy(
        logits.role[challenge_candidate],
        source.candidate_target_role[challenge_candidate],
        weight=weights,
    )
    return kind_loss + role_loss, {
        "kind_loss": float(kind_loss.item()),
        "span_role_loss": float(role_loss.item()),
    }


def _candidate_failure_counts(
    audit: SpanDecodeAudit,
    source: SpanLexicalSource,
    *,
    accepted_only: bool,
) -> dict[str, int]:
    mask = audit.selected_mask
    if accepted_only:
        mask = mask & audit.valid[..., None]
    selected = audit.selected_candidate[mask]
    if selected.numel() == 0:
        return {"partial": 0, "superset": 0, "overlap": 0}
    kinds = source.candidate_error_kind[selected]
    return {
        "partial": int(kinds.eq(ERROR_PARTIAL).sum().item()),
        "superset": int(kinds.eq(ERROR_SUPERSET).sum().item()),
        "overlap": int(kinds.eq(ERROR_OVERLAP).sum().item()),
    }


def _decoded_equal(left: DecodedChallenges, right: DecodedChallenges) -> torch.Tensor:
    return (
        left.record_index.eq(right.record_index).all(-1)
        & left.start.eq(right.start).all(-1)
        & left.outcome.eq(right.outcome).all(-1)
        & left.length.eq(right.length).all(-1)
        & left.word.eq(right.word).all((-1, -2))
        & left.word_mask.eq(right.word_mask).all((-1, -2))
    )


def _select_tables(
    decoded: DecodedChallenges,
    valid: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    selected, _, _ = select_with_challenges(
        candidates,
        decoded.start,
        decoded.word,
        decoded.word_mask,
        decoded.outcome,
        valid,
    )
    return selected


@torch.inference_mode()
def evaluate_cohort(
    parser: SpanQuotientChallengeParser,
    reasoner: LearnedPSPAGate,
    tokenizer: Tokenizer,
    *,
    family: int,
    length: int,
    count: int,
    batch_size: int,
    seed: int,
    renderer_seed: int,
    templates: tuple[int, ...],
    shifted_aliases: bool,
    device: torch.device,
) -> dict[str, Any]:
    totals = {
        "learned": 0,
        "oracle": 0,
        "shuffled": 0,
        "lineage": 0,
        "tuple": 0,
        "table": 0,
        "decoded_valid": 0,
        "class_zero_answer": 0,
        "class_zero_tuple": 0,
        "reindex_identical": 0,
        "start_mention": 0,
        "outcome_mention": 0,
        "word_mention": 0,
        "word_mentions": 0,
        "gold_mentions": 0,
        "exact_mentions": 0,
        "labeled_mentions": 0,
        "represented_mentions": 0,
        "selected_partial": 0,
        "selected_superset": 0,
        "selected_overlap": 0,
        "accepted_partial": 0,
        "accepted_superset": 0,
        "accepted_overlap": 0,
        "candidate_classes": 0,
        "selected_classes": 0,
        "missing_start": 0,
        "duplicate_start": 0,
        "missing_outcome": 0,
        "duplicate_outcome": 0,
        "missing_word": 0,
        "excess_word": 0,
        "nonexact_identity": 0,
    }
    batch_hashes: list[str] = []
    processed = 0
    while processed < count:
        current = min(batch_size, count - processed)
        cpu_batch = generate_batch(
            current,
            length,
            reasoner.algebra,
            seed=seed + processed * 1009,
            family=family,
        )
        batch_hashes.append(batch_sha256(cpu_batch))
        lexical = render_span_lexical_source(
            cpu_batch,
            reasoner.algebra,
            tokenizer,
            seed=renderer_seed + processed * 1013,
            templates=templates,
            shifted_aliases=shifted_aliases,
            seq_len=parser.model.cfg.seq_len,
        ).to(device)
        typed = render_semantic_source(
            cpu_batch,
            reasoner.algebra,
            seed=renderer_seed + processed * 1019,
            templates=(0, 1, 2, 3),
        ).to(device)
        batch = cpu_batch.to(device)
        memory = parser.encode_memory(lexical)
        logits = parser.score_memory(memory, lexical)
        decoded, audit = decode_span_logits(logits, lexical, reasoner.algebra)
        zero_logits = parser.score_memory(memory, lexical, class_messages=False)
        zero_decoded, zero_audit = decode_span_logits(
            zero_logits, lexical, reasoner.algebra
        )
        maximum_class = int(lexical.candidate_class.max().item())
        reindexed = replace(
            lexical,
            candidate_class=maximum_class - lexical.candidate_class,
        )
        reindex_logits = parser.score_memory(memory, reindexed)
        reindex_decoded, reindex_audit = decode_span_logits(
            reindex_logits, reindexed, reasoner.algebra
        )
        _, row_probabilities = reasoner.row_soft(
            typed.rendered, batch.generator_mask, hard=False
        )
        candidates = binary_completion_candidates(
            row_probabilities, batch.generator_mask
        )
        selected = _select_tables(decoded, audit.valid, candidates)
        selected_zero = _select_tables(zero_decoded, zero_audit.valid, candidates)
        selected_reindex = _select_tables(
            reindex_decoded, reindex_audit.valid, candidates
        )
        oracle, _, _ = select_with_challenges(
            candidates,
            batch.challenge_start,
            batch.challenge_word,
            batch.challenge_word_mask,
            batch.challenge_outcome,
            batch.challenge_mask,
        )
        shuffled = _select_tables(
            replace(decoded, outcome=decoded.outcome.roll(1, 0)),
            audit.valid,
            candidates,
        )

        def answer(tables: torch.Tensor) -> torch.Tensor:
            return execute_word(
                tables,
                batch.query_start,
                batch.query_word,
                batch.query_word_mask,
            ).argmax(-1)

        true_record, true_start, true_outcome, true_length, true_word = (
            gather_lexical_targets(lexical, decoded.record_index)
        )
        positions = torch.arange(
            reasoner.algebra.maximum_word_length, device=device
        )
        true_word_mask = positions[None, None] < true_length[..., None]
        word_exact = (decoded.word.eq(true_word) | ~true_word_mask).all(-1)
        tuple_exact = (
            true_record
            & decoded.start.eq(true_start)
            & decoded.outcome.eq(true_outcome)
            & decoded.length.eq(true_length)
            & word_exact
            & audit.valid
        ).all(-1)
        (
            zero_true_record,
            zero_true_start,
            zero_true_outcome,
            zero_true_length,
            zero_true_word,
        ) = gather_lexical_targets(lexical, zero_decoded.record_index)
        zero_true_word_mask = positions[None, None] < zero_true_length[..., None]
        zero_word_exact = (
            zero_decoded.word.eq(zero_true_word) | ~zero_true_word_mask
        ).all(-1)
        zero_tuple = (
            zero_true_record
            & zero_decoded.start.eq(zero_true_start)
            & zero_decoded.outcome.eq(zero_true_outcome)
            & zero_decoded.length.eq(zero_true_length)
            & zero_word_exact
            & zero_audit.valid
        ).all(-1)
        table_exact = selected.argmax(-1).eq(batch.true_tables.long()).all((-1, -2))
        answer_selected = answer(selected)
        answer_reindex = answer(selected_reindex)
        reindex_identical = (
            _decoded_equal(decoded, reindex_decoded)
            & audit.valid.eq(reindex_audit.valid).all(-1)
            & selected.eq(selected_reindex).all((-1, -2, -3))
            & answer_selected.eq(answer_reindex)
        )

        chosen = audit.selected_candidate.clamp_min(0)
        chosen_role = lexical.candidate_target_role[chosen]
        chosen_value = lexical.candidate_target_value[chosen]
        start_mention = audit.selected_mask[..., 0] & chosen_role[..., 0].eq(
            START_ROLE
        )
        outcome_mention = audit.selected_mask[..., 1] & chosen_role[..., 1].eq(
            OUTCOME_ROLE
        )
        word_selected = audit.selected_mask[..., 2:]
        word_role = chosen_role[..., 2:].eq(WORD_ROLE)
        word_value = chosen_value[..., 2:]
        word_mention = (
            word_selected
            & word_role
            & word_value.eq(true_word)
            & true_word_mask
        )
        gold_mentions = 2 * true_record.long() + true_length * true_record.long()
        exact_mentions = (
            start_mention.long()
            + outcome_mention.long()
            + word_mention.sum(-1)
        )
        failures = _candidate_failure_counts(
            audit, lexical, accepted_only=False
        )
        accepted_failures = _candidate_failure_counts(
            audit, lexical, accepted_only=True
        )
        selected_ids = audit.selected_candidate[audit.selected_mask]
        totals["learned"] += int(answer_selected.eq(batch.answer).sum().item())
        totals["oracle"] += int(answer(oracle).eq(batch.answer).sum().item())
        totals["shuffled"] += int(answer(shuffled).eq(batch.answer).sum().item())
        totals["lineage"] += int(
            answer(selected.roll(1, 0)).eq(batch.answer).sum().item()
        )
        totals["tuple"] += int(tuple_exact.sum().item())
        totals["table"] += int(table_exact.sum().item())
        totals["decoded_valid"] += int(audit.valid.all(-1).sum().item())
        totals["class_zero_answer"] += int(
            answer(selected_zero).eq(batch.answer).sum().item()
        )
        totals["class_zero_tuple"] += int(zero_tuple.sum().item())
        totals["reindex_identical"] += int(reindex_identical.sum().item())
        totals["start_mention"] += int(start_mention.sum().item())
        totals["outcome_mention"] += int(outcome_mention.sum().item())
        totals["word_mention"] += int(word_mention.sum().item())
        totals["word_mentions"] += int(true_word_mask.sum().item())
        totals["gold_mentions"] += int(gold_mentions.sum().item())
        totals["exact_mentions"] += int(exact_mentions.sum().item())
        totals["labeled_mentions"] += int(lexical.labeled_mentions.sum().item())
        totals["represented_mentions"] += int(
            lexical.represented_mentions.sum().item()
        )
        totals["selected_partial"] += failures["partial"]
        totals["selected_superset"] += failures["superset"]
        totals["selected_overlap"] += failures["overlap"]
        totals["accepted_partial"] += accepted_failures["partial"]
        totals["accepted_superset"] += accepted_failures["superset"]
        totals["accepted_overlap"] += accepted_failures["overlap"]
        totals["candidate_classes"] += maximum_class + 1
        totals["selected_classes"] += int(
            lexical.candidate_class[selected_ids].unique().numel()
            if selected_ids.numel()
            else 0
        )
        for metric in (
            "missing_start",
            "duplicate_start",
            "missing_outcome",
            "duplicate_outcome",
            "missing_word",
            "excess_word",
            "nonexact_identity",
        ):
            totals[metric] += int(getattr(audit, metric).sum().item())
        processed += current

    challenge_total = count * reasoner.algebra.maximum_challenges
    return {
        "family": FAMILIES[family],
        "length": length,
        "count": count,
        "seed": seed,
        "renderer_seed": renderer_seed,
        "templates": list(templates),
        "shifted_aliases": shifted_aliases,
        "batch_sha256": hashlib.sha256(
            "\n".join(batch_hashes).encode("ascii")
        ).hexdigest(),
        "learned_accuracy": totals["learned"] / count,
        "oracle_accuracy": totals["oracle"] / count,
        "shuffle_outcome_accuracy": totals["shuffled"] / count,
        "lineage_swap_accuracy": totals["lineage"] / count,
        "challenge_tuple_exact": totals["tuple"] / count,
        "selected_table_exact": totals["table"] / count,
        "decoded_all_valid": totals["decoded_valid"] / count,
        "class_zero_answer_accuracy": totals["class_zero_answer"] / count,
        "class_zero_tuple_exact": totals["class_zero_tuple"] / count,
        "class_reindex_bit_identical": totals["reindex_identical"] / count,
        "start_mention_exact": totals["start_mention"] / challenge_total,
        "outcome_mention_exact": totals["outcome_mention"] / challenge_total,
        "word_mention_exact": totals["word_mention"]
        / max(totals["word_mentions"], 1),
        "gold_mention_exact": totals["exact_mentions"]
        / max(totals["gold_mentions"], 1),
        "representability": totals["represented_mentions"]
        / max(totals["labeled_mentions"], 1),
        "selected_partial": totals["selected_partial"],
        "selected_superset": totals["selected_superset"],
        "selected_overlap": totals["selected_overlap"],
        "accepted_partial": totals["accepted_partial"],
        "accepted_superset": totals["accepted_superset"],
        "accepted_overlap": totals["accepted_overlap"],
        "candidate_classes": totals["candidate_classes"],
        "selected_classes": totals["selected_classes"],
        "missing_start": totals["missing_start"],
        "duplicate_start": totals["duplicate_start"],
        "missing_outcome": totals["missing_outcome"],
        "duplicate_outcome": totals["duplicate_outcome"],
        "missing_word": totals["missing_word"],
        "excess_word": totals["excess_word"],
        "nonexact_identity": totals["nonexact_identity"],
    }


def aggregate_evaluations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    averaged = (
        "learned_accuracy",
        "oracle_accuracy",
        "shuffle_outcome_accuracy",
        "lineage_swap_accuracy",
        "challenge_tuple_exact",
        "selected_table_exact",
        "decoded_all_valid",
        "class_zero_answer_accuracy",
        "class_zero_tuple_exact",
        "class_reindex_bit_identical",
        "start_mention_exact",
        "outcome_mention_exact",
        "word_mention_exact",
        "gold_mention_exact",
        "representability",
    )
    counted = (
        "selected_partial",
        "selected_superset",
        "selected_overlap",
        "accepted_partial",
        "accepted_superset",
        "accepted_overlap",
        "candidate_classes",
        "selected_classes",
        "missing_start",
        "duplicate_start",
        "missing_outcome",
        "duplicate_outcome",
        "missing_word",
        "excess_word",
        "nonexact_identity",
    )
    for split in ("development", "lexical_shift"):
        selected = [row for row in rows if row["split"] == split]
        total = sum(row["count"] for row in selected)
        summary = {
            metric: sum(row[metric] * row["count"] for row in selected) / total
            for metric in averaged
        }
        summary.update(
            {metric: sum(row[metric] for row in selected) for metric in counted}
        )
        summary["count"] = total
        summary["minimum_cohort_learned_accuracy"] = min(
            row["learned_accuracy"] for row in selected
        )
        summary["shuffle_drop_points"] = 100 * (
            summary["learned_accuracy"] - summary["shuffle_outcome_accuracy"]
        )
        summary["lineage_drop_points"] = 100 * (
            summary["learned_accuracy"] - summary["lineage_swap_accuracy"]
        )
        summaries[split] = summary
    return summaries


def assess_gate(summary: dict[str, Any]) -> dict[str, bool]:
    development = summary["development"]
    shift = summary["lexical_shift"]
    return {
        "representability_exact": (
            development["representability"] == 1.0
            and shift["representability"] == 1.0
        ),
        "development_answer_ge_95": development["learned_accuracy"] >= 0.95,
        "development_tuple_ge_95": development["challenge_tuple_exact"] >= 0.95,
        "development_table_ge_95": development["selected_table_exact"] >= 0.95,
        "shift_answer_ge_90": shift["learned_accuracy"] >= 0.90,
        "shift_tuple_ge_90": shift["challenge_tuple_exact"] >= 0.90,
        "shift_table_ge_90": shift["selected_table_exact"] >= 0.90,
        "shift_mentions_ge_90": shift["gold_mention_exact"] >= 0.90,
        "every_shift_cohort_answer_ge_90": (
            shift["minimum_cohort_learned_accuracy"] >= 0.90
        ),
        "development_oracle_ge_98": development["oracle_accuracy"] >= 0.98,
        "shift_oracle_ge_98": shift["oracle_accuracy"] >= 0.98,
        "development_shuffle_drop_ge_20": development["shuffle_drop_points"] >= 20,
        "shift_shuffle_drop_ge_20": shift["shuffle_drop_points"] >= 20,
        "development_lineage_drop_ge_20": development["lineage_drop_points"] >= 20,
        "shift_lineage_drop_ge_20": shift["lineage_drop_points"] >= 20,
        "zero_accepted_partial_or_superset": (
            development["accepted_partial"] == 0
            and development["accepted_superset"] == 0
            and shift["accepted_partial"] == 0
            and shift["accepted_superset"] == 0
        ),
        "class_reindex_bit_identical": (
            development["class_reindex_bit_identical"] == 1.0
            and shift["class_reindex_bit_identical"] == 1.0
        ),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise PresentedReasoningError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _adapter_state(parser: SpanQuotientChallengeParser) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in parser.state_dict().items()
        if not name.startswith("model.")
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise PresentedReasoningError("CUDA is required unless --allow-cpu is explicit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    input_hashes = {
        "base": sha256_file(args.base),
        "tokenizer": sha256_file(args.tokenizer),
        "warm_adapter": sha256_file(args.warm_adapter),
        "reasoner": sha256_file(args.reasoner_checkpoint),
        "first_subtoken_report": sha256_file(args.first_subtoken_report),
    }
    expected = {
        "warm_adapter": EXPECTED_WARM_ADAPTER_SHA256,
        "reasoner": EXPECTED_REASONER_SHA256,
        "first_subtoken_report": EXPECTED_FIRST_SUBTOKEN_REPORT_SHA256,
    }
    for name, digest in expected.items():
        if input_hashes[name] != digest:
            raise PresentedReasoningError(f"{name} hash differs")
    with args.first_subtoken_report.open(encoding="utf-8") as source:
        first_subtoken_control = json.load(source)

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model, model_config, backbone_receipt = load_frozen_pointer_backbone(
        args.base, device=device
    )
    warm_payload = torch.load(
        args.warm_adapter, map_location="cpu", weights_only=True
    )
    warm_metadata = warm_payload["compiler"]
    parser_config = LexicalBridgeConfig(
        layer=int(warm_metadata["layer"]),
        width=int(warm_metadata["width"]),
        heads=int(warm_metadata["heads"]),
        encoder_layers=int(warm_metadata["encoder_layers"]),
        ff=int(warm_metadata["ff"]),
    )
    del warm_payload
    parser = SpanQuotientChallengeParser(model, parser_config).to(device)
    load_warm_adapter(parser, args.warm_adapter)

    reasoner_payload = torch.load(
        args.reasoner_checkpoint, map_location="cpu", weights_only=True
    )
    algebra = PresentedAlgebraConfig(**reasoner_payload["algebra_config"])
    language = LanguageConfig(**reasoner_payload["language_config"])
    reasoner = LearnedPSPAGate(algebra, language).to(device)
    reasoner.load_state_dict(reasoner_payload["model"])
    reasoner.eval().requires_grad_(False)
    del reasoner_payload

    trainable = list(parser.adapter_parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_templates = tuple(int(value) for value in args.train_templates.split(","))
    started = time.monotonic()
    train_log: list[dict[str, float | int]] = []
    for update in range(1, args.updates + 1):
        cpu_batch = generate_batch(
            args.batch_size,
            1 + ((update * 997 + args.seed) % 4),
            algebra,
            seed=args.data_seed + update,
        )
        source = render_span_lexical_source(
            cpu_batch,
            algebra,
            tokenizer,
            seed=args.renderer_seed + update,
            templates=train_templates,
            shifted_aliases=False,
            seq_len=model_config.seq_len,
        ).to(device)
        parser.train()
        parser.model.eval()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = parser(source)
            loss, loss_metrics = span_quotient_loss(logits, source)
        if not torch.isfinite(loss):
            raise PresentedReasoningError(f"non-finite loss at update {update}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(grad_norm):
            raise PresentedReasoningError(f"non-finite gradient at update {update}")
        optimizer.step()
        if update == 1 or update % args.log_every == 0 or update == args.updates:
            elapsed = max(time.monotonic() - started, 1e-6)
            row = {
                "update": update,
                "loss": float(loss.item()),
                **loss_metrics,
                "grad_norm": float(grad_norm.item()),
                "examples_per_second": update * args.batch_size / elapsed,
            }
            train_log.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    elapsed = time.monotonic() - started
    parser.eval()
    evaluations: list[dict[str, Any]] = []
    for split, templates, shifted, offset in (
        ("development", train_templates, False, 0),
        ("lexical_shift", (3,), True, 100_000),
    ):
        for family in range(len(FAMILIES)):
            for depth in (8, 12):
                row = evaluate_cohort(
                    parser,
                    reasoner,
                    tokenizer,
                    family=family,
                    length=depth,
                    count=args.eval_count,
                    batch_size=args.eval_batch_size,
                    seed=args.eval_seed + offset + family * 100 + depth,
                    renderer_seed=(
                        args.eval_renderer_seed + offset + family * 100 + depth
                    ),
                    templates=templates,
                    shifted_aliases=shifted,
                    device=device,
                )
                row["split"] = split
                evaluations.append(row)
                print(json.dumps({"evaluation": row}, sort_keys=True), flush=True)
    summary = aggregate_evaluations(evaluations)
    gates = assess_gate(summary)
    checkpoint = args.output.with_suffix(".pt")
    if checkpoint.exists():
        raise PresentedReasoningError(f"refusing existing checkpoint: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary_checkpoint = checkpoint.with_suffix(".pt.partial")
    torch.save(
        {
            "schema": SCHEMA,
            "seed": args.seed,
            "parser_config": asdict(parser_config),
            "max_span_width": MAX_SPAN_WIDTH,
            "adapter_state": _adapter_state(parser),
            "input_hashes": input_hashes,
        },
        temporary_checkpoint,
    )
    os.replace(temporary_checkpoint, checkpoint)
    peak_memory = (
        int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
    )
    report = {
        "schema": SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "all_gates_pass": all(gates.values()),
        "gates": gates,
        "seed": args.seed,
        "source_commit": args.source_commit,
        "data_seed": args.data_seed,
        "renderer_seed": args.renderer_seed,
        "eval_seed": args.eval_seed,
        "eval_renderer_seed": args.eval_renderer_seed,
        "train_templates": list(train_templates),
        "max_span_width": MAX_SPAN_WIDTH,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "charged_examples": args.updates * args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "parser_config": asdict(parser_config),
        "adapter_parameters": sum(parameter.numel() for parameter in trainable),
        "base_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "base_checkpoint_format": backbone_receipt.checkpoint_format,
        "base_import": backbone_receipt.base_import,
        "input_hashes": input_hashes,
        "elapsed_seconds": elapsed,
        "examples_per_second": args.updates * args.batch_size / elapsed,
        "peak_memory_bytes": peak_memory,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "first_subtoken_control": {
            "report_sha256": input_hashes["first_subtoken_report"],
            "summary": first_subtoken_control["summary"],
        },
        "train_log": train_log,
        "evaluations": evaluations,
        "summary": summary,
        "decision": (
            "preserve_span_quotient_for_diverge"
            if all(gates.values())
            else "close_span_quotient_and_record_grounding_prerequisite_failure"
        ),
        "claim_boundary": (
            "Exact-surface whole-mention occurrence quotienting for controlled "
            "lexical challenges into frozen CSDC; not nominal coreference, "
            "DIVERGE, unrestricted language, or public reasoning."
        ),
    }
    _atomic_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--warm-adapter", type=Path, required=True)
    parser.add_argument("--reasoner-checkpoint", type=Path, required=True)
    parser.add_argument("--first-subtoken-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed", type=int, default=2026080507)
    parser.add_argument("--data-seed", type=int, default=202608051000)
    parser.add_argument("--renderer-seed", type=int, default=202608052000)
    parser.add_argument("--eval-seed", type=int, default=202608053000)
    parser.add_argument("--eval-renderer-seed", type=int, default=202608054000)
    parser.add_argument("--train-templates", default="0,1,2")
    parser.add_argument("--updates", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-count", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    print(json.dumps({"final": report["status"], "gates": report["gates"]}))


if __name__ == "__main__":
    main()
