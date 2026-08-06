#!/usr/bin/env python3
"""Candidate-only structural mention binder for DIVERGE-MQB1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


REGISTER_COUNT = 5
VALUE_COUNT = 128
PHASE_COUNT = 2
FIELD_COUNT = REGISTER_COUNT * PHASE_COUNT
NONE_VALUE = VALUE_COUNT
NONE_PHASE = PHASE_COUNT
NONE_ADDRESS = REGISTER_COUNT
MAX_CANDIDATES = 12


class MQB1ContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class MQB1Config:
    input_width: int = 192
    width: int = 256
    heads: int = 4
    layers: int = 2
    ff_multiplier: int = 4
    pair_width: int = 96
    maximum_candidates: int = MAX_CANDIDATES

    def validate(self) -> None:
        if self.input_width <= 0 or self.width <= 0 or self.pair_width <= 0:
            raise MQB1ContractError("MQB1 widths must be positive")
        if self.width % self.heads or self.layers <= 0 or self.ff_multiplier <= 0:
            raise MQB1ContractError("MQB1 encoder geometry differs")
        if self.maximum_candidates != MAX_CANDIDATES:
            raise MQB1ContractError("MQB1 candidate cap differs from the frozen gate")


@dataclass(frozen=True)
class MentionBinderLogits:
    value: torch.Tensor
    phase: torch.Tensor
    address: torch.Tensor
    field: torch.Tensor
    pair: torch.Tensor


@dataclass(frozen=True)
class FieldAssignment:
    candidate_for_field: torch.Tensor
    score: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class MentionBinding:
    before: torch.Tensor
    after: torch.Tensor
    valid: torch.Tensor
    provenance: torch.Tensor
    selected_values: torch.Tensor
    selected_pair_score: torch.Tensor
    overflow: torch.Tensor


def exact_field_assignment(
    scores: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> FieldAssignment:
    """Maximum-weight one-to-one field assignment with optional candidates."""

    if (
        scores.ndim != 3
        or scores.shape[2] != FIELD_COUNT
        or candidate_mask.shape != scores.shape[:2]
        or candidate_mask.dtype != torch.bool
    ):
        raise MQB1ContractError("field assignment tensor interface differs")
    batch, candidates, _ = scores.shape
    if candidates < FIELD_COUNT or candidates > MAX_CANDIDATES:
        raise MQB1ContractError("field assignment candidate count differs")
    negative = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~candidate_mask[..., None], negative)
    state_count = 1 << FIELD_COUNT
    dp = torch.full(
        (batch, state_count), negative, dtype=scores.dtype, device=scores.device
    )
    dp[:, 0] = 0
    masks = torch.arange(state_count, device=scores.device)
    backtrace: list[torch.Tensor] = []
    for candidate in range(candidates):
        updated = dp.clone()
        chosen = torch.full(
            (batch, state_count), -1, dtype=torch.int8, device=scores.device
        )
        for field in range(FIELD_COUNT):
            sources = masks[(masks & (1 << field)).eq(0)]
            targets = sources | (1 << field)
            proposed = dp[:, sources] + scores[:, candidate, field, None]
            current = updated[:, targets]
            better = proposed > current
            updated[:, targets] = torch.where(better, proposed, current)
            chosen[:, targets] = torch.where(
                better,
                torch.full_like(chosen[:, targets], field),
                chosen[:, targets],
            )
        dp = updated
        backtrace.append(chosen)

    full_state = state_count - 1
    score = dp[:, full_state]
    valid = score > negative / 2
    assignment = torch.full(
        (batch, FIELD_COUNT), -1, dtype=torch.long, device=scores.device
    )
    state = torch.full(
        (batch,), full_state, dtype=torch.long, device=scores.device
    )
    rows = torch.arange(batch, device=scores.device)
    for candidate in range(candidates - 1, -1, -1):
        field = backtrace[candidate][rows, state].long()
        active = valid & field.ge(0)
        if active.any():
            assignment[rows[active], field[active]] = candidate
            state[active] ^= 1 << field[active]
    valid = valid & assignment.ge(0).all(-1) & state.eq(0)
    return FieldAssignment(assignment, score, valid)


class MentionEvidenceBinder(nn.Module):
    """Bind model-predicted values to complete typed contextual mentions."""

    def __init__(self, config: MQB1Config):
        super().__init__()
        config.validate()
        self.config = config
        self.input_projection = nn.Linear(config.input_width, config.width, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=config.heads,
            dim_feedforward=config.width * config.ff_multiplier,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=config.layers, enable_nested_tensor=False
        )
        self.local_projection = nn.Sequential(
            nn.LayerNorm(config.width * 4),
            nn.Linear(config.width * 4, config.width),
            nn.GELU(),
            nn.LayerNorm(config.width),
        )
        self.value_head = nn.Linear(config.width, VALUE_COUNT + 1)
        self.phase_head = nn.Linear(config.width, PHASE_COUNT + 1)
        self.address_head = nn.Linear(config.width, REGISTER_COUNT + 1)
        self.pair_projection = nn.Linear(config.width, config.pair_width, bias=False)
        self.pair_scale = nn.Parameter(torch.tensor(1.0))
        self.pair_bias = nn.Parameter(torch.tensor(0.0))

    def _mentions(
        self, word_features: torch.Tensor, word_mask: torch.Tensor
    ) -> torch.Tensor:
        if (
            word_features.ndim != 3
            or word_features.shape[2] != self.config.input_width
            or word_mask.shape != word_features.shape[:2]
            or word_mask.dtype != torch.bool
            or not word_mask.any(-1).all()
        ):
            raise MQB1ContractError("mention binder tensor interface differs")
        memory = self.input_projection(word_features)
        memory = self.encoder(memory, src_key_padding_mask=~word_mask)
        zero = torch.zeros_like(memory[:, :1])
        left_one = torch.cat((zero, memory[:, :-1]), dim=1)
        left_two = torch.cat((zero, zero, memory[:, :-2]), dim=1)
        right_one = torch.cat((memory[:, 1:], zero), dim=1)
        mention = self.local_projection(
            torch.cat((memory, left_one, left_two, right_one), dim=-1)
        )
        return mention.masked_fill(~word_mask[..., None], 0)

    def forward(
        self, word_features: torch.Tensor, word_mask: torch.Tensor
    ) -> MentionBinderLogits:
        mention = self._mentions(word_features, word_mask)
        value = self.value_head(mention).float()
        phase = self.phase_head(mention).float()
        address = self.address_head(mention).float()
        numeric_margin = value[..., :VALUE_COUNT].logsumexp(-1) - value[..., NONE_VALUE]
        phase_margin = phase[..., :PHASE_COUNT] - phase[..., NONE_PHASE, None]
        address_margin = address[..., :REGISTER_COUNT] - address[..., NONE_ADDRESS, None]
        field = torch.stack(
            [
                numeric_margin
                + phase_margin[..., field // REGISTER_COUNT]
                + address_margin[..., field % REGISTER_COUNT]
                for field in range(FIELD_COUNT)
            ],
            dim=-1,
        )
        pair_memory = F.normalize(self.pair_projection(mention).float(), dim=-1)
        pair = (
            self.pair_scale.float()
            * torch.matmul(pair_memory, pair_memory.transpose(-1, -2))
            * math.sqrt(self.config.pair_width)
            + self.pair_bias.float()
        )
        pair = 0.5 * (pair + pair.transpose(-1, -2))
        return MentionBinderLogits(value, phase, address, field, pair)

    @torch.no_grad()
    def decode(
        self,
        logits: MentionBinderLogits,
        word_mask: torch.Tensor,
    ) -> MentionBinding:
        if (
            logits.value.ndim != 3
            or logits.value.shape[-1] != VALUE_COUNT + 1
            or logits.phase.shape != (*logits.value.shape[:2], PHASE_COUNT + 1)
            or logits.address.shape != (*logits.value.shape[:2], REGISTER_COUNT + 1)
            or logits.field.shape != (*logits.value.shape[:2], FIELD_COUNT)
            or logits.pair.shape != (*logits.value.shape[:2], logits.value.shape[1])
            or word_mask.shape != logits.value.shape[:2]
        ):
            raise MQB1ContractError("mention decode tensor interface differs")
        batch, words = word_mask.shape
        if words < FIELD_COUNT:
            raise MQB1ContractError("mention source cannot cover all typed fields")
        numeric_values = logits.value[..., :VALUE_COUNT]
        best_value_score, best_value = numeric_values.max(-1)
        numeric_margin = best_value_score - logits.value[..., NONE_VALUE]
        ranked = numeric_margin.masked_fill(~word_mask, -torch.inf)
        candidate_count = min(self.config.maximum_candidates, words)
        _, candidate_words = ranked.topk(candidate_count, dim=-1)
        candidate_mask = word_mask.gather(1, candidate_words)
        field_scores = logits.field.gather(
            1, candidate_words[..., None].expand(-1, -1, FIELD_COUNT)
        )
        assignment = exact_field_assignment(field_scores, candidate_mask)
        candidate_for_field = assignment.candidate_for_field.clamp_min(0)
        provenance = candidate_words.gather(1, candidate_for_field)
        selected_values = best_value.gather(1, provenance)
        selected_margins = numeric_margin.gather(1, provenance)
        overflow = ((numeric_margin > 0) & word_mask).sum(-1) > MAX_CANDIDATES
        before_words = provenance[:, :REGISTER_COUNT]
        after_words = provenance[:, REGISTER_COUNT:]
        rows = torch.arange(batch, device=word_mask.device)[:, None]
        pair_score = logits.pair[rows, before_words, after_words]
        valid = (
            assignment.valid
            & selected_margins.gt(0).all(-1)
            & pair_score.gt(0).all(-1)
            & ~overflow
        )
        before = selected_values[:, :REGISTER_COUNT]
        after = selected_values[:, REGISTER_COUNT:]
        return MentionBinding(
            before=before,
            after=after,
            valid=valid,
            provenance=provenance,
            selected_values=selected_values,
            selected_pair_score=pair_score,
            overflow=overflow,
        )

    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def source_audit() -> dict[str, object]:
    source = inspect.getsource(sys.modules[__name__])
    executable = source[: source.index("def source_audit")]
    forbidden = (
        "from diverge_" + "mei1_data",
        "from diverge_" + "v0",
        "apply_" + "transaction",
        "exact_" + "program",
        "render_" + "probe",
        "tokenizer.decode",
        "import re",
    )
    findings = [needle for needle in forbidden if needle in executable]
    return {"pass": not findings, "forbidden_findings": findings}


def architecture_receipt(model: MentionEvidenceBinder) -> dict[str, object]:
    return {
        "config": asdict(model.config),
        "trainable_parameters": model.trainable_parameters(),
        "field_count": FIELD_COUNT,
        "value_count": VALUE_COUNT,
        "candidate_cap": MAX_CANDIDATES,
        "whole_mention_values": True,
        "exact_one_to_one_assignment": True,
        "fieldwise_averaging": False,
        "source_audit": source_audit(),
    }
