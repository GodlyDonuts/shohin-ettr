#!/usr/bin/env python3
"""Candidate-only query-conditioned gatherer for DIVERGE-QTG1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from diverge_mqb1_runtime import (
    FIELD_COUNT,
    MAX_CANDIDATES,
    REGISTER_COUNT,
    VALUE_COUNT,
    exact_field_assignment,
)


NONE_VALUE = VALUE_COUNT


class QTG1ContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class QTG1Config:
    input_width: int = 192
    width: int = 256
    heads: int = 4
    layers: int = 1
    ff_multiplier: int = 4
    pointer_width: int = 96
    maximum_candidates: int = MAX_CANDIDATES

    def validate(self) -> None:
        if min(self.input_width, self.width, self.pointer_width) <= 0:
            raise QTG1ContractError("QTG1 widths must be positive")
        if self.width % self.heads or self.layers <= 0 or self.ff_multiplier <= 0:
            raise QTG1ContractError("QTG1 encoder geometry differs")
        if self.maximum_candidates != MAX_CANDIDATES:
            raise QTG1ContractError("QTG1 candidate cap differs")


@dataclass(frozen=True)
class QueryGatherLogits:
    pointer: torch.Tensor
    value: torch.Tensor
    field: torch.Tensor


@dataclass(frozen=True)
class QueryGatherBinding:
    before: torch.Tensor
    after: torch.Tensor
    valid: torch.Tensor
    provenance: torch.Tensor
    selected_values: torch.Tensor
    overflow: torch.Tensor


class QueryConditionedGatherer(nn.Module):
    """Gather one atomic source mention under each typed language query."""

    def __init__(self, config: QTG1Config):
        super().__init__()
        config.validate()
        self.config = config
        self.query_projection = nn.Linear(config.input_width, config.width, bias=False)
        self.evidence_projection = nn.Linear(
            config.input_width, config.width, bias=False
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=config.heads,
            dim_feedforward=config.width * config.ff_multiplier,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.evidence_encoder = nn.TransformerEncoder(
            layer, num_layers=config.layers, enable_nested_tensor=False
        )
        self.pointer_query = nn.Linear(config.width, config.pointer_width, bias=False)
        self.pointer_key = nn.Linear(config.width, config.pointer_width, bias=False)
        self.value_head = nn.Sequential(
            nn.LayerNorm(config.width * 3),
            nn.Linear(config.width * 3, config.width),
            nn.GELU(),
            nn.Linear(config.width, VALUE_COUNT + 1),
        )

    def forward(
        self,
        query_features: torch.Tensor,
        query_mask: torch.Tensor,
        evidence_features: torch.Tensor,
        evidence_mask: torch.Tensor,
    ) -> QueryGatherLogits:
        if (
            query_features.ndim != 4
            or query_features.shape[:2] != evidence_features.shape[:2]
            or query_features.shape[1] != FIELD_COUNT
            or query_features.shape[-1] != self.config.input_width
            or evidence_features.ndim != 4
            or evidence_features.shape[-1] != self.config.input_width
            or query_mask.shape != query_features.shape[:3]
            or evidence_mask.shape != evidence_features.shape[:3]
            or query_mask.dtype != torch.bool
            or evidence_mask.dtype != torch.bool
            or not query_mask.any(-1).all()
            or not evidence_mask.any(-1).all()
        ):
            raise QTG1ContractError("QTG1 tensor interface differs")
        batch, fields, words, _ = evidence_features.shape
        query = self.query_projection(query_features)
        query = (query * query_mask[..., None]).sum(2) / query_mask.sum(2).clamp_min(1)[
            ..., None
        ]
        evidence = self.evidence_projection(evidence_features)
        flat = evidence.reshape(batch * fields, words, self.config.width)
        flat_mask = evidence_mask.reshape(batch * fields, words)
        flat = self.evidence_encoder(flat, src_key_padding_mask=~flat_mask)
        evidence = flat.reshape(batch, fields, words, self.config.width)
        pointer = torch.einsum(
            "bfd,bftd->bft",
            self.pointer_query(query),
            self.pointer_key(evidence),
        ) / math.sqrt(self.config.pointer_width)
        expanded_query = query[:, :, None].expand(-1, -1, words, -1)
        value = self.value_head(
            torch.cat((evidence, expanded_query, evidence * expanded_query), dim=-1)
        ).float()
        numeric_margin = value[..., :VALUE_COUNT].logsumexp(-1) - value[..., NONE_VALUE]
        field = pointer.float() + numeric_margin
        pointer = pointer.float().masked_fill(~evidence_mask, -torch.inf)
        field = field.masked_fill(~evidence_mask, -torch.inf)
        return QueryGatherLogits(pointer, value, field)

    @torch.no_grad()
    def decode(
        self,
        logits: QueryGatherLogits,
        evidence_mask: torch.Tensor,
    ) -> QueryGatherBinding:
        if (
            logits.pointer.ndim != 3
            or logits.pointer.shape[1] != FIELD_COUNT
            or logits.value.shape
            != (*logits.pointer.shape, VALUE_COUNT + 1)
            or logits.field.shape != logits.pointer.shape
            or evidence_mask.shape != logits.pointer.shape
            or evidence_mask.dtype != torch.bool
        ):
            raise QTG1ContractError("QTG1 decode tensor interface differs")
        batch, _, words = logits.pointer.shape
        if words < FIELD_COUNT:
            raise QTG1ContractError("QTG1 evidence cannot cover every field")
        numeric = logits.value[..., :VALUE_COUNT]
        best_score, best_value = numeric.max(-1)
        numeric_margin = best_score - logits.value[..., NONE_VALUE]
        candidate_score = numeric_margin.max(1).values.masked_fill(
            ~evidence_mask[:, 0], -torch.inf
        )
        candidate_count = min(MAX_CANDIDATES, words)
        _, candidate_words = candidate_score.topk(candidate_count, dim=-1)
        candidate_mask = evidence_mask[:, 0].gather(1, candidate_words)
        word_field_scores = logits.field.transpose(1, 2)
        candidate_field_scores = word_field_scores.gather(
            1, candidate_words[..., None].expand(-1, -1, FIELD_COUNT)
        )
        assignment = exact_field_assignment(candidate_field_scores, candidate_mask)
        candidate_for_field = assignment.candidate_for_field.clamp_min(0)
        provenance = candidate_words.gather(1, candidate_for_field)
        rows = torch.arange(batch, device=evidence_mask.device)[:, None]
        fields = torch.arange(FIELD_COUNT, device=evidence_mask.device)[None]
        selected_values = best_value[rows, fields, provenance]
        selected_margins = numeric_margin[rows, fields, provenance]
        overflow = ((candidate_score > 0) & evidence_mask[:, 0]).sum(-1) > MAX_CANDIDATES
        valid = assignment.valid & selected_margins.gt(0).all(-1) & ~overflow
        return QueryGatherBinding(
            before=selected_values[:, :REGISTER_COUNT],
            after=selected_values[:, REGISTER_COUNT:],
            valid=valid,
            provenance=provenance,
            selected_values=selected_values,
            overflow=overflow,
        )

    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def source_audit() -> dict[str, object]:
    source = inspect.getsource(sys.modules[__name__])
    executable = source[: source.index("def source_audit")]
    forbidden = (
        "from diverge_" + "mei1_data",
        "from diverge_" + "mqb1_data",
        "from diverge_" + "v0",
        "apply_" + "transaction",
        "exact_" + "program",
        "render_" + "probe",
        "tokenizer.decode",
        "import re",
    )
    findings = [needle for needle in forbidden if needle in executable]
    return {"pass": not findings, "forbidden_findings": findings}


def architecture_receipt(model: QueryConditionedGatherer) -> dict[str, object]:
    return {
        "config": asdict(model.config),
        "trainable_gatherer_parameters": model.trainable_parameters(),
        "field_queries": FIELD_COUNT,
        "shared_gatherer": True,
        "exact_one_to_one_assignment": True,
        "fieldwise_averaging": False,
        "source_audit": source_audit(),
    }
