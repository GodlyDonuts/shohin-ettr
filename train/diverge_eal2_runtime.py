#!/usr/bin/env python3
"""Identifiable temporal reader and local-law adapter for DIVERGE-EAL2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from diverge_ats1_data import BYTE_VOCAB_SIZE, CLS_ID, PAD_ID
from diverge_eal1_runtime import (
    CHECKPOINT_SCHEMA as EAL1_CHECKPOINT_SCHEMA,
    EAL1RuntimeError,
    LawCompilation,
    compile_episode_laws as compile_complete_roles,
    encode_source,
    module_state_sha256,
    scan_integer_spans,
    sha256_path,
)


SCHEMA = "shohin-diverge-eal2-runtime-v1"
CHECKPOINT_SCHEMA = "shohin-diverge-eal2-checkpoint-v1"
MAX_SOURCE_BYTES = 320
MENTIONS = 4
TEMPORAL_ROLES = 2


class EAL2RuntimeError(RuntimeError):
    """An EAL2 temporal interface violates its frozen contract."""


@dataclass(frozen=True, slots=True)
class TemporalReaderConfig:
    width: int = 192
    layers: int = 2
    max_bytes: int = MAX_SOURCE_BYTES

    def validate(self) -> None:
        if (
            self.width != 192
            or self.layers != 2
            or self.width % 2
            or self.max_bytes != MAX_SOURCE_BYTES
        ):
            raise EAL2RuntimeError("EAL2 reader geometry differs")


class NaturalTemporalReader(nn.Module):
    """Own only the observable BEFORE versus AFTER semantic coordinate."""

    def __init__(self, config: TemporalReaderConfig | None = None) -> None:
        super().__init__()
        self.config = config or TemporalReaderConfig()
        self.config.validate()
        self.embedding = nn.Embedding(BYTE_VOCAB_SIZE, self.config.width)
        self.encoder = nn.GRU(
            input_size=self.config.width,
            hidden_size=self.config.width // 2,
            num_layers=self.config.layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.output_norm = nn.LayerNorm(self.config.width)
        self.temporal_head = nn.Sequential(
            nn.LayerNorm(self.config.width),
            nn.Linear(self.config.width, self.config.width),
            nn.GELU(),
            nn.Linear(self.config.width, TEMPORAL_ROLES),
        )

    def forward(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric_bounds: torch.Tensor,
    ) -> torch.Tensor:
        if (
            byte_ids.ndim != 2
            or byte_ids.shape != attention_mask.shape
            or byte_ids.shape[1] != self.config.max_bytes
            or byte_ids.dtype != torch.long
            or numeric_bounds.shape != (byte_ids.shape[0], MENTIONS, 2)
        ):
            raise EAL2RuntimeError("EAL2 reader tensor interface differs")
        lengths = attention_mask.bool().sum(dim=1)
        if torch.any(lengths < 2) or not torch.all(byte_ids[:, 0].eq(CLS_ID)):
            raise EAL2RuntimeError("EAL2 source mask or CLS differs")
        packed = pack_padded_sequence(
            self.embedding(byte_ids),
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(
            encoded,
            batch_first=True,
            total_length=self.config.max_bytes,
        )
        hidden = self.output_norm(hidden)
        positions = torch.arange(self.config.max_bytes, device=byte_ids.device).view(
            1, 1, -1
        )
        mention_mask = (positions >= numeric_bounds[:, :, 0].unsqueeze(-1)) & (
            positions < numeric_bounds[:, :, 1].unsqueeze(-1)
        )
        if torch.any(mention_mask.sum(dim=-1) < 1):
            raise EAL2RuntimeError("EAL2 numeric mention is empty")
        mention_hidden = torch.einsum(
            "bms,bsw->bmw", mention_mask.to(hidden.dtype), hidden
        ) / mention_mask.sum(dim=-1, keepdim=True).to(hidden.dtype)
        return self.temporal_head(mention_hidden).float()

    def record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "config": asdict(self.config),
            "parameter_count": sum(
                parameter.numel() for parameter in self.parameters()
            ),
            "state_sha256": module_state_sha256(self),
        }


def scan_register_ids(text: str, registers: Sequence[str]) -> tuple[int, int, int, int]:
    table = tuple(str(value) for value in registers)
    if len(table) != 2 or len(set(table)) != 2:
        raise EAL2RuntimeError("EAL2 register table differs")
    occurrences = []
    for register_index, register in enumerate(table):
        if not register.isalpha() or not register.islower():
            raise EAL2RuntimeError("EAL2 register carrier differs")
        occurrences.extend(
            (match.start(), match.end(), register_index)
            for match in re.finditer(rf"(?<![a-z]){re.escape(register)}(?![a-z])", text)
        )
    spans = scan_integer_spans(text)
    if len(spans) != MENTIONS:
        raise EAL2RuntimeError("EAL2 source does not expose four integers")
    output = []
    previous_end = 0
    for start, end in spans:
        candidates = [
            occurrence
            for occurrence in occurrences
            if previous_end <= occurrence[0] and occurrence[1] <= start
        ]
        if not candidates:
            raise EAL2RuntimeError("EAL2 mention has no local register owner")
        output.append(max(candidates, key=lambda value: value[1])[2])
        previous_end = end
    if output.count(0) != 2 or output.count(1) != 2:
        raise EAL2RuntimeError("EAL2 register mention geometry differs")
    return tuple(output)  # type: ignore[return-value]


def tensorize_temporal_sources(
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    text_key: str = "source_text",
    role_key: str | None = "numeric_role_ids",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(records)
    byte_ids = torch.full((batch, MAX_SOURCE_BYTES), PAD_ID, dtype=torch.long)
    attention = torch.zeros((batch, MAX_SOURCE_BYTES), dtype=torch.bool)
    numeric_bounds = torch.zeros((batch, MENTIONS, 2), dtype=torch.long)
    register_ids = torch.zeros((batch, MENTIONS), dtype=torch.long)
    targets = torch.zeros((batch, MENTIONS), dtype=torch.long)
    for row_index, record in enumerate(records):
        text = str(record[text_key])
        encoded = encode_source(text)
        byte_ids[row_index, : len(encoded)] = torch.tensor(encoded)
        attention[row_index, : len(encoded)] = True
        spans = scan_integer_spans(text)
        for mention_index, (start, end) in enumerate(spans):
            numeric_bounds[row_index, mention_index] = torch.tensor(
                (start + 1, end + 1)
            )
        register_ids[row_index] = torch.tensor(
            scan_register_ids(text, record["registers"])
        )
        if role_key is not None:
            roles = tuple(int(value) for value in record[role_key])
            if sorted(roles) != list(range(4)):
                raise EAL2RuntimeError("EAL2 complete role target differs")
            targets[row_index] = torch.tensor(tuple(value // 2 for value in roles))
    return (
        byte_ids.to(device),
        attention.to(device),
        numeric_bounds.to(device),
        register_ids.to(device),
        targets.to(device),
    )


def hard_temporal_assignment(
    logits: torch.Tensor,
    register_ids: Sequence[int] | torch.Tensor,
) -> tuple[int, int, int, int]:
    if logits.shape != (MENTIONS, TEMPORAL_ROLES):
        raise EAL2RuntimeError("EAL2 temporal logits differ")
    register_tuple = tuple(int(value) for value in register_ids)
    output = [-1] * MENTIONS
    for register in range(2):
        indices = [
            index for index, value in enumerate(register_tuple) if value == register
        ]
        if len(indices) != 2:
            raise EAL2RuntimeError("EAL2 register group differs")
        left, right = indices
        normal = float(logits[left, 0] + logits[right, 1])
        swapped = float(logits[left, 1] + logits[right, 0])
        if normal >= swapped:
            output[left], output[right] = 0, 1
        else:
            output[left], output[right] = 1, 0
    return tuple(output)  # type: ignore[return-value]


def compose_complete_roles(
    records: Sequence[Mapping[str, Any]],
    temporal_assignments: Sequence[Sequence[int]],
    *,
    text_key: str,
) -> list[tuple[int, int, int, int]]:
    if len(records) != len(temporal_assignments):
        raise EAL2RuntimeError("EAL2 temporal/evidence count differs")
    output = []
    for record, temporal in zip(records, temporal_assignments, strict=True):
        registers = scan_register_ids(str(record[text_key]), record["registers"])
        values = tuple(int(value) for value in temporal)
        if any(value not in (0, 1) for value in values):
            raise EAL2RuntimeError("EAL2 temporal role leaves its carrier")
        roles = tuple(
            temporal_role * 2 + register
            for temporal_role, register in zip(values, registers, strict=True)
        )
        if sorted(roles) != list(range(4)):
            raise EAL2RuntimeError("EAL2 composed roles are not complete")
        output.append(roles)  # type: ignore[arg-type]
    return output


def compile_episode_laws(
    public: Mapping[str, Any],
    temporal_assignments: Sequence[Sequence[int]],
    *,
    reader_state_sha256: str,
    text_key: str = "source_text",
    evidence_limit_per_operation: int | None = None,
) -> LawCompilation:
    evidence = tuple(public["evidence"])
    complete = compose_complete_roles(evidence, temporal_assignments, text_key=text_key)
    try:
        return compile_complete_roles(
            public,
            complete,
            reader_state_sha256=reader_state_sha256,
            text_key=text_key,
            evidence_limit_per_operation=evidence_limit_per_operation,
        )
    except EAL1RuntimeError as error:
        raise EAL2RuntimeError(str(error)) from error


def load_reader(
    path: Path,
    expected_sha256: str,
) -> tuple[NaturalTemporalReader, Mapping[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise EAL2RuntimeError("EAL2 checkpoint file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        if payload.get("schema") == EAL1_CHECKPOINT_SCHEMA:
            raise EAL2RuntimeError("refusing incompatible EAL1 checkpoint")
        raise EAL2RuntimeError("EAL2 checkpoint schema differs")
    model = NaturalTemporalReader(TemporalReaderConfig(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    if module_state_sha256(model) != payload["model_state_sha256"]:
        raise EAL2RuntimeError("EAL2 reader state hash differs")
    return model, payload


__all__ = [
    "CHECKPOINT_SCHEMA",
    "EAL2RuntimeError",
    "NaturalTemporalReader",
    "TemporalReaderConfig",
    "compile_episode_laws",
    "compose_complete_roles",
    "hard_temporal_assignment",
    "load_reader",
    "scan_register_ids",
    "tensorize_temporal_sources",
]
