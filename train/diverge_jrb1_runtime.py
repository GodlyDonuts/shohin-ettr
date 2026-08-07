#!/usr/bin/env python3
"""Permutation-equivariant joint register binder for DIVERGE-JRB1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from diverge_ats1_data import BYTE_OFFSET, BYTE_VOCAB_SIZE, CLS_ID, PAD_ID
from diverge_eal1_data import scan_integer_spans
from diverge_eal1_runtime import encode_source, module_state_sha256, sha256_path
from diverge_jrb1_data import REGISTERS


SCHEMA = "shohin-diverge-jrb1-runtime-v1"
CHECKPOINT_SCHEMA = "shohin-diverge-jrb1-checkpoint-v1"
MAX_SOURCE_BYTES = 512
MAX_REGISTER_BYTES = 32
MAX_MENTIONS = 4
TEMPORAL_SOURCE_BYTES = 320


class JRB1RuntimeError(RuntimeError):
    """A joint register binder violates its frozen contract."""


@dataclass(frozen=True, slots=True)
class JointRegisterBinderConfig:
    width: int = 128
    layers: int = 2
    max_source_bytes: int = MAX_SOURCE_BYTES
    max_register_bytes: int = MAX_REGISTER_BYTES

    def validate(self) -> None:
        if (
            self.width != 128
            or self.layers != 2
            or self.max_source_bytes != MAX_SOURCE_BYTES
            or self.max_register_bytes != MAX_REGISTER_BYTES
            or self.width % 2
        ):
            raise JRB1RuntimeError("JRB1 model geometry differs")


class JointRegisterBinder(nn.Module):
    """Bind evidence, initial values, and queries to one dynamic register table."""

    def __init__(self, config: JointRegisterBinderConfig | None = None) -> None:
        super().__init__()
        self.config = config or JointRegisterBinderConfig()
        self.config.validate()
        self.embedding = nn.Embedding(BYTE_VOCAB_SIZE, self.config.width)
        self.source_encoder = nn.GRU(
            input_size=self.config.width,
            hidden_size=self.config.width // 2,
            num_layers=self.config.layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.register_encoder = nn.GRU(
            input_size=self.config.width,
            hidden_size=self.config.width // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.source_norm = nn.LayerNorm(self.config.width)
        self.register_norm = nn.LayerNorm(self.config.width)
        self.mention_projection = nn.Linear(self.config.width, self.config.width)
        self.query_projection = nn.Linear(self.config.width, self.config.width)
        self.register_projection = nn.Linear(self.config.width, self.config.width)
        self.logit_scale = nn.Parameter(torch.tensor([1.0]))

    def _encode(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        register_ids: torch.Tensor,
        register_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            source_ids.ndim != 2
            or source_ids.shape != source_mask.shape
            or source_ids.dtype != torch.long
            or source_mask.dtype != torch.bool
            or source_ids.shape[1] > MAX_SOURCE_BYTES
            or register_ids.ndim != 3
            or register_ids.shape[:2] != (source_ids.shape[0], REGISTERS)
            or register_ids.shape != register_mask.shape
            or register_ids.dtype != torch.long
            or register_mask.dtype != torch.bool
            or register_ids.shape[2] > MAX_REGISTER_BYTES
        ):
            raise JRB1RuntimeError("JRB1 forward tensor geometry differs")
        source_lengths = source_mask.sum(dim=1)
        register_lengths = register_mask.flatten(0, 1).sum(dim=1)
        if torch.any(source_lengths < 2) or torch.any(register_lengths < 2):
            raise JRB1RuntimeError("JRB1 input sequence is empty")
        source_packed = pack_padded_sequence(
            self.embedding(source_ids),
            source_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        source_encoded, _ = self.source_encoder(source_packed)
        source_hidden, _ = pad_packed_sequence(
            source_encoded,
            batch_first=True,
            total_length=source_ids.shape[1],
        )
        flat_registers = register_ids.flatten(0, 1)
        register_packed = pack_padded_sequence(
            self.embedding(flat_registers),
            register_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, register_state = self.register_encoder(register_packed)
        register_hidden = torch.cat(
            (register_state[-2], register_state[-1]), dim=-1
        ).view(source_ids.shape[0], REGISTERS, self.config.width)
        return self.source_norm(source_hidden), self.register_norm(register_hidden)

    def _score(
        self, source: torch.Tensor, registers: torch.Tensor, *, query: bool
    ) -> torch.Tensor:
        source_projection = self.query_projection if query else self.mention_projection
        source_key = torch.nn.functional.normalize(source_projection(source), dim=-1)
        register_key = torch.nn.functional.normalize(
            self.register_projection(registers), dim=-1
        )
        logits = torch.einsum("bmw,brw->bmr", source_key, register_key)
        return (logits * self.logit_scale.exp().clamp(max=100.0)).float()

    def forward_mentions(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        register_ids: torch.Tensor,
        register_mask: torch.Tensor,
        mention_bounds: torch.Tensor,
        mention_mask: torch.Tensor,
    ) -> torch.Tensor:
        source_hidden, register_hidden = self._encode(
            source_ids, source_mask, register_ids, register_mask
        )
        if (
            mention_bounds.ndim != 3
            or mention_bounds.shape[:2] != mention_mask.shape
            or mention_bounds.shape[2] != 2
            or mention_bounds.shape[1] > MAX_MENTIONS
            or mention_mask.dtype != torch.bool
        ):
            raise JRB1RuntimeError("JRB1 mention tensor geometry differs")
        positions = torch.arange(source_ids.shape[1], device=source_ids.device).view(
            1, 1, -1
        )
        spans = (positions >= mention_bounds[:, :, 0].unsqueeze(-1)) & (
            positions < mention_bounds[:, :, 1].unsqueeze(-1)
        )
        spans &= mention_mask.unsqueeze(-1)
        if torch.any(spans.sum(dim=-1)[mention_mask] < 1):
            raise JRB1RuntimeError("JRB1 numeric mention is empty")
        pooled = torch.einsum("bms,bsw->bmw", spans.to(source_hidden.dtype), source_hidden)
        pooled = pooled / spans.sum(dim=-1, keepdim=True).clamp(min=1).to(
            source_hidden.dtype
        )
        return self._score(pooled, register_hidden, query=False)

    def forward_query(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        register_ids: torch.Tensor,
        register_mask: torch.Tensor,
    ) -> torch.Tensor:
        source_hidden, register_hidden = self._encode(
            source_ids, source_mask, register_ids, register_mask
        )
        pooled = torch.einsum(
            "bs,bsw->bw", source_mask.to(source_hidden.dtype), source_hidden
        ) / source_mask.sum(dim=1, keepdim=True).to(source_hidden.dtype)
        return self._score(pooled.unsqueeze(1), register_hidden, query=True).squeeze(1)

    def record(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "config": asdict(self.config),
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "state_sha256": module_state_sha256(self),
        }


def _encode_ascii(text: str, maximum: int) -> tuple[int, ...]:
    try:
        encoded = text.encode("ascii")
    except UnicodeEncodeError as error:
        raise JRB1RuntimeError("JRB1 source is not ASCII") from error
    values = (CLS_ID, *(value + BYTE_OFFSET for value in encoded))
    if len(values) > maximum:
        raise JRB1RuntimeError("JRB1 source exceeds frozen width")
    return values


def tensorize_register_sources(
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    text_key: str,
    registers_key: str = "registers",
    mention_count: int | None,
    rotate_register_table: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if not records:
        raise JRB1RuntimeError("JRB1 source batch is empty")
    source_values = [_encode_ascii(str(record[text_key]), MAX_SOURCE_BYTES) for record in records]
    register_values = []
    for record in records:
        registers = tuple(str(value) for value in record[registers_key])
        if len(registers) != REGISTERS or len(set(registers)) != REGISTERS:
            raise JRB1RuntimeError("JRB1 register table differs")
        if rotate_register_table:
            registers = registers[1:] + registers[:1]
        register_values.append(
            tuple(_encode_ascii(value, MAX_REGISTER_BYTES) for value in registers)
        )
    source_width = max(len(value) for value in source_values)
    register_width = max(len(value) for table in register_values for value in table)
    source_ids = torch.full((len(records), source_width), PAD_ID, dtype=torch.long)
    source_mask = torch.zeros_like(source_ids, dtype=torch.bool)
    register_ids = torch.full(
        (len(records), REGISTERS, register_width), PAD_ID, dtype=torch.long
    )
    register_mask = torch.zeros_like(register_ids, dtype=torch.bool)
    bounds = None
    bounds_mask = None
    if mention_count is not None:
        if mention_count < 1 or mention_count > MAX_MENTIONS:
            raise JRB1RuntimeError("JRB1 mention count differs")
        bounds = torch.zeros((len(records), mention_count, 2), dtype=torch.long)
        bounds_mask = torch.ones((len(records), mention_count), dtype=torch.bool)
    for row, values in enumerate(source_values):
        source_ids[row, : len(values)] = torch.tensor(values)
        source_mask[row, : len(values)] = True
        for register_index, register in enumerate(register_values[row]):
            register_ids[row, register_index, : len(register)] = torch.tensor(register)
            register_mask[row, register_index, : len(register)] = True
        if bounds is not None:
            spans = scan_integer_spans(str(records[row][text_key]))
            if len(spans) != mention_count:
                raise JRB1RuntimeError("JRB1 numeric mention count differs")
            bounds[row] = torch.tensor(
                tuple((start + 1, end + 1) for start, end in spans)
            )
    return (
        source_ids.to(device),
        source_mask.to(device),
        register_ids.to(device),
        register_mask.to(device),
        None if bounds is None else bounds.to(device),
        None if bounds_mask is None else bounds_mask.to(device),
    )


def tensorize_temporal_without_register_scan(
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    text_key: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare the frozen EAL2 reader without its old exact register scanner."""
    if not records:
        raise JRB1RuntimeError("JRB1 temporal source batch is empty")
    byte_ids = torch.full(
        (len(records), TEMPORAL_SOURCE_BYTES), PAD_ID, dtype=torch.long
    )
    attention = torch.zeros_like(byte_ids, dtype=torch.bool)
    bounds = torch.zeros((len(records), MAX_MENTIONS, 2), dtype=torch.long)
    for row, record in enumerate(records):
        text = str(record[text_key])
        encoded = encode_source(text)
        spans = scan_integer_spans(text)
        if len(encoded) > TEMPORAL_SOURCE_BYTES or len(spans) != MAX_MENTIONS:
            raise JRB1RuntimeError("JRB1 temporal source geometry differs")
        byte_ids[row, : len(encoded)] = torch.tensor(encoded)
        attention[row, : len(encoded)] = True
        bounds[row] = torch.tensor(
            tuple((start + 1, end + 1) for start, end in spans)
        )
    return byte_ids.to(device), attention.to(device), bounds.to(device)


def load_binder(
    path: Path, expected_sha256: str
) -> tuple[JointRegisterBinder, Mapping[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise JRB1RuntimeError("JRB1 checkpoint file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise JRB1RuntimeError("JRB1 checkpoint schema differs")
    model = JointRegisterBinder(JointRegisterBinderConfig(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    if module_state_sha256(model) != payload["model_state_sha256"]:
        raise JRB1RuntimeError("JRB1 model state hash differs")
    return model, payload


__all__ = [
    "CHECKPOINT_SCHEMA",
    "JRB1RuntimeError",
    "JointRegisterBinder",
    "JointRegisterBinderConfig",
    "MAX_MENTIONS",
    "MAX_REGISTER_BYTES",
    "MAX_SOURCE_BYTES",
    "TEMPORAL_SOURCE_BYTES",
    "load_binder",
    "tensorize_register_sources",
    "tensorize_temporal_without_register_scan",
]
