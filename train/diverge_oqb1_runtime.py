#!/usr/bin/env python3
"""Episode-local occurrence quotient and semantic register owner for OQB1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from diverge_ats1_data import BYTE_OFFSET, BYTE_VOCAB_SIZE, CLS_ID, PAD_ID
from diverge_eal1_data import scan_integer_spans
from diverge_eal1_runtime import module_state_sha256, sha256_path
from diverge_jrb1_runtime import tensorize_temporal_without_register_scan


SCHEMA = "shohin-diverge-oqb1-runtime-v1"
CHECKPOINT_SCHEMA = "shohin-diverge-oqb1-checkpoint-v1"
MAX_SOURCE_BYTES = 512
MAX_MENTIONS = 4
REGISTERS = 2
SLOT_MARKERS = ("zzslotaz", "zzslotbz")
QuotientMode = Literal["coherent", "broken"]


class OQB1RuntimeError(RuntimeError):
    """An occurrence-quotient register bus violates its frozen contract."""


@dataclass(frozen=True, slots=True)
class OccurrenceQuotientBinderConfig:
    width: int = 128
    layers: int = 2
    max_source_bytes: int = MAX_SOURCE_BYTES

    def validate(self) -> None:
        if (
            self.width != 128
            or self.layers != 2
            or self.max_source_bytes != MAX_SOURCE_BYTES
            or self.width % 2
        ):
            raise OQB1RuntimeError("OQB1 model geometry differs")


def _broken_slot(text: str, start: int, end: int, salt: str) -> int:
    digest = hashlib.sha256(
        f"oqb1-broken|{salt}|{start}|{end}|{text}".encode("ascii")
    ).digest()
    return digest[0] % REGISTERS


def exact_occurrence_quotient(
    text: str,
    register_table: Sequence[str],
    *,
    mode: QuotientMode = "coherent",
    salt: str = "",
) -> tuple[str, tuple[int, int], tuple[int, int]]:
    """Replace exact declared-name occurrences by anonymous table-position marks."""
    table = tuple(str(value) for value in register_table)
    if (
        mode not in ("coherent", "broken")
        or len(table) != REGISTERS
        or len(set(table)) != REGISTERS
        or any(not value.isalpha() or not value.islower() for value in table)
        or any(marker in text for marker in SLOT_MARKERS)
    ):
        raise OQB1RuntimeError("OQB1 quotient input differs")
    occurrences: list[tuple[int, int, int]] = []
    for position, register in enumerate(table):
        occurrences.extend(
            (match.start(), match.end(), position)
            for match in re.finditer(rf"(?<![a-z]){re.escape(register)}(?![a-z])", text)
        )
    occurrences.sort()
    if any(left[1] > right[0] for left, right in zip(occurrences, occurrences[1:])):
        raise OQB1RuntimeError("OQB1 quotient occurrences overlap")
    output = text
    assigned = [0, 0]
    found = [0, 0]
    for start, end, position in reversed(occurrences):
        slot = position if mode == "coherent" else _broken_slot(text, start, end, salt)
        output = output[:start] + SLOT_MARKERS[slot] + output[end:]
        assigned[slot] += 1
        found[position] += 1
    return output, tuple(found), tuple(assigned)


class OccurrenceQuotientRegisterBinder(nn.Module):
    """Learn semantic attachment after exact identity has been quotiented."""

    def __init__(self, config: OccurrenceQuotientBinderConfig | None = None) -> None:
        super().__init__()
        self.config = config or OccurrenceQuotientBinderConfig()
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
        self.mention_head = nn.Sequential(
            nn.LayerNorm(self.config.width),
            nn.Linear(self.config.width, self.config.width),
            nn.GELU(),
            nn.Linear(self.config.width, REGISTERS),
        )
        self.query_head = nn.Sequential(
            nn.LayerNorm(self.config.width),
            nn.Linear(self.config.width, self.config.width),
            nn.GELU(),
            nn.Linear(self.config.width, REGISTERS),
        )
        self.query_gate = nn.Linear(self.config.width, 1)

    def _encode(
        self, source_ids: torch.Tensor, source_mask: torch.Tensor
    ) -> torch.Tensor:
        if (
            source_ids.ndim != 2
            or source_ids.shape != source_mask.shape
            or source_ids.dtype != torch.long
            or source_mask.dtype != torch.bool
            or source_ids.shape[1] > self.config.max_source_bytes
        ):
            raise OQB1RuntimeError("OQB1 source tensor geometry differs")
        lengths = source_mask.sum(dim=1)
        if torch.any(lengths < 2) or not torch.all(source_ids[:, 0].eq(CLS_ID)):
            raise OQB1RuntimeError("OQB1 source mask differs")
        packed = pack_padded_sequence(
            self.embedding(source_ids),
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(
            encoded, batch_first=True, total_length=source_ids.shape[1]
        )
        return self.output_norm(hidden)

    def forward_mentions(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        mention_bounds: torch.Tensor,
        mention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self._encode(source_ids, source_mask)
        if (
            mention_bounds.ndim != 3
            or mention_bounds.shape[:2] != mention_mask.shape
            or mention_bounds.shape[2] != 2
            or mention_bounds.shape[1] > MAX_MENTIONS
            or mention_mask.dtype != torch.bool
        ):
            raise OQB1RuntimeError("OQB1 mention tensor geometry differs")
        positions = torch.arange(source_ids.shape[1], device=source_ids.device).view(
            1, 1, -1
        )
        spans = (positions >= mention_bounds[:, :, 0].unsqueeze(-1)) & (
            positions < mention_bounds[:, :, 1].unsqueeze(-1)
        )
        spans &= mention_mask.unsqueeze(-1)
        if torch.any(spans.sum(dim=-1)[mention_mask] < 1):
            raise OQB1RuntimeError("OQB1 numeric mention is empty")
        pooled = torch.einsum("bms,bsw->bmw", spans.to(hidden.dtype), hidden)
        pooled /= spans.sum(dim=-1, keepdim=True).clamp(min=1).to(hidden.dtype)
        return self.mention_head(pooled).float()

    def forward_query(
        self, source_ids: torch.Tensor, source_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = self._encode(source_ids, source_mask)
        evidence = self.query_head(hidden) + self.query_gate(hidden)
        evidence = evidence.masked_fill(~source_mask.unsqueeze(-1), float("-inf"))
        return torch.logsumexp(evidence.float(), dim=1)

    def record(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "config": asdict(self.config),
            "parameter_count": sum(
                parameter.numel() for parameter in self.parameters()
            ),
            "state_sha256": module_state_sha256(self),
        }


def _encode_ascii(text: str) -> tuple[int, ...]:
    try:
        encoded = text.encode("ascii")
    except UnicodeEncodeError as error:
        raise OQB1RuntimeError("OQB1 source is not ASCII") from error
    values = (CLS_ID, *(value + BYTE_OFFSET for value in encoded))
    if len(values) > MAX_SOURCE_BYTES:
        raise OQB1RuntimeError("OQB1 source exceeds frozen width")
    return values


def tensorize_quotient_sources(
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    text_key: str,
    table_key: str = "registers",
    mention_count: int | None,
    mode: QuotientMode = "coherent",
    reverse_table: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
]:
    if not records:
        raise OQB1RuntimeError("OQB1 source batch is empty")
    texts = []
    valid = []
    for index, record in enumerate(records):
        table = tuple(str(value) for value in record[table_key])
        if reverse_table:
            table = (table[1], table[0])
        quotient, found, _ = exact_occurrence_quotient(
            str(record[text_key]),
            table,
            mode=mode,
            salt=f"{record.get('serial', index)}|{record.get('salt', '')}",
        )
        texts.append(quotient)
        valid.append(any(found) if mention_count is None else all(found))
    encoded = [_encode_ascii(text) for text in texts]
    width = max(len(value) for value in encoded)
    source_ids = torch.full((len(records), width), PAD_ID, dtype=torch.long)
    source_mask = torch.zeros_like(source_ids, dtype=torch.bool)
    bounds = None
    bounds_mask = None
    if mention_count is not None:
        if mention_count < 1 or mention_count > MAX_MENTIONS:
            raise OQB1RuntimeError("OQB1 mention count differs")
        bounds = torch.zeros((len(records), mention_count, 2), dtype=torch.long)
        bounds_mask = torch.ones((len(records), mention_count), dtype=torch.bool)
    for row, values in enumerate(encoded):
        source_ids[row, : len(values)] = torch.tensor(values)
        source_mask[row, : len(values)] = True
        if bounds is not None:
            spans = scan_integer_spans(texts[row])
            if len(spans) != mention_count:
                raise OQB1RuntimeError("OQB1 numeric mention count differs")
            bounds[row] = torch.tensor(
                tuple((start + 1, end + 1) for start, end in spans)
            )
    return (
        source_ids.to(device),
        source_mask.to(device),
        None if bounds is None else bounds.to(device),
        None if bounds_mask is None else bounds_mask.to(device),
        torch.tensor(valid, dtype=torch.bool, device=device),
    )


def load_binder(
    path: Path, expected_sha256: str
) -> tuple[OccurrenceQuotientRegisterBinder, Mapping[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise OQB1RuntimeError("OQB1 checkpoint file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise OQB1RuntimeError("OQB1 checkpoint schema differs")
    model = OccurrenceQuotientRegisterBinder(
        OccurrenceQuotientBinderConfig(**payload["config"])
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    if module_state_sha256(model) != payload["model_state_sha256"]:
        raise OQB1RuntimeError("OQB1 model state hash differs")
    return model, payload


__all__ = [
    "CHECKPOINT_SCHEMA",
    "OccurrenceQuotientBinderConfig",
    "OccurrenceQuotientRegisterBinder",
    "OQB1RuntimeError",
    "SLOT_MARKERS",
    "exact_occurrence_quotient",
    "load_binder",
    "tensorize_quotient_sources",
    "tensorize_temporal_without_register_scan",
]
