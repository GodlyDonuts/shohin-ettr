#!/usr/bin/env python3
"""Learned episode-local evidence-operation binding for DIVERGE-OPB1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from diverge_ats1_data import BYTE_OFFSET, BYTE_VOCAB_SIZE, CLS_ID, PAD_ID
from diverge_eal1_runtime import (
    EpisodeLawPacket,
    LawCompilation,
    canonical_sha256,
    module_state_sha256,
    sha256_path,
)
from diverge_nls1_runtime import NeuralLawSynthesizer, hard_rows
from diverge_sve1_runtime import decode_evidence_events


SCHEMA = "shohin-diverge-opb1-runtime-v1"
CHECKPOINT_SCHEMA = "shohin-diverge-opb1-checkpoint-v1"
OPERATIONS = 8
DEMONSTRATIONS = 3
MAX_SOURCE_BYTES = 512
MAX_ALIAS_BYTES = 32


class OPB1RuntimeError(RuntimeError):
    """An evidence-operation pointer violates its frozen contract."""


@dataclass(frozen=True, slots=True)
class EvidenceOperationPointerConfig:
    width: int = 128
    layers: int = 2
    max_source_bytes: int = MAX_SOURCE_BYTES
    max_alias_bytes: int = MAX_ALIAS_BYTES

    def validate(self) -> None:
        if (
            self.width != 128
            or self.layers != 2
            or self.max_source_bytes != MAX_SOURCE_BYTES
            or self.max_alias_bytes != MAX_ALIAS_BYTES
            or self.width % 2
        ):
            raise OPB1RuntimeError("OPB1 model geometry differs")


class EvidenceOperationPointer(nn.Module):
    """Select one alias-table entry from a complete raw evidence statement."""

    def __init__(self, config: EvidenceOperationPointerConfig | None = None) -> None:
        super().__init__()
        self.config = config or EvidenceOperationPointerConfig()
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
        self.alias_encoder = nn.GRU(
            input_size=self.config.width,
            hidden_size=self.config.width // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.source_norm = nn.LayerNorm(self.config.width)
        self.alias_norm = nn.LayerNorm(self.config.width)
        self.source_projection = nn.Linear(self.config.width, self.config.width)
        self.alias_projection = nn.Linear(self.config.width, self.config.width)
        self.token_gate = nn.Linear(self.config.width, 1)
        self.logit_scale = nn.Parameter(torch.tensor([1.0]))

    def forward(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        alias_ids: torch.Tensor,
        alias_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            source_ids.ndim != 2
            or source_ids.shape != source_mask.shape
            or source_ids.dtype != torch.long
            or source_mask.dtype != torch.bool
            or source_ids.shape[1] > MAX_SOURCE_BYTES
            or alias_ids.ndim != 3
            or alias_ids.shape[:2] != (source_ids.shape[0], OPERATIONS)
            or alias_ids.shape != alias_mask.shape
            or alias_ids.dtype != torch.long
            or alias_mask.dtype != torch.bool
            or alias_ids.shape[2] > MAX_ALIAS_BYTES
        ):
            raise OPB1RuntimeError("OPB1 forward tensor geometry differs")
        source_lengths = source_mask.sum(dim=1)
        alias_lengths = alias_mask.flatten(0, 1).sum(dim=1)
        if (
            torch.any(source_lengths < 2)
            or torch.any(alias_lengths < 2)
            or not torch.all(source_ids[:, 0].eq(CLS_ID))
        ):
            raise OPB1RuntimeError("OPB1 input sequence is empty")

        source_packed = pack_padded_sequence(
            self.embedding(source_ids),
            source_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.source_encoder(source_packed)
        source_hidden, _ = pad_packed_sequence(
            encoded, batch_first=True, total_length=source_ids.shape[1]
        )
        flat_aliases = alias_ids.flatten(0, 1)
        alias_packed = pack_padded_sequence(
            self.embedding(flat_aliases),
            alias_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, alias_state = self.alias_encoder(alias_packed)
        alias_hidden = torch.cat((alias_state[-2], alias_state[-1]), dim=-1).view(
            source_ids.shape[0], OPERATIONS, self.config.width
        )

        source_keys = torch.nn.functional.normalize(
            self.source_projection(self.source_norm(source_hidden)), dim=-1
        )
        alias_keys = torch.nn.functional.normalize(
            self.alias_projection(self.alias_norm(alias_hidden)), dim=-1
        )
        evidence = torch.einsum("bsw,bow->bso", source_keys, alias_keys)
        evidence = evidence * self.logit_scale.exp().clamp(max=100.0)
        evidence = evidence + self.token_gate(source_hidden)
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


def _encode_ascii(text: str, maximum: int) -> tuple[int, ...]:
    try:
        encoded = text.encode("ascii")
    except UnicodeEncodeError as error:
        raise OPB1RuntimeError("OPB1 source is not ASCII") from error
    values = (CLS_ID, *(value + BYTE_OFFSET for value in encoded))
    if len(values) > maximum:
        raise OPB1RuntimeError("OPB1 source exceeds frozen width")
    return values


def tensorize_operation_sources(
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    text_key: str = "source_text",
    aliases_key: str = "aliases",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not records:
        raise OPB1RuntimeError("OPB1 source batch is empty")
    sources = [
        _encode_ascii(str(record[text_key]), MAX_SOURCE_BYTES) for record in records
    ]
    tables = []
    for record in records:
        aliases = tuple(str(value) for value in record[aliases_key])
        if (
            len(aliases) != OPERATIONS
            or len(set(aliases)) != OPERATIONS
            or any(not value.isalpha() or not value.islower() for value in aliases)
        ):
            raise OPB1RuntimeError("OPB1 alias table differs")
        tables.append(tuple(_encode_ascii(value, MAX_ALIAS_BYTES) for value in aliases))
    source_width = max(len(value) for value in sources)
    alias_width = max(len(alias) for table in tables for alias in table)
    source_ids = torch.full((len(records), source_width), PAD_ID, dtype=torch.long)
    source_mask = torch.zeros_like(source_ids, dtype=torch.bool)
    alias_ids = torch.full(
        (len(records), OPERATIONS, alias_width), PAD_ID, dtype=torch.long
    )
    alias_mask = torch.zeros_like(alias_ids, dtype=torch.bool)
    for row, values in enumerate(sources):
        source_ids[row, : len(values)] = torch.tensor(values)
        source_mask[row, : len(values)] = True
        for alias_index, alias in enumerate(tables[row]):
            alias_ids[row, alias_index, : len(alias)] = torch.tensor(alias)
            alias_mask[row, alias_index, : len(alias)] = True
    return (
        source_ids.to(device),
        source_mask.to(device),
        alias_ids.to(device),
        alias_mask.to(device),
    )


@torch.no_grad()
def compile_pointer_event_laws(
    aliases: Sequence[str],
    evidence_commitments: Sequence[str],
    event_sequences: Sequence[Sequence[int]],
    operation_indices: Sequence[int],
    model: NeuralLawSynthesizer,
    *,
    device: torch.device,
    event_owner_sha256: str,
    pointer_owner_sha256: str,
    law_owner_sha256: str,
) -> LawCompilation:
    """Compile laws using only pointer-selected groups and complete value events."""
    table = tuple(str(value) for value in aliases)
    if (
        len(table) != OPERATIONS
        or len(set(table)) != OPERATIONS
        or len(evidence_commitments) != OPERATIONS * DEMONSTRATIONS
        or len(event_sequences) != len(evidence_commitments)
        or len(operation_indices) != len(evidence_commitments)
    ):
        raise OPB1RuntimeError("OPB1 episode geometry differs")
    grouped: list[list[tuple[int, int, int, int]]] = [[] for _ in range(OPERATIONS)]
    accepted_commitments = []
    for commitment, sequence, operation in zip(
        evidence_commitments, event_sequences, operation_indices, strict=True
    ):
        operation = int(operation)
        if operation not in range(OPERATIONS):
            return LawCompilation(None, "operation_not_complete", tuple(), 0)
        try:
            decoded = decode_evidence_events(sequence)
        except RuntimeError:
            return LawCompilation(
                None, "event_not_complete", tuple(), len(accepted_commitments)
            )
        by_role = {int(role): int(value) for role, value in decoded}
        if sorted(by_role) != [0, 1, 2, 3]:
            return LawCompilation(
                None, "event_not_complete", tuple(), len(accepted_commitments)
            )
        grouped[operation].append((by_role[0], by_role[1], by_role[2], by_role[3]))
        accepted_commitments.append(str(commitment))
    if any(len(value) != DEMONSTRATIONS for value in grouped):
        return LawCompilation(
            None, "operation_not_complete", tuple(), len(accepted_commitments)
        )

    values = torch.tensor(grouped, dtype=torch.long, device=device)
    mask = torch.ones((OPERATIONS, DEMONSTRATIONS), dtype=torch.bool, device=device)
    rows = tuple(hard_rows(logits) for logits in model(values, mask))
    owner_hash = canonical_sha256(
        [
            "opb1-owner",
            event_owner_sha256,
            pointer_owner_sha256,
            law_owner_sha256,
        ]
    )
    provisional = EpisodeLawPacket(
        aliases=table,
        rows=rows,
        evidence_commitments=tuple(accepted_commitments),
        reader_state_sha256=owner_hash,
        commitment="",
    )
    packet = replace(provisional, commitment=canonical_sha256(provisional.payload()))
    return LawCompilation(
        packet=packet,
        error=None,
        support_sizes=tuple((1, 1) for _ in range(OPERATIONS)),
        evidence_count=len(accepted_commitments),
    )


def load_operation_pointer(
    path: Path, expected_sha256: str
) -> tuple[EvidenceOperationPointer, Mapping[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise OPB1RuntimeError("OPB1 checkpoint file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise OPB1RuntimeError("OPB1 checkpoint schema differs")
    model = EvidenceOperationPointer(
        EvidenceOperationPointerConfig(**payload["config"])
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    if module_state_sha256(model) != payload["model_state_sha256"]:
        raise OPB1RuntimeError("OPB1 model state hash differs")
    return model, payload


__all__ = [
    "CHECKPOINT_SCHEMA",
    "EvidenceOperationPointer",
    "EvidenceOperationPointerConfig",
    "OPB1RuntimeError",
    "compile_pointer_event_laws",
    "load_operation_pointer",
    "tensorize_operation_sources",
]
