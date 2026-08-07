#!/usr/bin/env python3
"""Spanless value-event transduction and law compilation for DIVERGE-SVE1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Sequence

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
from diverge_mze1_runtime import PRIME, ROW_CANDIDATES
from diverge_oqb1_runtime import QuotientMode, exact_occurrence_quotient


SCHEMA = "shohin-diverge-sve1-runtime-v1"
CHECKPOINT_SCHEMA = "shohin-diverge-sve1-checkpoint-v1"
MAX_SOURCE_BYTES = 512
EVIDENCE_EVENTS = 4
INITIAL_EVENTS = 2
REGISTERS = 2
TEMPORAL_ROLES = 2
VALUES = PRIME
EVIDENCE_CLASSES = EVIDENCE_EVENTS * VALUES
INITIAL_CLASSES = INITIAL_EVENTS * VALUES
EVIDENCE_BLANK_ID = EVIDENCE_CLASSES
INITIAL_BLANK_ID = INITIAL_CLASSES
EventKind = Literal["evidence", "initial"]


class SVE1RuntimeError(RuntimeError):
    """A spanless value-event runtime invariant was violated."""


@dataclass(frozen=True, slots=True)
class SpanlessValueEventConfig:
    width: int = 192
    layers: int = 2
    max_source_bytes: int = MAX_SOURCE_BYTES

    def validate(self) -> None:
        if (
            self.width != 192
            or self.layers != 2
            or self.max_source_bytes != MAX_SOURCE_BYTES
            or self.width % 2
        ):
            raise SVE1RuntimeError("SVE1 model geometry differs")


class SpanlessValueEventTransducer(nn.Module):
    """Emit complete value-bearing events directly from a byte sequence."""

    def __init__(self, config: SpanlessValueEventConfig | None = None) -> None:
        super().__init__()
        self.config = config or SpanlessValueEventConfig()
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
        self.evidence_head = nn.Linear(self.config.width, EVIDENCE_CLASSES + 1)
        self.initial_head = nn.Linear(self.config.width, INITIAL_CLASSES + 1)

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
            raise SVE1RuntimeError("SVE1 source tensor geometry differs")
        lengths = source_mask.sum(dim=1)
        if torch.any(lengths < 2) or not torch.all(source_ids[:, 0].eq(CLS_ID)):
            raise SVE1RuntimeError("SVE1 source mask differs")
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

    def forward(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        kind: EventKind,
    ) -> torch.Tensor:
        hidden = self._encode(source_ids, source_mask)
        if kind == "evidence":
            return self.evidence_head(hidden).float()
        if kind == "initial":
            return self.initial_head(hidden).float()
        raise SVE1RuntimeError("SVE1 event kind differs")

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
        raise SVE1RuntimeError("SVE1 source is not ASCII") from error
    values = (CLS_ID, *(value + BYTE_OFFSET for value in encoded))
    if len(values) > MAX_SOURCE_BYTES:
        raise SVE1RuntimeError("SVE1 source exceeds frozen width")
    return values


def digit_scrub(text: str) -> str:
    """Delete value evidence without locating or parsing numeric spans."""
    return "".join("x" if "0" <= value <= "9" else value for value in text)


def tensorize_event_sources(
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    text_key: str,
    table_key: str = "register_table",
    expected_occurrences: tuple[int, int],
    quotient_mode: QuotientMode = "coherent",
    reverse_table: bool = False,
    scrub_values: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not records or expected_occurrences not in ((2, 2), (1, 1)):
        raise SVE1RuntimeError("SVE1 source batch geometry differs")
    texts = []
    valid = []
    for index, record in enumerate(records):
        table = tuple(str(value) for value in record[table_key])
        if reverse_table:
            table = (table[1], table[0])
        quotient, found, _ = exact_occurrence_quotient(
            str(record[text_key]),
            table,
            mode=quotient_mode,
            salt=f"sve1|{record.get('serial', index)}|{record.get('salt', '')}",
        )
        if scrub_values:
            quotient = digit_scrub(quotient)
        texts.append(quotient)
        valid.append(found == expected_occurrences)
    encoded = [_encode_ascii(value) for value in texts]
    width = max(len(value) for value in encoded)
    source_ids = torch.full((len(records), width), PAD_ID, dtype=torch.long)
    source_mask = torch.zeros_like(source_ids, dtype=torch.bool)
    for row, values in enumerate(encoded):
        source_ids[row, : len(values)] = torch.tensor(values)
        source_mask[row, : len(values)] = True
    return (
        source_ids.to(device),
        source_mask.to(device),
        source_mask.sum(dim=1).to(device),
        torch.tensor(valid, dtype=torch.bool, device=device),
    )


def greedy_ctc_decode(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    *,
    blank_id: int,
) -> list[tuple[int, ...]]:
    if logits.ndim != 3 or logits.shape[:2] != (
        lengths.shape[0],
        logits.shape[1],
    ):
        raise SVE1RuntimeError("SVE1 CTC logit geometry differs")
    predictions = logits.detach().cpu().argmax(dim=-1)
    output = []
    for row, length in zip(predictions, lengths.detach().cpu(), strict=True):
        collapsed = []
        previous = None
        for item in row[: int(length)]:
            token = int(item)
            if token != blank_id and token != previous:
                collapsed.append(token)
            previous = token
        output.append(tuple(collapsed))
    return output


def decode_evidence_events(events: Sequence[int]) -> tuple[tuple[int, int], ...]:
    values = tuple((int(token) // VALUES, int(token) % VALUES) for token in events)
    if len(values) != EVIDENCE_EVENTS or sorted(role for role, _ in values) != [
        0,
        1,
        2,
        3,
    ]:
        raise SVE1RuntimeError("SVE1 evidence event sequence is incomplete")
    return values


def decode_initial_events(events: Sequence[int]) -> tuple[int, int]:
    values = tuple((int(token) // VALUES, int(token) % VALUES) for token in events)
    if len(values) != INITIAL_EVENTS or sorted(slot for slot, _ in values) != [0, 1]:
        raise SVE1RuntimeError("SVE1 initial event sequence is incomplete")
    state = [0, 0]
    for slot, value in values:
        state[slot] = value
    return state[0], state[1]


def _operation_index(text: str, aliases: Sequence[str]) -> int:
    present = []
    for index, alias in enumerate(aliases):
        if not alias.isalpha() or not alias.islower():
            raise SVE1RuntimeError("SVE1 alias carrier differs")
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", text):
            present.append(index)
    if len(present) != 1:
        raise SVE1RuntimeError("SVE1 evidence does not bind exactly one alias")
    return present[0]


def compile_event_laws(
    public: Mapping[str, Any],
    event_sequences: Sequence[Sequence[int]],
    *,
    owner_state_sha256: str,
    text_key: str,
    hash_key: str,
) -> LawCompilation:
    """Compile exact bounded laws from model-emitted events, never raw digits."""
    aliases = tuple(str(value) for value in public["aliases"])
    evidence = tuple(public["evidence"])
    if len(aliases) != 8 or len(set(aliases)) != 8:
        raise SVE1RuntimeError("SVE1 episode alias table differs")
    if len(event_sequences) != len(evidence):
        raise SVE1RuntimeError("SVE1 event/evidence count differs")
    supports = [[set(range(len(ROW_CANDIDATES))) for _ in range(2)] for _ in range(8)]
    commitments = []
    consumed = 0
    for record, sequence in zip(evidence, event_sequences, strict=True):
        try:
            events = decode_evidence_events(sequence)
        except SVE1RuntimeError:
            return LawCompilation(None, "event_not_complete", tuple(), consumed)
        operation = _operation_index(str(record[text_key]), aliases)
        by_role = {role: value for role, value in events}
        before = (by_role[0], by_role[1])
        after = (by_role[2], by_role[3])
        for output in range(2):
            supports[operation][output] = {
                index
                for index in supports[operation][output]
                if (
                    ROW_CANDIDATES[index][0] * before[0]
                    + ROW_CANDIDATES[index][1] * before[1]
                )
                % PRIME
                == after[output]
            }
            if not supports[operation][output]:
                sizes = tuple(tuple(len(value) for value in item) for item in supports)
                return LawCompilation(None, "empty_support", sizes, consumed + 1)
        commitments.append(str(record[hash_key]))
        consumed += 1
    sizes = tuple(tuple(len(value) for value in item) for item in supports)
    if any(size != 1 for item in sizes for size in item):
        return LawCompilation(None, "underdetermined", sizes, consumed)
    rows = tuple(
        tuple(ROW_CANDIDATES[next(iter(support))] for support in item)
        for item in supports
    )
    provisional = EpisodeLawPacket(
        aliases=aliases,
        rows=rows,  # type: ignore[arg-type]
        evidence_commitments=tuple(commitments),
        reader_state_sha256=owner_state_sha256,
        commitment="",
    )
    packet = replace(provisional, commitment=canonical_sha256(provisional.payload()))
    return LawCompilation(packet, None, sizes, consumed)


def load_transducer(
    path: Path, expected_sha256: str
) -> tuple[SpanlessValueEventTransducer, Mapping[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise SVE1RuntimeError("SVE1 checkpoint file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise SVE1RuntimeError("SVE1 checkpoint schema differs")
    model = SpanlessValueEventTransducer(SpanlessValueEventConfig(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    if module_state_sha256(model) != payload.get("model_state_sha256"):
        raise SVE1RuntimeError("SVE1 checkpoint state differs")
    return model, payload


__all__ = [
    "CHECKPOINT_SCHEMA",
    "EVIDENCE_BLANK_ID",
    "EVIDENCE_CLASSES",
    "INITIAL_BLANK_ID",
    "INITIAL_CLASSES",
    "SVE1RuntimeError",
    "SpanlessValueEventConfig",
    "SpanlessValueEventTransducer",
    "compile_event_laws",
    "decode_evidence_events",
    "decode_initial_events",
    "digit_scrub",
    "greedy_ctc_decode",
    "load_transducer",
    "tensorize_event_sources",
]
