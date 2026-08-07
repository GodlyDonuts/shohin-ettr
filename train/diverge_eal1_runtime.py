#!/usr/bin/env python3
"""Natural transition reader and episode-local law state for DIVERGE-EAL1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import itertools
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from diverge_ats1_data import BYTE_OFFSET, BYTE_VOCAB_SIZE, CLS_ID, PAD_ID
from diverge_mze1_runtime import PRIME, ROW_CANDIDATES


SCHEMA = "shohin-diverge-eal1-runtime-v1"
CHECKPOINT_SCHEMA = "shohin-diverge-eal1-checkpoint-v1"
MAX_SOURCE_BYTES = 320
MENTIONS = 4
ROLES = 4
OPERATIONS = 8
OUTPUTS = 2
_INTEGER = re.compile(r"(?<![A-Za-z0-9])(?:0|[1-9][0-9]?)(?![A-Za-z0-9])")
_PERMUTATIONS = tuple(itertools.permutations(range(ROLES)))


class EAL1RuntimeError(RuntimeError):
    """An EAL1 compiler, packet, or execution violates its contract."""


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("ascii"))
        digest.update(
            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def scan_integer_spans(text: str) -> tuple[tuple[int, int], ...]:
    try:
        text.encode("ascii")
    except UnicodeEncodeError as error:
        raise EAL1RuntimeError("EAL1 source is not ASCII") from error
    return tuple(match.span() for match in _INTEGER.finditer(text))


def encode_source(text: str) -> tuple[int, ...]:
    raw = text.encode("ascii")
    if not raw or len(raw) + 1 > MAX_SOURCE_BYTES:
        raise EAL1RuntimeError("EAL1 source width differs")
    return (CLS_ID, *(value + BYTE_OFFSET for value in raw))


@dataclass(frozen=True, slots=True)
class TransitionReaderConfig:
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
            raise EAL1RuntimeError("EAL1 reader geometry differs")


class NaturalTransitionReader(nn.Module):
    """Assign source numeric mentions to before/after register roles."""

    def __init__(self, config: TransitionReaderConfig | None = None) -> None:
        super().__init__()
        self.config = config or TransitionReaderConfig()
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
        self.role_head = nn.Sequential(
            nn.LayerNorm(self.config.width),
            nn.Linear(self.config.width, self.config.width),
            nn.GELU(),
            nn.Linear(self.config.width, ROLES),
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
            raise EAL1RuntimeError("EAL1 reader tensor interface differs")
        lengths = attention_mask.bool().sum(dim=1)
        if torch.any(lengths < 2) or not torch.all(byte_ids[:, 0].eq(CLS_ID)):
            raise EAL1RuntimeError("EAL1 source mask or CLS differs")
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
            raise EAL1RuntimeError("EAL1 numeric mention is empty")
        mention_hidden = torch.einsum(
            "bms,bsw->bmw", mention_mask.to(hidden.dtype), hidden
        ) / mention_mask.sum(dim=-1, keepdim=True).to(hidden.dtype)
        return self.role_head(mention_hidden).float()

    def record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "config": asdict(self.config),
            "parameter_count": sum(
                parameter.numel() for parameter in self.parameters()
            ),
            "state_sha256": module_state_sha256(self),
        }


def tensorize_sources(
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    text_key: str = "source_text",
    role_key: str | None = "numeric_role_ids",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(records)
    byte_ids = torch.full((batch, MAX_SOURCE_BYTES), PAD_ID, dtype=torch.long)
    attention = torch.zeros((batch, MAX_SOURCE_BYTES), dtype=torch.bool)
    numeric_bounds = torch.zeros((batch, MENTIONS, 2), dtype=torch.long)
    targets = torch.zeros((batch, MENTIONS), dtype=torch.long)
    for row_index, record in enumerate(records):
        text = str(record[text_key])
        encoded = encode_source(text)
        byte_ids[row_index, : len(encoded)] = torch.tensor(encoded)
        attention[row_index, : len(encoded)] = True
        spans = scan_integer_spans(text)
        if len(spans) != MENTIONS:
            raise EAL1RuntimeError("EAL1 source does not expose four integers")
        for mention_index, (start, end) in enumerate(spans):
            numeric_bounds[row_index, mention_index] = torch.tensor(
                (start + 1, end + 1)
            )
        if role_key is not None:
            roles = tuple(int(value) for value in record[role_key])
            if sorted(roles) != list(range(ROLES)):
                raise EAL1RuntimeError("EAL1 role target is not a permutation")
            targets[row_index] = torch.tensor(roles)
    return (
        byte_ids.to(device),
        attention.to(device),
        numeric_bounds.to(device),
        targets.to(device),
    )


def hard_role_permutation(logits: torch.Tensor) -> tuple[int, int, int, int]:
    if logits.shape != (MENTIONS, ROLES):
        raise EAL1RuntimeError("EAL1 role logits differ")
    scores = [
        sum(float(logits[index, role]) for index, role in enumerate(permutation))
        for permutation in _PERMUTATIONS
    ]
    best = max(range(len(scores)), key=lambda index: (scores[index], -index))
    return _PERMUTATIONS[best]  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class EpisodeLawPacket:
    aliases: tuple[str, ...]
    rows: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    evidence_commitments: tuple[str, ...]
    reader_state_sha256: str
    commitment: str

    def payload(self) -> dict[str, object]:
        return {
            "aliases": list(self.aliases),
            "rows": [[list(row) for row in matrix] for matrix in self.rows],
            "evidence_commitments": list(self.evidence_commitments),
            "reader_state_sha256": self.reader_state_sha256,
        }

    def record(self) -> dict[str, object]:
        return {**self.payload(), "commitment": self.commitment}


@dataclass(frozen=True, slots=True)
class LawCompilation:
    packet: EpisodeLawPacket | None
    error: str | None
    support_sizes: tuple[tuple[int, int], ...]
    evidence_count: int

    def record(self) -> dict[str, object]:
        return {
            "packet": None if self.packet is None else self.packet.record(),
            "error": self.error,
            "support_sizes": [list(value) for value in self.support_sizes],
            "evidence_count": self.evidence_count,
        }


def _operation_index(text: str, aliases: Sequence[str]) -> int:
    present = []
    for index, alias in enumerate(aliases):
        if not alias.isalpha() or not alias.islower():
            raise EAL1RuntimeError("EAL1 alias carrier differs")
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", text):
            present.append(index)
    if len(present) != 1:
        raise EAL1RuntimeError("EAL1 evidence does not bind exactly one alias")
    return present[0]


def compile_episode_laws(
    public: Mapping[str, Any],
    role_assignments: Sequence[Sequence[int]],
    *,
    reader_state_sha256: str,
    text_key: str = "source_text",
    evidence_limit_per_operation: int | None = None,
) -> LawCompilation:
    aliases = tuple(str(value) for value in public["aliases"])
    evidence = tuple(public["evidence"])
    if len(aliases) != OPERATIONS or len(set(aliases)) != OPERATIONS:
        raise EAL1RuntimeError("EAL1 episode alias table differs")
    if len(role_assignments) != len(evidence):
        raise EAL1RuntimeError("EAL1 role/evidence count differs")
    supports = [
        [set(range(len(ROW_CANDIDATES))) for _ in range(OUTPUTS)]
        for _ in range(OPERATIONS)
    ]
    counts = [0] * OPERATIONS
    commitments = []
    consumed = 0
    for record, assignment in zip(evidence, role_assignments, strict=True):
        text = str(record[text_key])
        operation = _operation_index(text, aliases)
        if (
            evidence_limit_per_operation is not None
            and counts[operation] >= evidence_limit_per_operation
        ):
            continue
        roles = tuple(int(value) for value in assignment)
        if sorted(roles) != list(range(ROLES)):
            return LawCompilation(None, "role_not_permutation", tuple(), consumed)
        spans = scan_integer_spans(text)
        values = tuple(int(text[start:end]) for start, end in spans)
        by_role = {role: value for role, value in zip(roles, values, strict=True)}
        before = (by_role[0], by_role[1])
        after = (by_role[2], by_role[3])
        for output in range(OUTPUTS):
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
                sizes = tuple(
                    tuple(len(value) for value in operation_support)
                    for operation_support in supports
                )
                return LawCompilation(None, "empty_support", sizes, consumed + 1)
        counts[operation] += 1
        consumed += 1
        commitment_key = {
            "source_text": "source_sha256",
            "counterfactual_text": "counterfactual_sha256",
            "scrubbed_text": "scrubbed_sha256",
        }.get(text_key)
        if commitment_key is None:
            raise EAL1RuntimeError("EAL1 evidence text view differs")
        commitments.append(str(record[commitment_key]))
    sizes = tuple(
        tuple(len(value) for value in operation_support)
        for operation_support in supports
    )
    if any(size != 1 for operation_sizes in sizes for size in operation_sizes):
        return LawCompilation(None, "underdetermined", sizes, consumed)
    rows = tuple(
        tuple(ROW_CANDIDATES[next(iter(support))] for support in operation_support)
        for operation_support in supports
    )
    provisional = EpisodeLawPacket(
        aliases=aliases,
        rows=rows,  # type: ignore[arg-type]
        evidence_commitments=tuple(commitments),
        reader_state_sha256=reader_state_sha256,
        commitment="",
    )
    packet = replace(provisional, commitment=canonical_sha256(provisional.payload()))
    return LawCompilation(packet, None, sizes, consumed)


def validate_packet(packet: EpisodeLawPacket) -> None:
    if (
        len(packet.aliases) != OPERATIONS
        or len(set(packet.aliases)) != OPERATIONS
        or len(packet.rows) != OPERATIONS
        or any(len(matrix) != OUTPUTS for matrix in packet.rows)
        or any(row not in ROW_CANDIDATES for matrix in packet.rows for row in matrix)
        or canonical_sha256(packet.payload()) != packet.commitment
    ):
        raise EAL1RuntimeError("EAL1 sealed law packet differs")


def execute_program(
    packet: EpisodeLawPacket,
    program: Mapping[str, Any],
) -> tuple[int, int]:
    validate_packet(packet)
    state = tuple(int(value) for value in program["initial_state"])
    if len(state) != 2 or any(value < 0 or value >= PRIME for value in state):
        raise EAL1RuntimeError("EAL1 initial state leaves Z/97Z")
    aliases = {alias: index for index, alias in enumerate(packet.aliases)}
    symbols = tuple(str(value) for value in program["symbols"])
    if int(program["depth"]) != len(symbols):
        raise EAL1RuntimeError("EAL1 program depth differs")
    for symbol in symbols:
        if symbol not in aliases:
            raise EAL1RuntimeError("EAL1 program symbol is unbound")
        matrix = packet.rows[aliases[symbol]]
        state = tuple((row[0] * state[0] + row[1] * state[1]) % PRIME for row in matrix)
    return state  # type: ignore[return-value]


def rebind_packet(
    packet: EpisodeLawPacket,
    aliases: Sequence[str],
) -> EpisodeLawPacket:
    validate_packet(packet)
    provisional = replace(packet, aliases=tuple(aliases), commitment="")
    return replace(provisional, commitment=canonical_sha256(provisional.payload()))


def load_reader(
    path: Path,
    expected_sha256: str,
) -> tuple[NaturalTransitionReader, Mapping[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise EAL1RuntimeError("EAL1 checkpoint file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise EAL1RuntimeError("EAL1 checkpoint schema differs")
    model = NaturalTransitionReader(TransitionReaderConfig(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    if module_state_sha256(model) != payload["model_state_sha256"]:
        raise EAL1RuntimeError("EAL1 reader state hash differs")
    return model, payload


__all__ = [
    "CHECKPOINT_SCHEMA",
    "EAL1RuntimeError",
    "EpisodeLawPacket",
    "LawCompilation",
    "MAX_SOURCE_BYTES",
    "NaturalTransitionReader",
    "TransitionReaderConfig",
    "canonical_sha256",
    "compile_episode_laws",
    "execute_program",
    "hard_role_permutation",
    "load_reader",
    "module_state_sha256",
    "rebind_packet",
    "scan_integer_spans",
    "sha256_path",
    "tensorize_sources",
    "validate_packet",
]
