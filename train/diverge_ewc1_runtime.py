"""Learned permutation-equivariant WORLD compiler for DIVERGE-EWC1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Literal, Mapping, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from diverge_ats1_data import BYTE_OFFSET, BYTE_VOCAB_SIZE, CLS_ID, PAD_ID
from diverge_ewc1_data import (
    MAX_ALIAS_OCCURRENCES,
    MAX_NUMERIC_MENTIONS,
    MAX_WORLD_BYTES,
    scan_integer_spans,
    scan_symbol_occurrences,
)


SCHEMA = "shohin-diverge-ewc1-runtime-v1"
CompilerMode = Literal["equivariant", "absolute"]


class EWC1RuntimeError(RuntimeError):
    """A learned WORLD compilation violates the EWC1 contract."""


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def module_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("ascii"))
        digest.update(
            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class WorldCompilerConfig:
    width: int = 192
    layers: int = 2
    max_bytes: int = MAX_WORLD_BYTES
    max_numeric_mentions: int = MAX_NUMERIC_MENTIONS
    max_alias_occurrences: int = MAX_ALIAS_OCCURRENCES
    mode: CompilerMode = "equivariant"

    def validate(self) -> None:
        if self.width != 192 or self.layers != 2 or self.width % 2:
            raise EWC1RuntimeError("EWC1 encoder geometry differs")
        if (
            self.max_bytes != MAX_WORLD_BYTES
            or self.max_numeric_mentions != MAX_NUMERIC_MENTIONS
            or self.max_alias_occurrences != MAX_ALIAS_OCCURRENCES
        ):
            raise EWC1RuntimeError("EWC1 candidate geometry differs")
        if self.mode not in ("equivariant", "absolute"):
            raise EWC1RuntimeError("EWC1 compiler mode differs")


@dataclass(frozen=True, slots=True)
class TensorizedWorlds:
    byte_ids: torch.Tensor
    attention_mask: torch.Tensor
    numeric_bounds: torch.Tensor
    numeric_mask: torch.Tensor
    register_masks: torch.Tensor
    alias_masks: torch.Tensor
    alias_mask: torch.Tensor
    alias_group_ids: torch.Tensor
    numeric_targets: torch.Tensor
    operation_targets: torch.Tensor


@dataclass(frozen=True, slots=True)
class CompiledWorld:
    initial_state: tuple[int, int]
    symbols: tuple[int, ...]
    source_sha256: str
    compiler_sha256: str
    numeric_provenance: tuple[tuple[int, int, int], ...]
    operation_provenance: tuple[tuple[int, int, int], ...]
    commitment: str

    def record(self) -> dict[str, object]:
        return {
            "initial_state": list(self.initial_state),
            "symbols": list(self.symbols),
            "source_sha256": self.source_sha256,
            "compiler_sha256": self.compiler_sha256,
            "numeric_provenance": [list(value) for value in self.numeric_provenance],
            "operation_provenance": [
                list(value) for value in self.operation_provenance
            ],
            "commitment": self.commitment,
        }


class EquivariantWorldCompiler(nn.Module):
    """Bind values to declared registers and select ordered alias mentions."""

    def __init__(self, config: WorldCompilerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        width = config.width
        self.embedding = nn.Embedding(BYTE_VOCAB_SIZE, width)
        self.encoder = nn.GRU(
            input_size=width,
            hidden_size=width // 2,
            num_layers=config.layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.output_norm = nn.LayerNorm(width)
        self.numeric_projection = nn.Linear(width, width)
        self.register_projection = nn.Linear(width, width)
        self.global_numeric_projection = nn.Linear(width, width)
        self.absolute_register_keys = nn.Parameter(torch.empty(2, width))
        self.numeric_score = nn.Linear(width, 1)
        self.alias_projection = nn.Linear(width, width)
        self.alias_group_projection = nn.Linear(width, width)
        self.global_alias_projection = nn.Linear(width, width)
        self.operation_score = nn.Linear(width, 1)
        nn.init.normal_(self.absolute_register_keys, std=0.02)

    def _encode(
        self, byte_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if (
            byte_ids.ndim != 2
            or byte_ids.shape != attention_mask.shape
            or byte_ids.shape[1] != self.config.max_bytes
            or byte_ids.dtype != torch.long
            or attention_mask.dtype != torch.bool
        ):
            raise EWC1RuntimeError("EWC1 encoder tensor interface differs")
        lengths = attention_mask.sum(dim=1)
        if torch.any(lengths < 2) or not torch.all(byte_ids[:, 0].eq(CLS_ID)):
            raise EWC1RuntimeError("EWC1 source mask or CLS differs")
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
        return self.output_norm(hidden)

    @staticmethod
    def _pool(hidden: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        counts = masks.sum(dim=-1, keepdim=True).clamp_min(1)
        return torch.einsum(
            "bms,bsw->bmw", masks.to(hidden.dtype), hidden
        ) / counts.to(hidden.dtype)

    def forward(
        self,
        batch: TensorizedWorlds,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encode(batch.byte_ids, batch.attention_mask)
        positions = torch.arange(
            self.config.max_bytes, device=hidden.device
        ).view(1, 1, -1)
        numeric_masks = (
            (positions >= batch.numeric_bounds[:, :, 0].unsqueeze(-1))
            & (positions < batch.numeric_bounds[:, :, 1].unsqueeze(-1))
            & batch.numeric_mask.unsqueeze(-1)
        )
        numeric_hidden = self._pool(hidden, numeric_masks)
        register_hidden = self._pool(hidden, batch.register_masks)
        alias_hidden = self._pool(hidden, batch.alias_masks)
        global_hidden = hidden[:, 0]

        source_register_keys = self.register_projection(register_hidden)
        absolute_register_keys = self.absolute_register_keys.unsqueeze(0).expand(
            hidden.shape[0], -1, -1
        )
        if self.config.mode == "equivariant":
            register_keys = source_register_keys + 0.0 * absolute_register_keys
        else:
            register_keys = absolute_register_keys + 0.0 * source_register_keys
        pair_hidden = torch.tanh(
            self.numeric_projection(numeric_hidden).unsqueeze(1)
            + register_keys.unsqueeze(2)
            + self.global_numeric_projection(global_hidden).unsqueeze(1).unsqueeze(2)
        )
        numeric_logits = self.numeric_score(pair_hidden).squeeze(-1).float()
        numeric_logits = numeric_logits.masked_fill(
            ~batch.numeric_mask.unsqueeze(1), -1.0e9
        )

        group_one_hot = torch.nn.functional.one_hot(
            batch.alias_group_ids.clamp_min(0), num_classes=8
        ).to(alias_hidden.dtype)
        group_one_hot = group_one_hot * batch.alias_mask.unsqueeze(-1)
        group_counts = group_one_hot.sum(dim=1).clamp_min(1.0)
        group_hidden = torch.einsum(
            "bag,baw->bgw", group_one_hot, alias_hidden
        ) / group_counts.unsqueeze(-1)
        gathered_group = torch.gather(
            group_hidden,
            1,
            batch.alias_group_ids.clamp_min(0).unsqueeze(-1).expand(
                -1, -1, hidden.shape[-1]
            ),
        )
        operation_hidden = torch.tanh(
            self.alias_projection(alias_hidden)
            + self.alias_group_projection(gathered_group)
            + self.global_alias_projection(global_hidden).unsqueeze(1)
        )
        operation_logits = self.operation_score(operation_hidden).squeeze(-1).float()
        operation_logits = operation_logits.masked_fill(~batch.alias_mask, -1.0e9)
        return numeric_logits, operation_logits

    def record(self) -> dict[str, object]:
        return {"schema": SCHEMA, "config": asdict(self.config)}


def _encode_source(text: str) -> tuple[int, ...]:
    try:
        raw = text.encode("ascii")
    except UnicodeEncodeError as error:
        raise EWC1RuntimeError("EWC1 WORLD is not ASCII") from error
    if not raw or len(raw) + 1 > MAX_WORLD_BYTES:
        raise EWC1RuntimeError("EWC1 WORLD width differs")
    return (CLS_ID, *(value + BYTE_OFFSET for value in raw))


def tensorize_worlds(
    records: Sequence[Mapping[str, Any]], device: torch.device
) -> TensorizedWorlds:
    if not records:
        raise EWC1RuntimeError("EWC1 batch is empty")
    batch_size = len(records)
    byte_ids = torch.full(
        (batch_size, MAX_WORLD_BYTES), PAD_ID, dtype=torch.long
    )
    attention = torch.zeros((batch_size, MAX_WORLD_BYTES), dtype=torch.bool)
    numeric_bounds = torch.zeros(
        (batch_size, MAX_NUMERIC_MENTIONS, 2), dtype=torch.long
    )
    numeric_mask = torch.zeros(
        (batch_size, MAX_NUMERIC_MENTIONS), dtype=torch.bool
    )
    register_masks = torch.zeros(
        (batch_size, 2, MAX_WORLD_BYTES), dtype=torch.bool
    )
    alias_masks = torch.zeros(
        (batch_size, MAX_ALIAS_OCCURRENCES, MAX_WORLD_BYTES), dtype=torch.bool
    )
    alias_mask = torch.zeros(
        (batch_size, MAX_ALIAS_OCCURRENCES), dtype=torch.bool
    )
    alias_group_ids = torch.full(
        (batch_size, MAX_ALIAS_OCCURRENCES), -1, dtype=torch.long
    )
    numeric_targets = torch.zeros((batch_size, 2), dtype=torch.long)
    operation_targets = torch.zeros(
        (batch_size, MAX_ALIAS_OCCURRENCES), dtype=torch.float32
    )
    for row_index, record in enumerate(records):
        text = str(record["source_text"])
        encoded = _encode_source(text)
        byte_ids[row_index, : len(encoded)] = torch.tensor(encoded)
        attention[row_index, : len(encoded)] = True
        numerics = scan_integer_spans(text)
        if not 2 <= len(numerics) <= MAX_NUMERIC_MENTIONS:
            raise EWC1RuntimeError("EWC1 numeric candidate geometry differs")
        for mention_index, (left, right) in enumerate(numerics):
            numeric_bounds[row_index, mention_index] = torch.tensor(
                (left + 1, right + 1)
            )
            numeric_mask[row_index, mention_index] = True

        aliases = tuple(str(value) for value in record["aliases"])
        registers_raw = tuple(str(value) for value in record["registers"])
        if len(aliases) != 8 or len(registers_raw) != 2:
            raise EWC1RuntimeError("EWC1 declared symbol geometry differs")
        register_occurrences = scan_symbol_occurrences(text, registers_raw)
        for left, right, register_index in register_occurrences:
            register_masks[row_index, register_index, left + 1 : right + 1] = True
        if torch.any(register_masks[row_index].sum(dim=-1) < 1):
            raise EWC1RuntimeError("EWC1 register mention is absent")

        alias_occurrences = scan_symbol_occurrences(text, aliases)
        if not alias_occurrences or len(alias_occurrences) > MAX_ALIAS_OCCURRENCES:
            raise EWC1RuntimeError("EWC1 alias candidate geometry differs")
        for mention_index, (left, right, group_index) in enumerate(alias_occurrences):
            alias_masks[row_index, mention_index, left + 1 : right + 1] = True
            alias_mask[row_index, mention_index] = True
            alias_group_ids[row_index, mention_index] = group_index
        if "numeric_targets" in record:
            numeric_targets[row_index] = torch.tensor(record["numeric_targets"])
        if "operation_targets" in record:
            targets = tuple(float(value) for value in record["operation_targets"])
            if len(targets) != len(alias_occurrences):
                raise EWC1RuntimeError("EWC1 operation target geometry differs")
            operation_targets[row_index, : len(targets)] = torch.tensor(targets)
    return TensorizedWorlds(
        byte_ids=byte_ids.to(device),
        attention_mask=attention.to(device),
        numeric_bounds=numeric_bounds.to(device),
        numeric_mask=numeric_mask.to(device),
        register_masks=register_masks.to(device),
        alias_masks=alias_masks.to(device),
        alias_mask=alias_mask.to(device),
        alias_group_ids=alias_group_ids.to(device),
        numeric_targets=numeric_targets.to(device),
        operation_targets=operation_targets.to(device),
    )


def hard_numeric_assignment(
    logits: torch.Tensor, valid_count: int
) -> tuple[int, int]:
    if logits.shape[0] != 2 or valid_count < 2 or valid_count > logits.shape[1]:
        raise EWC1RuntimeError("EWC1 numeric assignment geometry differs")
    best: tuple[float, int, int] | None = None
    for first in range(valid_count):
        for second in range(valid_count):
            if first == second:
                continue
            candidate = (
                float(logits[0, first] + logits[1, second]),
                -first,
                -second,
            )
            if best is None or candidate > best:
                best = candidate
    if best is None:
        raise EWC1RuntimeError("EWC1 numeric assignment is empty")
    return -best[1], -best[2]


@torch.no_grad()
def compile_world(
    model: EquivariantWorldCompiler,
    record: Mapping[str, Any],
    *,
    device: torch.device,
) -> CompiledWorld:
    text = str(record["source_text"])
    source_sha256 = hashlib.sha256(text.encode("ascii")).hexdigest()
    if source_sha256 != record["source_sha256"]:
        raise EWC1RuntimeError("EWC1 source commitment differs")
    batch = tensorize_worlds([record], device)
    model.eval()
    numeric_logits, operation_logits = model(batch)
    numeric_spans = scan_integer_spans(text)
    assignment = hard_numeric_assignment(numeric_logits[0], len(numeric_spans))
    initial_state = tuple(
        int(text[numeric_spans[index][0] : numeric_spans[index][1]])
        for index in assignment
    )
    aliases = tuple(str(value) for value in record["aliases"])
    alias_occurrences = scan_symbol_occurrences(text, aliases)
    selected = tuple(
        occurrence
        for occurrence, logit in zip(
            alias_occurrences,
            operation_logits[0, : len(alias_occurrences)],
            strict=True,
        )
        if float(logit) >= 0.0
    )
    if not selected:
        raise EWC1RuntimeError("EWC1 selected no operation mentions")
    compiler_sha256 = module_state_sha256(model)
    payload = {
        "initial_state": list(initial_state),
        "symbols": [value[2] for value in selected],
        "source_sha256": source_sha256,
        "compiler_sha256": compiler_sha256,
        "numeric_provenance": [
            [register, numeric_spans[index][0], numeric_spans[index][1]]
            for register, index in enumerate(assignment)
        ],
        "operation_provenance": [list(value) for value in selected],
    }
    return CompiledWorld(
        initial_state=(initial_state[0], initial_state[1]),
        symbols=tuple(value[2] for value in selected),
        source_sha256=source_sha256,
        compiler_sha256=compiler_sha256,
        numeric_provenance=tuple(
            (register, numeric_spans[index][0], numeric_spans[index][1])
            for register, index in enumerate(assignment)
        ),
        operation_provenance=selected,
        commitment=canonical_sha256(payload),
    )


__all__ = [
    "CompiledWorld",
    "CompilerMode",
    "EWC1RuntimeError",
    "EquivariantWorldCompiler",
    "SCHEMA",
    "TensorizedWorlds",
    "WorldCompilerConfig",
    "canonical_sha256",
    "compile_world",
    "hard_numeric_assignment",
    "module_state_sha256",
    "tensorize_worlds",
]
