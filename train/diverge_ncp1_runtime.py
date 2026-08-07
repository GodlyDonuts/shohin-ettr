#!/usr/bin/env python3
"""Content-addressed CTC command pointer for DIVERGE-NCP1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from diverge_ats1_data import BYTE_OFFSET, BYTE_VOCAB_SIZE, CLS_ID, PAD_ID
from diverge_eal1_runtime import module_state_sha256, sha256_path
from diverge_ncp1_data import OPERATIONS


SCHEMA = "shohin-diverge-ncp1-runtime-v1"
CHECKPOINT_SCHEMA = "shohin-diverge-ncp1-checkpoint-v1"
MAX_COMMAND_BYTES = 1_536
MAX_ALIAS_BYTES = 32
BLANK_ID = OPERATIONS


class NCP1RuntimeError(RuntimeError):
    """A natural command pointer violates its frozen contract."""


@dataclass(frozen=True, slots=True)
class CommandPointerConfig:
    width: int = 128
    layers: int = 2
    max_command_bytes: int = MAX_COMMAND_BYTES
    max_alias_bytes: int = MAX_ALIAS_BYTES

    def validate(self) -> None:
        if (
            self.width != 128
            or self.layers != 2
            or self.max_command_bytes != MAX_COMMAND_BYTES
            or self.max_alias_bytes != MAX_ALIAS_BYTES
            or self.width % 2
        ):
            raise NCP1RuntimeError("NCP1 model geometry differs")


class NaturalCommandPointer(nn.Module):
    """Decode a variable-length program as pointers into an episode alias table."""

    def __init__(self, config: CommandPointerConfig | None = None) -> None:
        super().__init__()
        self.config = config or CommandPointerConfig()
        self.config.validate()
        self.embedding = nn.Embedding(BYTE_VOCAB_SIZE, self.config.width)
        self.command_encoder = nn.GRU(
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
        self.command_norm = nn.LayerNorm(self.config.width)
        self.alias_norm = nn.LayerNorm(self.config.width)
        self.command_projection = nn.Linear(self.config.width, self.config.width)
        self.alias_projection = nn.Linear(self.config.width, self.config.width)
        self.blank_head = nn.Linear(self.config.width, 1)
        self.logit_scale = nn.Parameter(torch.tensor([1.0]))

    def forward(
        self,
        command_ids: torch.Tensor,
        command_mask: torch.Tensor,
        alias_ids: torch.Tensor,
        alias_mask: torch.Tensor,
    ) -> torch.Tensor:
        if (
            command_ids.ndim != 2
            or command_ids.shape != command_mask.shape
            or command_ids.dtype != torch.long
            or command_mask.dtype != torch.bool
            or command_ids.shape[1] > MAX_COMMAND_BYTES
            or alias_ids.ndim != 3
            or alias_ids.shape[:2] != (command_ids.shape[0], OPERATIONS)
            or alias_ids.shape != alias_mask.shape
            or alias_ids.dtype != torch.long
            or alias_mask.dtype != torch.bool
            or alias_ids.shape[2] > MAX_ALIAS_BYTES
        ):
            raise NCP1RuntimeError("NCP1 forward tensor geometry differs")
        command_lengths = command_mask.sum(dim=1)
        alias_lengths = alias_mask.flatten(0, 1).sum(dim=1)
        if torch.any(command_lengths < 2) or torch.any(alias_lengths < 2):
            raise NCP1RuntimeError("NCP1 input sequence is empty")
        command_packed = pack_padded_sequence(
            self.embedding(command_ids),
            command_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        command_encoded, _ = self.command_encoder(command_packed)
        command_hidden, _ = pad_packed_sequence(
            command_encoded,
            batch_first=True,
            total_length=command_ids.shape[1],
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
            command_ids.shape[0], OPERATIONS, self.config.width
        )

        command_keys = torch.nn.functional.normalize(
            self.command_projection(self.command_norm(command_hidden)), dim=-1
        )
        alias_keys = torch.nn.functional.normalize(
            self.alias_projection(self.alias_norm(alias_hidden)), dim=-1
        )
        pointer_logits = torch.einsum("btw,bow->bto", command_keys, alias_keys)
        pointer_logits = pointer_logits * self.logit_scale.exp().clamp(max=100.0)
        blank_logits = self.blank_head(command_hidden)
        return torch.cat((pointer_logits, blank_logits), dim=-1).float()

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
        raise NCP1RuntimeError("NCP1 source is not ASCII") from error
    values = (CLS_ID, *(value + BYTE_OFFSET for value in encoded))
    if len(values) > maximum:
        raise NCP1RuntimeError("NCP1 source exceeds frozen width")
    return values


def tensorize_commands(
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    text_key: str = "source_text",
    aliases_key: str = "aliases",
    rotate_alias_table: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not records:
        raise NCP1RuntimeError("NCP1 command batch is empty")
    command_values = [
        _encode_ascii(str(record[text_key]), MAX_COMMAND_BYTES) for record in records
    ]
    alias_values = []
    for record in records:
        aliases = tuple(str(value) for value in record[aliases_key])
        if len(aliases) != OPERATIONS or len(set(aliases)) != OPERATIONS:
            raise NCP1RuntimeError("NCP1 alias table differs")
        if rotate_alias_table:
            aliases = aliases[1:] + aliases[:1]
        alias_values.append(
            tuple(_encode_ascii(value, MAX_ALIAS_BYTES) for value in aliases)
        )
    command_width = max(len(value) for value in command_values)
    alias_width = max(len(alias) for table in alias_values for alias in table)
    command_ids = torch.full((len(records), command_width), PAD_ID, dtype=torch.long)
    command_mask = torch.zeros_like(command_ids, dtype=torch.bool)
    alias_ids = torch.full(
        (len(records), OPERATIONS, alias_width), PAD_ID, dtype=torch.long
    )
    alias_mask = torch.zeros_like(alias_ids, dtype=torch.bool)
    for row, values in enumerate(command_values):
        command_ids[row, : len(values)] = torch.tensor(values)
        command_mask[row, : len(values)] = True
        for alias_index, alias in enumerate(alias_values[row]):
            alias_ids[row, alias_index, : len(alias)] = torch.tensor(alias)
            alias_mask[row, alias_index, : len(alias)] = True
    lengths = command_mask.sum(dim=1)
    return (
        command_ids.to(device),
        command_mask.to(device),
        alias_ids.to(device),
        alias_mask.to(device),
        lengths.to(device),
    )


def greedy_ctc_decode(
    logits: torch.Tensor, lengths: torch.Tensor
) -> list[tuple[int, ...]]:
    if logits.ndim != 3 or logits.shape[2] != OPERATIONS + 1:
        raise NCP1RuntimeError("NCP1 decode logits differ")
    predictions = logits.detach().cpu().argmax(dim=-1)
    output = []
    for row, length in zip(predictions, lengths.detach().cpu(), strict=True):
        collapsed = []
        previous = None
        for token in row[: int(length)]:
            value = int(token)
            if value != previous and value != BLANK_ID:
                collapsed.append(value)
            previous = value
        output.append(tuple(collapsed))
    return output


def load_pointer(
    path: Path, expected_sha256: str
) -> tuple[NaturalCommandPointer, Mapping[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise NCP1RuntimeError("NCP1 checkpoint file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise NCP1RuntimeError("NCP1 checkpoint schema differs")
    model = NaturalCommandPointer(CommandPointerConfig(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    if module_state_sha256(model) != payload["model_state_sha256"]:
        raise NCP1RuntimeError("NCP1 model state hash differs")
    return model, payload


__all__ = [
    "BLANK_ID",
    "CHECKPOINT_SCHEMA",
    "CommandPointerConfig",
    "MAX_ALIAS_BYTES",
    "MAX_COMMAND_BYTES",
    "NCP1RuntimeError",
    "NaturalCommandPointer",
    "greedy_ctc_decode",
    "load_pointer",
    "tensorize_commands",
]
