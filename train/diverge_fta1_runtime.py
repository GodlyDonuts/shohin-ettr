"""Length-equivariant finite-state source compiler for DIVERGE-FTA1."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from diverge_ats1_data import (
    BYTE_VOCAB_SIZE,
    CLS_ID,
    MAX_SEGMENT_BYTES,
    OPERATION_NAMES,
    ROLE_NAMES,
)


class FTA1RuntimeError(RuntimeError):
    """The finite-state source compiler contract was violated."""


SCHEMA = "shohin-diverge-fta1-runtime-v1"


@dataclass(frozen=True, slots=True)
class FTA1Config:
    width: int = 192
    layers: int = 2
    max_bytes: int = MAX_SEGMENT_BYTES

    def validate(self) -> None:
        if self.width <= 0 or self.width % 2 or self.layers <= 0:
            raise FTA1RuntimeError("finite-state geometry differs")
        if self.max_bytes != MAX_SEGMENT_BYTES:
            raise FTA1RuntimeError("FTA1 v1 fixes the source-segment width")


class FiniteStateSourceCompiler(nn.Module):
    """Assign source roles with one tied recurrent update at every byte."""

    def __init__(self, config: FTA1Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.byte_embedding = nn.Embedding(BYTE_VOCAB_SIZE, config.width)
        self.encoder = nn.GRU(
            input_size=config.width,
            hidden_size=config.width // 2,
            num_layers=config.layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.output_norm = nn.LayerNorm(config.width)
        self.role_head = nn.Linear(config.width, len(ROLE_NAMES))
        self.operation_head = nn.Sequential(
            nn.LayerNorm(config.width),
            nn.Linear(config.width, config.width),
            nn.GELU(),
            nn.Linear(config.width, len(OPERATION_NAMES)),
        )

    def forward(
        self,
        byte_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            byte_ids.ndim != 2
            or byte_ids.shape != attention_mask.shape
            or byte_ids.shape[1] != self.config.max_bytes
            or byte_ids.dtype != torch.long
        ):
            raise FTA1RuntimeError("finite-state compiler tensor interface differs")
        active = attention_mask.bool()
        lengths = active.sum(dim=1)
        if torch.any(lengths < 2) or not torch.all(byte_ids[:, 0].eq(CLS_ID)):
            raise FTA1RuntimeError("finite-state source mask or CLS differs")
        embedded = self.byte_embedding(byte_ids)
        packed = pack_padded_sequence(
            embedded,
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
        role_logits = self.role_head(hidden).float()
        weights = active.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        operation_logits = self.operation_head(pooled).float()
        return role_logits, operation_logits

    def record(self) -> dict[str, object]:
        return {"schema": SCHEMA, "config": asdict(self.config)}


__all__ = [
    "FTA1Config",
    "FTA1RuntimeError",
    "FiniteStateSourceCompiler",
    "SCHEMA",
]
