"""Paired model-owned fault gate for the qualified DSET edit generator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


KEEP_INDEX = 0
REPLACE_INDEX = 1


class GSET1Error(RuntimeError):
    """The frozen GSET1 architecture or artifact contract differs."""


@dataclass(frozen=True)
class GSET1Config:
    hidden_size: int
    gate_width: int = 256
    dropout: float = 0.0

    def validate(self) -> None:
        if min(self.hidden_size, self.gate_width) <= 0 or not 0.0 <= self.dropout < 1.0:
            raise GSET1Error("GSET1 gate dimensions differ")


class GSET1FaultGate(nn.Module):
    """Small decision owner that maps a complete source/draft state to an edit action."""

    def __init__(self, config: GSET1Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.norm = nn.LayerNorm(config.hidden_size)
        self.up = nn.Linear(config.hidden_size, config.gate_width)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(config.dropout)
        self.out = nn.Linear(config.gate_width, 2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2 or hidden.shape[-1] != self.config.hidden_size:
            raise GSET1Error("GSET1 hidden-state geometry differs")
        return self.out(self.dropout(self.activation(self.up(self.norm(hidden.float())))))

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def trainable_parameter_name_sha256(self) -> str:
        names = sorted(name for name, parameter in self.named_parameters() if parameter.requires_grad)
        return hashlib.sha256("\n".join(names).encode()).hexdigest()


def save_gate_checkpoint(
    path: Path,
    gate: GSET1FaultGate,
    optimizer: torch.optim.Optimizer,
    update: int,
    metadata: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema": "shohin-gset1-fault-gate-checkpoint-v1",
            "update": int(update),
            "metadata": metadata,
            "config": asdict(gate.config),
            "gate": gate.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        temporary,
    )
    temporary.replace(path)


def load_gate_checkpoint(path: Path, *, device: str = "cpu") -> tuple[GSET1FaultGate, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema") != "shohin-gset1-fault-gate-checkpoint-v1":
        raise GSET1Error("GSET1 checkpoint schema differs")
    metadata = payload.get("metadata")
    config = payload.get("config")
    if not isinstance(metadata, dict) or not isinstance(config, dict):
        raise GSET1Error("GSET1 checkpoint metadata differs")
    gate = GSET1FaultGate(GSET1Config(**config)).to(device)
    gate.load_state_dict(payload["gate"], strict=True)
    gate.eval()
    return gate, metadata
