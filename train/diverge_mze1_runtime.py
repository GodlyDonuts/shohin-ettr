#!/usr/bin/env python3
"""Learned finite-field transition owner for DIVERGE-MZE1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import torch.nn as nn


SCHEMA = "shohin-diverge-mze1-runtime-v1"
CHECKPOINT_SCHEMA = "shohin-diverge-mze1-checkpoint-v1"
PRIME = 97
OPERATIONS = 8
OUTPUTS = 2
COEFFICIENTS = (-2, -1, 0, 1, 2)
ROW_CANDIDATES = tuple((left, right) for left in COEFFICIENTS for right in COEFFICIENTS)


class MZE1RuntimeError(RuntimeError):
    """A learned executor violates the frozen finite-field contract."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
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


@dataclass(frozen=True, slots=True)
class MZE1Config:
    prime: int = PRIME
    operations: int = OPERATIONS
    outputs: int = OUTPUTS
    coefficients: tuple[int, ...] = COEFFICIENTS

    def validate(self) -> None:
        if (
            self.prime != PRIME
            or self.operations != OPERATIONS
            or self.outputs != OUTPUTS
            or tuple(self.coefficients) != COEFFICIENTS
        ):
            raise MZE1RuntimeError("MZE1 algebra geometry differs")


class PresentedZ97Executor(nn.Module):
    """Infer one coherent 2x2 linear law for each opaque operation token."""

    def __init__(self, config: MZE1Config | None = None) -> None:
        super().__init__()
        self.config = config or MZE1Config()
        self.config.validate()
        self.row_logits = nn.Parameter(
            torch.zeros(OPERATIONS, OUTPUTS, len(ROW_CANDIDATES))
        )
        self.register_buffer(
            "row_candidates",
            torch.tensor(ROW_CANDIDATES, dtype=torch.long),
            persistent=True,
        )

    def outcome_nll(
        self,
        operations: torch.Tensor,
        states: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Marginalize every coefficient row consistent with observed outcomes."""

        if (
            operations.ndim != 1
            or states.shape != (operations.shape[0], 2)
            or targets.shape != states.shape
            or operations.dtype != torch.long
            or states.dtype != torch.long
            or targets.dtype != torch.long
        ):
            raise MZE1RuntimeError("MZE1 training tensor geometry differs")
        if (
            torch.any(operations < 0)
            or torch.any(operations >= OPERATIONS)
            or torch.any(states < 0)
            or torch.any(states >= PRIME)
            or torch.any(targets < 0)
            or torch.any(targets >= PRIME)
        ):
            raise MZE1RuntimeError("MZE1 training value leaves Z/97Z")
        candidates = self.row_candidates.to(states.device)
        values = (
            states[:, None, 0] * candidates[None, :, 0]
            + states[:, None, 1] * candidates[None, :, 1]
        ).remainder(PRIME)
        logits = self.row_logits[operations]
        losses = []
        for output in range(OUTPUTS):
            compatible = values.eq(targets[:, output : output + 1])
            if torch.any(~compatible.any(dim=1)):
                raise MZE1RuntimeError("MZE1 catalog cannot explain an outcome")
            numerator = torch.logsumexp(
                logits[:, output].masked_fill(~compatible, -torch.inf), dim=-1
            )
            denominator = torch.logsumexp(logits[:, output], dim=-1)
            losses.append(denominator - numerator)
        return torch.stack(losses, dim=-1).mean()

    def hard_rows(self) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
        indices = self.row_logits.detach().cpu().argmax(dim=-1)
        return tuple(
            (
                ROW_CANDIDATES[int(indices[operation, 0])],
                ROW_CANDIDATES[int(indices[operation, 1])],
            )
            for operation in range(OPERATIONS)
        )

    def transition(self, operation: int, state: tuple[int, int]) -> tuple[int, int]:
        if operation < 0 or operation >= OPERATIONS:
            raise MZE1RuntimeError("MZE1 operation is outside its carrier")
        if any(value < 0 or value >= PRIME for value in state):
            raise MZE1RuntimeError("MZE1 state is outside Z/97Z")
        rows = self.hard_rows()[operation]
        return tuple((row[0] * state[0] + row[1] * state[1]) % PRIME for row in rows)  # type: ignore[return-value]

    def record(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "config": asdict(self.config),
            "parameter_count": sum(
                parameter.numel() for parameter in self.parameters()
            ),
            "state_sha256": module_state_sha256(self),
            "hard_rows": [
                [list(row) for row in operation] for operation in self.hard_rows()
            ],
        }


def freeze_transition(
    model: PresentedZ97Executor,
) -> Callable[[int, tuple[int, int]], tuple[int, int]]:
    """Commit learned rows once so recurrent execution performs no Torch launch."""

    rows = model.hard_rows()

    def transition(operation: int, state: tuple[int, int]) -> tuple[int, int]:
        if operation < 0 or operation >= OPERATIONS:
            raise MZE1RuntimeError("MZE1 operation is outside its carrier")
        if any(value < 0 or value >= PRIME for value in state):
            raise MZE1RuntimeError("MZE1 state is outside Z/97Z")
        selected = rows[operation]
        return tuple(
            (row[0] * state[0] + row[1] * state[1]) % PRIME for row in selected
        )  # type: ignore[return-value]

    return transition


def load_executor(
    path: Path,
    expected_sha256: str,
    *,
    arm: str = "treatment",
) -> tuple[PresentedZ97Executor, Mapping[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise MZE1RuntimeError("MZE1 checkpoint file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise MZE1RuntimeError("MZE1 checkpoint schema differs")
    if arm not in ("treatment", "shuffled"):
        raise MZE1RuntimeError("MZE1 checkpoint arm differs")
    config = MZE1Config(**payload["config"])
    model = PresentedZ97Executor(config)
    model.load_state_dict(payload[f"{arm}_state_dict"], strict=True)
    model.eval()
    expected_state = str(payload[f"{arm}_state_sha256"])
    if module_state_sha256(model) != expected_state:
        raise MZE1RuntimeError("MZE1 checkpoint state hash differs")
    return model, payload


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CHECKPOINT_SCHEMA",
    "COEFFICIENTS",
    "MZE1Config",
    "MZE1RuntimeError",
    "OPERATIONS",
    "OUTPUTS",
    "PRIME",
    "PresentedZ97Executor",
    "ROW_CANDIDATES",
    "canonical_sha256",
    "freeze_transition",
    "load_executor",
    "module_state_sha256",
    "sha256_path",
]
