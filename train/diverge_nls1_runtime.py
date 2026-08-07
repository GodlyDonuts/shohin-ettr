#!/usr/bin/env python3
"""Permutation-invariant neural episode-law synthesizer for DIVERGE-NLS1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from diverge_eal1_runtime import (
    EpisodeLawPacket,
    LawCompilation,
    canonical_sha256,
    module_state_sha256,
    scan_integer_spans,
    sha256_path,
)
from diverge_eal2_runtime import compose_complete_roles
from diverge_mze1_runtime import PRIME, ROW_CANDIDATES
from diverge_nls1_data import DEMONSTRATIONS, VALUES


SCHEMA = "shohin-diverge-nls1-runtime-v1"
CHECKPOINT_SCHEMA = "shohin-diverge-nls1-checkpoint-v1"
OPERATIONS = 8
OUTPUTS = 2


class NLS1RuntimeError(RuntimeError):
    """A neural law packet or model violates its frozen contract."""


@dataclass(frozen=True, slots=True)
class NeuralLawSynthesizerConfig:
    value_width: int = 64
    hidden_width: int = 256
    demonstrations: int = DEMONSTRATIONS
    values: int = VALUES

    def validate(self) -> None:
        if (
            self.value_width != 64
            or self.hidden_width != 256
            or self.demonstrations != DEMONSTRATIONS
            or self.values != VALUES
        ):
            raise NLS1RuntimeError("NLS1 model geometry differs")


class NeuralLawSynthesizer(nn.Module):
    """Map an unordered set of complete transitions to two coefficient rows."""

    def __init__(self, config: NeuralLawSynthesizerConfig | None = None) -> None:
        super().__init__()
        self.config = config or NeuralLawSynthesizerConfig()
        self.config.validate()
        self.value_embedding = nn.Embedding(PRIME, self.config.value_width)
        self.demonstration_encoder = nn.Sequential(
            nn.Linear(
                self.config.values * self.config.value_width, self.config.hidden_width
            ),
            nn.GELU(),
            nn.Linear(self.config.hidden_width, self.config.hidden_width),
            nn.GELU(),
        )
        self.row_head = nn.Sequential(
            nn.LayerNorm(self.config.hidden_width),
            nn.Linear(self.config.hidden_width, self.config.hidden_width),
            nn.GELU(),
            nn.Linear(self.config.hidden_width, OUTPUTS * len(ROW_CANDIDATES)),
        )

    def forward(self, demonstrations: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if (
            demonstrations.ndim != 3
            or demonstrations.shape[1:] != (DEMONSTRATIONS, VALUES)
            or demonstrations.dtype != torch.long
            or mask.shape != demonstrations.shape[:2]
            or mask.dtype != torch.bool
            or torch.any(demonstrations < 0)
            or torch.any(demonstrations >= PRIME)
            or torch.any(mask.sum(dim=1) < 1)
        ):
            raise NLS1RuntimeError("NLS1 forward tensor geometry differs")
        encoded = self.demonstration_encoder(
            self.value_embedding(demonstrations).flatten(start_dim=2)
        )
        pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1)
        return self.row_head(pooled).view(-1, OUTPUTS, len(ROW_CANDIDATES)).float()

    def record(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "config": asdict(self.config),
            "parameter_count": sum(
                parameter.numel() for parameter in self.parameters()
            ),
            "state_sha256": module_state_sha256(self),
        }


def tensorize_law_rows(
    rows: Sequence[Mapping[str, Any]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    demonstrations = torch.tensor(
        [row["demonstrations"] for row in rows], dtype=torch.long, device=device
    )
    mask = torch.ones(demonstrations.shape[:2], dtype=torch.bool, device=device)
    targets = torch.tensor(
        [row["target_row_ids"] for row in rows], dtype=torch.long, device=device
    )
    return demonstrations, mask, targets


def hard_rows(logits: torch.Tensor) -> tuple[tuple[int, int], tuple[int, int]]:
    if logits.shape != (OUTPUTS, len(ROW_CANDIDATES)):
        raise NLS1RuntimeError("NLS1 hard-row logits differ")
    indices = logits.detach().cpu().argmax(dim=-1)
    return tuple(ROW_CANDIDATES[int(index)] for index in indices)  # type: ignore[return-value]


def _operation_index(text: str, aliases: Sequence[str]) -> int:
    present = [
        index
        for index, alias in enumerate(aliases)
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", text)
    ]
    if len(present) != 1:
        raise NLS1RuntimeError("NLS1 evidence does not bind one operation")
    return present[0]


def episode_demonstrations(
    public: Mapping[str, Any],
    temporal_assignments: Sequence[Sequence[int]],
    *,
    text_key: str,
) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
    evidence = tuple(public["evidence"])
    complete = compose_complete_roles(evidence, temporal_assignments, text_key=text_key)
    aliases = tuple(str(value) for value in public["aliases"])
    grouped: list[list[tuple[int, int, int, int]]] = [[] for _ in range(OPERATIONS)]
    for record, roles in zip(evidence, complete, strict=True):
        text = str(record[text_key])
        spans = scan_integer_spans(text)
        values = tuple(int(text[start:end]) for start, end in spans)
        by_role = {role: value for role, value in zip(roles, values, strict=True)}
        grouped[_operation_index(text, aliases)].append(
            (by_role[0], by_role[1], by_role[2], by_role[3])
        )
    if any(len(value) != DEMONSTRATIONS for value in grouped):
        raise NLS1RuntimeError("NLS1 episode demonstration count differs")
    return tuple(tuple(value) for value in grouped)


@torch.no_grad()
def compile_neural_laws(
    public: Mapping[str, Any],
    temporal_assignments: Sequence[Sequence[int]],
    model: NeuralLawSynthesizer,
    *,
    device: torch.device,
    reader_state_sha256: str,
    text_key: str = "source_text",
    control: str = "normal",
) -> LawCompilation:
    if control not in ("normal", "one_example", "scrub_outcomes"):
        raise NLS1RuntimeError("NLS1 compilation control differs")
    grouped = episode_demonstrations(public, temporal_assignments, text_key=text_key)
    values = torch.tensor(grouped, dtype=torch.long, device=device)
    mask = torch.ones((OPERATIONS, DEMONSTRATIONS), dtype=torch.bool, device=device)
    if control == "one_example":
        mask[:, 1:] = False
    elif control == "scrub_outcomes":
        values[:, :, 2:] = 0
    rows = tuple(hard_rows(logits) for logits in model(values, mask))
    aliases = tuple(str(value) for value in public["aliases"])
    commitments = tuple(
        str(record[f"{text_key.removesuffix('_text')}_sha256"])
        for record in public["evidence"]
    )
    owner_hash = canonical_sha256(
        ["nls1-owner", reader_state_sha256, module_state_sha256(model)]
    )
    provisional = EpisodeLawPacket(
        aliases=aliases,
        rows=rows,
        evidence_commitments=commitments,
        reader_state_sha256=owner_hash,
        commitment="",
    )
    packet = replace(provisional, commitment=canonical_sha256(provisional.payload()))
    return LawCompilation(
        packet=packet,
        error=None,
        support_sizes=tuple((1, 1) for _ in range(OPERATIONS)),
        evidence_count=len(public["evidence"]),
    )


def load_synthesizer(
    path: Path, expected_sha256: str
) -> tuple[NeuralLawSynthesizer, Mapping[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise NLS1RuntimeError("NLS1 checkpoint file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise NLS1RuntimeError("NLS1 checkpoint schema differs")
    model = NeuralLawSynthesizer(NeuralLawSynthesizerConfig(**payload["config"]))
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    if module_state_sha256(model) != payload["model_state_sha256"]:
        raise NLS1RuntimeError("NLS1 model state hash differs")
    return model, payload


__all__ = [
    "CHECKPOINT_SCHEMA",
    "NLS1RuntimeError",
    "NeuralLawSynthesizer",
    "NeuralLawSynthesizerConfig",
    "compile_neural_laws",
    "episode_demonstrations",
    "hard_rows",
    "load_synthesizer",
    "tensorize_law_rows",
]
