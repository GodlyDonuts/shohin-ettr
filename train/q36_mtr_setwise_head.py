#!/usr/bin/env python3
"""Permutation-equivariant setwise semantic commit head for Q36 trajectories."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SetwiseCommitHead(nn.Module):
    """Score each trajectory using itself and a symmetric competing-set context."""

    def __init__(
        self, hidden_size: int, width: int = 512, projection: int = 256
    ) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projection),
            nn.GELU(),
        )
        self.score = nn.Sequential(
            nn.Linear(projection * 4, width),
            nn.GELU(),
            nn.Linear(width, width // 2),
            nn.GELU(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.shape[1] != 3:
            raise ValueError("setwise commit hidden geometry differs")
        projected = self.project(hidden.float())
        context = (
            torch.stack(
                (
                    projected[:, 1] + projected[:, 2],
                    projected[:, 0] + projected[:, 2],
                    projected[:, 0] + projected[:, 1],
                ),
                dim=1,
            )
            * 0.5
        )
        features = torch.cat(
            (projected, context, projected - context, projected * context), dim=-1
        )
        return self.score(features).squeeze(-1)


def setwise_selection_loss(
    scores: torch.Tensor,
    correct: torch.Tensor,
    *,
    binary_weight: float = 0.5,
) -> torch.Tensor:
    """Combine absolute correctness calibration with listwise correct-set ranking."""

    if (
        scores.ndim != 2
        or correct.shape != scores.shape
        or correct.dtype != torch.bool
        or not 0.0 <= binary_weight <= 1.0
    ):
        raise ValueError("setwise commit target geometry differs")
    binary = F.binary_cross_entropy_with_logits(scores, correct.float())
    has_correct = correct.any(dim=1)
    listwise_rows: list[torch.Tensor] = []
    for row_scores, row_correct, row_has_correct in zip(
        scores, correct, has_correct, strict=True
    ):
        if bool(row_has_correct):
            listwise_rows.append(
                torch.logsumexp(row_scores, dim=0)
                - torch.logsumexp(row_scores[row_correct], dim=0)
            )
        else:
            listwise_rows.append(F.softplus(row_scores).mean())
    listwise = torch.stack(listwise_rows).mean()
    return binary_weight * binary + (1.0 - binary_weight) * listwise
