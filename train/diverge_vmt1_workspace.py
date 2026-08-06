"""Permutation-invariant verified multi-trajectory matching for DIVERGE-VMT1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from diverge_ltm1_workspace import (
    FactorizedLatentTrajectoryWorkspace,
    LatentTrajectoryConfig,
    LatentTrajectoryError,
    LatentTrajectoryOutput,
    ordered_trace_targets,
)


class VerifiedTrajectoryError(LatentTrajectoryError):
    """The VMT1 pair geometry or correctness contract differs."""


@dataclass(frozen=True, slots=True)
class VerifiedAssignmentOutput:
    loss: torch.Tensor
    assignment_loss: torch.Tensor
    validity_loss: torch.Tensor
    correct_response_nll: torch.Tensor
    assignment_posterior: torch.Tensor
    assignment_costs: torch.Tensor
    best_assignment: torch.Tensor
    matched_trace_cosine: torch.Tensor
    crossed_trace_cosine: torch.Tensor
    selector_correct: torch.Tensor
    swapped_selector_correct: torch.Tensor
    selected_correct_response_nll: torch.Tensor


def paired_ordered_trace_targets(
    embedding: nn.Embedding,
    response_pair_rows: Sequence[Sequence[Sequence[int]]],
    recurrent_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create ordered targets for two complete observed trajectories per prompt."""

    if not response_pair_rows or any(len(pair) != 2 for pair in response_pair_rows):
        raise VerifiedTrajectoryError("VMT1 requires exactly two response trajectories")
    flat = [response for pair in response_pair_rows for response in pair]
    targets, active = ordered_trace_targets(embedding, flat, recurrent_steps)
    batch = len(response_pair_rows)
    return (
        targets.view(batch, 2, recurrent_steps, -1),
        active.view(batch, 2, recurrent_steps),
    )


def complete_trace_cost_matrix(
    probes: torch.Tensor,
    targets: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Return whole ordered-trace costs for every lineage/target pairing."""

    if probes.ndim != 4 or probes.shape[1] != 2:
        raise VerifiedTrajectoryError("VMT1 probes must contain two trajectories")
    batch, _, steps, width = probes.shape
    if targets.shape != (batch, 2, steps, width):
        raise VerifiedTrajectoryError("VMT1 target geometry differs")
    if active.shape != (batch, 2, steps) or not active.any(dim=-1).all():
        raise VerifiedTrajectoryError("VMT1 target mask geometry differs")
    cosine = F.cosine_similarity(
        probes.float().unsqueeze(2),
        targets.float().unsqueeze(1),
        dim=-1,
        eps=1e-6,
    )
    weights = active.to(dtype=cosine.dtype).unsqueeze(1)
    return ((1.0 - cosine) * weights).sum(dim=-1) / weights.sum(dim=-1)


def verified_pair_assignment_objective(
    trace_cost: torch.Tensor,
    validity_logits: torch.Tensor,
    target_correct: torch.Tensor,
    correct_response_nll: torch.Tensor,
    *,
    assignment_temperature: float,
    validity_margin: float,
    trace_weight: float,
    validity_weight: float,
) -> VerifiedAssignmentOutput:
    """Marginalize the two complete bijections and rank the correct lineage."""

    shape = trace_cost.shape
    if len(shape) != 3 or shape[1:] != (2, 2):
        raise VerifiedTrajectoryError("VMT1 trace cost must have shape [batch,2,2]")
    batch = shape[0]
    if validity_logits.shape != (batch, 2):
        raise VerifiedTrajectoryError("VMT1 validity logits differ")
    if target_correct.shape != (batch, 2):
        raise VerifiedTrajectoryError("VMT1 correctness geometry differs")
    if correct_response_nll.shape != (batch, 2):
        raise VerifiedTrajectoryError("VMT1 correct-response NLL geometry differs")
    if not torch.equal(
        target_correct.to(torch.long).sum(dim=-1),
        torch.ones(batch, device=target_correct.device, dtype=torch.long),
    ):
        raise VerifiedTrajectoryError(
            "every VMT1 pair must have exactly one correct target"
        )
    if assignment_temperature <= 0.0 or validity_margin < 0.0:
        raise VerifiedTrajectoryError("VMT1 assignment temperature or margin differs")
    if trace_weight < 0.0 or validity_weight < 0.0:
        raise VerifiedTrajectoryError("VMT1 loss weights must be nonnegative")
    tensors = (trace_cost, validity_logits, correct_response_nll)
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise VerifiedTrajectoryError("VMT1 objective contains nonfinite values")

    permutations = torch.tensor(
        ((0, 1), (1, 0)), device=trace_cost.device, dtype=torch.long
    )
    lineage = torch.arange(2, device=trace_cost.device)
    assignment_costs = torch.stack(
        [
            trace_cost[:, lineage, permutation].mean(dim=-1)
            for permutation in permutations
        ],
        dim=-1,
    )
    assignment_posterior = torch.softmax(
        -assignment_costs / assignment_temperature, dim=-1
    )
    assignment_loss_rows = -assignment_temperature * torch.logsumexp(
        -assignment_costs / assignment_temperature, dim=-1
    ) + assignment_temperature * math.log(2.0)
    assignment_loss = assignment_loss_rows.mean()

    labels = torch.stack(
        [target_correct[:, permutation] for permutation in permutations], dim=1
    ).to(dtype=validity_logits.dtype)
    scores = validity_logits.unsqueeze(1).expand_as(labels)
    bce = F.binary_cross_entropy_with_logits(scores, labels, reduction="none").mean(
        dim=-1
    )
    correct_scores = (scores * labels).sum(dim=-1)
    wrong_scores = (scores * (1.0 - labels)).sum(dim=-1)
    rank = F.softplus(validity_margin - (correct_scores - wrong_scores))
    validity_by_assignment = bce + rank
    detached_assignment = assignment_posterior.detach()
    validity_loss = (detached_assignment * validity_by_assignment).sum(dim=-1).mean()

    correct_nll_by_assignment = (correct_response_nll.unsqueeze(1) * labels).sum(dim=-1)
    correct_response_loss = (
        (detached_assignment * correct_nll_by_assignment).sum(dim=-1).mean()
    )
    loss = (
        correct_response_loss
        + trace_weight * assignment_loss
        + validity_weight * validity_loss
    )

    best_assignment = assignment_posterior.argmax(dim=-1)
    row = torch.arange(batch, device=trace_cost.device)
    best_labels = labels[row, best_assignment]
    selected = validity_logits.argmax(dim=-1)
    selector_correct = best_labels[row, selected].to(torch.bool)
    swapped_selected = validity_logits.flip(dims=(1,)).argmax(dim=-1)
    swapped_selector_correct = best_labels[row, swapped_selected].to(torch.bool)
    selected_correct_response_nll = correct_response_nll[row, selected]
    best_cost = assignment_costs[row, best_assignment]
    crossed_cost = assignment_costs[row, 1 - best_assignment]
    return VerifiedAssignmentOutput(
        loss=loss,
        assignment_loss=assignment_loss,
        validity_loss=validity_loss,
        correct_response_nll=correct_response_loss,
        assignment_posterior=assignment_posterior,
        assignment_costs=assignment_costs,
        best_assignment=best_assignment,
        matched_trace_cosine=1.0 - best_cost,
        crossed_trace_cosine=1.0 - crossed_cost,
        selector_correct=selector_correct,
        swapped_selector_correct=swapped_selector_correct,
        selected_correct_response_nll=selected_correct_response_nll,
    )


def vmt1_architecture_sha256(config: LatentTrajectoryConfig) -> str:
    config.validate()
    if config.candidate_count != 2:
        raise VerifiedTrajectoryError("VMT1 requires exactly two trajectories")
    payload = {
        "schema": "shohin-diverge-vmt1-v1",
        "config": asdict(config),
        "mechanism": (
            "paired-complete-observed-trajectories+exact-bijection-marginal+"
            "correct-lineage-language-credit+model-owned-terminal-validity+"
            "single-coherent-lineage-decode"
        ),
        "field_averaging": False,
        "teacher_or_verifier_at_inference": False,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "FactorizedLatentTrajectoryWorkspace",
    "LatentTrajectoryConfig",
    "LatentTrajectoryOutput",
    "VerifiedAssignmentOutput",
    "VerifiedTrajectoryError",
    "complete_trace_cost_matrix",
    "paired_ordered_trace_targets",
    "verified_pair_assignment_objective",
    "vmt1_architecture_sha256",
]
