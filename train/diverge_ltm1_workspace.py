"""Factorized complete latent trajectories with smooth whole-lineage credit.

LTM1 keeps every candidate trajectory separate.  Training marginalizes scalar
energies over complete candidates; inference gathers exactly one candidate
prefix.  Candidate state fields are never averaged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from integrated_reasoning_workspace import (
    IntegratedWorkspaceConfig,
    TiedWorkspaceCell,
)


class LatentTrajectoryError(ValueError):
    """The LTM1 geometry or tensor contract differs."""


@dataclass(frozen=True, slots=True)
class LatentTrajectoryConfig:
    backbone_width: int
    latent_width: int = 384
    trajectory_slots: int = 8
    recurrent_steps: int = 8
    fault_bits: int = 2
    attention_heads: int = 8
    ff_multiplier: int = 2
    dropout: float = 0.0

    @property
    def candidate_count(self) -> int:
        return 1 << self.fault_bits

    def validate(self) -> None:
        dimensions = (
            self.backbone_width,
            self.latent_width,
            self.trajectory_slots,
            self.recurrent_steps,
            self.fault_bits,
            self.attention_heads,
            self.ff_multiplier,
        )
        if any(value <= 0 for value in dimensions):
            raise LatentTrajectoryError("LTM1 dimensions must be positive")
        if self.fault_bits > 4:
            raise LatentTrajectoryError("LTM1 admits at most four fault bits")
        if self.latent_width % self.attention_heads:
            raise LatentTrajectoryError(
                "latent width must divide evenly across attention heads"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise LatentTrajectoryError("dropout is outside [0, 1)")


@dataclass(frozen=True, slots=True)
class LatentTrajectoryOutput:
    candidate_prefixes: torch.Tensor
    trajectory_probes: torch.Tensor
    prior_logits: torch.Tensor
    stop_logits: torch.Tensor
    step_delta_norms: torch.Tensor


@dataclass(frozen=True, slots=True)
class MarginalTrajectoryLoss:
    loss: torch.Tensor
    marginal_energy: torch.Tensor
    balance_loss: torch.Tensor
    posterior: torch.Tensor
    candidate_energy: torch.Tensor


def _binary_assignments(fault_bits: int) -> torch.Tensor:
    candidate_count = 1 << fault_bits
    values = [
        [(candidate >> bit) & 1 for bit in range(fault_bits)]
        for candidate in range(candidate_count)
    ]
    return torch.tensor(values, dtype=torch.long)


class FactorizedLatentTrajectoryWorkspace(nn.Module):
    """Build and recurrently update all complete sticky fault-line assignments."""

    def __init__(self, config: LatentTrajectoryConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        width = config.latent_width
        slots = config.trajectory_slots
        bits = config.fault_bits

        self.prompt_projection = nn.Linear(config.backbone_width, width, bias=False)
        self.initial_slots = nn.Parameter(torch.empty(slots, width))
        self.shared_seed = nn.Linear(width, width)
        self.factor_seed = nn.Linear(width, bits * 2 * width)
        self.factor_slots = nn.Parameter(torch.empty(bits, 2, slots, width))
        self.prior_head = nn.Linear(width, config.candidate_count)

        cell_config = IntegratedWorkspaceConfig(
            backbone_width=width,
            workspace_width=width,
            workspace_slots=slots,
            recurrent_steps=config.recurrent_steps,
            attention_heads=config.attention_heads,
            ff_multiplier=config.ff_multiplier,
            dropout=config.dropout,
        )
        self.cell = TiedWorkspaceCell(cell_config)
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, config.backbone_width, bias=False)
        self.trace_norm = nn.LayerNorm(width)
        self.trace_projection = nn.Linear(width, config.backbone_width, bias=False)
        self.register_buffer(
            "assignments",
            _binary_assignments(bits),
            persistent=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.initial_slots, mean=0.0, std=0.02)
        nn.init.normal_(self.factor_slots, mean=0.0, std=0.02)
        nn.init.zeros_(self.prior_head.bias)

    def _validate_prompt(
        self,
        prompt_features: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if prompt_features.ndim != 3:
            raise LatentTrajectoryError("prompt features must have rank three")
        batch, tokens, width = prompt_features.shape
        if width != self.config.backbone_width:
            raise LatentTrajectoryError("prompt feature width differs")
        if prompt_attention_mask.shape != (batch, tokens):
            raise LatentTrajectoryError("prompt attention mask geometry differs")
        if not torch.isfinite(prompt_features).all():
            raise LatentTrajectoryError("prompt features contain nonfinite values")
        active = prompt_attention_mask.to(dtype=torch.bool)
        if not active.any(dim=1).all():
            raise LatentTrajectoryError("every prompt must contain an active token")
        return active

    def forward(
        self,
        prompt_features: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> LatentTrajectoryOutput:
        active = self._validate_prompt(prompt_features, prompt_attention_mask)
        batch, tokens, _ = prompt_features.shape
        candidates = self.config.candidate_count
        bits = self.config.fault_bits
        slots = self.config.trajectory_slots
        width = self.config.latent_width

        prompt = self.prompt_projection(prompt_features)
        active_float = active.to(dtype=prompt.dtype).unsqueeze(-1)
        summary = (prompt * active_float).sum(dim=1) / active_float.sum(dim=1)
        shared = self.initial_slots.unsqueeze(0) + self.shared_seed(summary).unsqueeze(1)

        dynamic = self.factor_seed(summary).view(batch, bits, 2, width)
        patches = self.factor_slots.unsqueeze(0) + dynamic.unsqueeze(3)
        candidate_states: list[torch.Tensor] = []
        for assignment in self.assignments:
            selected = [
                patches[:, bit, int(assignment[bit].item())]
                for bit in range(bits)
            ]
            candidate_states.append(
                shared + torch.stack(selected, dim=0).sum(dim=0) / math.sqrt(bits)
            )
        state = torch.stack(candidate_states, dim=1)
        if state.shape != (batch, candidates, slots, width):
            raise LatentTrajectoryError("candidate state geometry differs")

        prior_logits = self.prior_head(summary)
        repeated_prompt = (
            prompt.unsqueeze(1)
            .expand(batch, candidates, tokens, width)
            .reshape(batch * candidates, tokens, width)
        )
        repeated_padding = (
            (~active)
            .unsqueeze(1)
            .expand(batch, candidates, tokens)
            .reshape(batch * candidates, tokens)
        )
        flat_state = state.reshape(batch * candidates, slots, width)
        probes: list[torch.Tensor] = []
        stop_logits: list[torch.Tensor] = []
        delta_norms: list[torch.Tensor] = []
        for _ in range(self.config.recurrent_steps):
            flat_state, stop_logit, delta = self.cell(
                flat_state,
                repeated_prompt,
                repeated_padding,
            )
            current = flat_state.view(batch, candidates, slots, width)
            probe = self.trace_projection(self.trace_norm(current.mean(dim=2)))
            probes.append(probe)
            stop_logits.append(stop_logit.view(batch, candidates))
            delta_norms.append(
                delta.square().mean(dim=(1, 2)).sqrt().view(batch, candidates)
            )

        final_state = flat_state.view(batch, candidates, slots, width)
        prefixes = self.output_projection(self.output_norm(final_state))
        return LatentTrajectoryOutput(
            candidate_prefixes=prefixes,
            trajectory_probes=torch.stack(probes, dim=2),
            prior_logits=prior_logits,
            stop_logits=torch.stack(stop_logits, dim=2),
            step_delta_norms=torch.stack(delta_norms, dim=2),
        )

    def select_prefix(
        self,
        output: LatentTrajectoryOutput,
        *,
        strategy: str = "highest_prior",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if strategy == "highest_prior":
            indices = output.prior_logits.argmax(dim=-1)
        elif strategy == "lowest_prior":
            indices = output.prior_logits.argmin(dim=-1)
        elif strategy == "reset":
            indices = output.prior_logits.argmax(dim=-1)
            selected = torch.zeros_like(output.candidate_prefixes[:, 0])
            return selected, indices
        else:
            raise LatentTrajectoryError(f"unknown selection strategy: {strategy}")
        batch_indices = torch.arange(indices.shape[0], device=indices.device)
        return output.candidate_prefixes[batch_indices, indices], indices

    def halting_regularizer(self, output: LatentTrajectoryOutput) -> torch.Tensor:
        probabilities = output.stop_logits.sigmoid()
        monotone = F.relu(probabilities[:, :, :-1] - probabilities[:, :, 1:]).mean()
        final = (1.0 - probabilities[:, :, -1]).mean()
        return monotone + final

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def ordered_trace_targets(
    embedding: nn.Embedding,
    response_rows: Sequence[Sequence[int]],
    recurrent_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached contiguous chunk means in frozen embedding geometry."""

    if recurrent_steps <= 0 or not response_rows:
        raise LatentTrajectoryError("trace target geometry is empty")
    if any(not row for row in response_rows):
        raise LatentTrajectoryError("every response must contain at least one token")
    device = embedding.weight.device
    width = embedding.weight.shape[1]
    targets = torch.zeros(
        len(response_rows), recurrent_steps, width, device=device, dtype=torch.float32
    )
    active = torch.zeros(
        len(response_rows), recurrent_steps, device=device, dtype=torch.bool
    )
    with torch.no_grad():
        for row_index, raw_row in enumerate(response_rows):
            row = list(raw_row)
            chunks = min(recurrent_steps, len(row))
            for chunk in range(chunks):
                start = (chunk * len(row)) // chunks
                end = ((chunk + 1) * len(row)) // chunks
                token_ids = torch.tensor(row[start:end], device=device, dtype=torch.long)
                targets[row_index, chunk] = embedding(token_ids).float().mean(dim=0)
                active[row_index, chunk] = True
    return targets, active


def trajectory_alignment_energy(
    probes: torch.Tensor,
    targets: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Measure ordered semantic distance for every complete trajectory."""

    if probes.ndim != 4:
        raise LatentTrajectoryError("trajectory probes must have rank four")
    batch, _, steps, width = probes.shape
    if targets.shape != (batch, steps, width):
        raise LatentTrajectoryError("trace target geometry differs")
    if active.shape != (batch, steps) or not active.any(dim=1).all():
        raise LatentTrajectoryError("trace mask geometry differs")
    distances = 1.0 - F.cosine_similarity(
        probes.float(),
        targets.float().unsqueeze(1),
        dim=-1,
        eps=1e-6,
    )
    weights = active.to(dtype=distances.dtype).unsqueeze(1)
    return (distances * weights).sum(dim=-1) / weights.sum(dim=-1)


def complete_trajectory_marginal_loss(
    candidate_nll: torch.Tensor,
    trace_energy: torch.Tensor,
    prior_logits: torch.Tensor,
    *,
    trace_weight: float,
    balance_weight: float,
) -> MarginalTrajectoryLoss:
    """Marginalize scalar energies without averaging candidate state fields."""

    if candidate_nll.ndim != 2 or candidate_nll.shape != trace_energy.shape:
        raise LatentTrajectoryError("candidate energy geometry differs")
    if prior_logits.shape != candidate_nll.shape:
        raise LatentTrajectoryError("candidate prior geometry differs")
    if trace_weight < 0.0 or balance_weight < 0.0:
        raise LatentTrajectoryError("loss weights must be nonnegative")
    tensors = (candidate_nll, trace_energy, prior_logits)
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise LatentTrajectoryError("candidate loss contains nonfinite values")

    candidate_energy = candidate_nll + trace_weight * trace_energy
    log_prior = prior_logits.log_softmax(dim=-1)
    log_joint = log_prior - candidate_energy
    marginal_energy = -torch.logsumexp(log_joint, dim=-1).mean()
    posterior = log_joint.softmax(dim=-1)
    mean_posterior = posterior.mean(dim=0)
    candidates = candidate_nll.shape[1]
    balance_loss = (
        mean_posterior
        * (mean_posterior.clamp_min(1e-9).log() + math.log(candidates))
    ).sum()
    loss = marginal_energy + balance_weight * balance_loss
    return MarginalTrajectoryLoss(
        loss=loss,
        marginal_energy=marginal_energy,
        balance_loss=balance_loss,
        posterior=posterior,
        candidate_energy=candidate_energy,
    )


def latent_trajectory_architecture_sha256(config: LatentTrajectoryConfig) -> str:
    config.validate()
    payload = {
        "schema": "shohin-diverge-ltm1-v1",
        "config": asdict(config),
        "mechanism": (
            "factorized-sticky-complete-trajectories+tied-source-conditioned-"
            "recurrence+ordered-trace-alignment+whole-sequence-logsumexp+"
            "single-lineage-map-decode"
        ),
        "field_averaging": False,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
