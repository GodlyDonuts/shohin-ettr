"""Counterfactual evidence energy as a model-owned latent transition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from counterexample_guided_revision import (
    ConsequenceHead,
    RevisionConfig,
    RevisionDynamicsError,
    RevisionInitializer,
)


EquilibriumArm = Literal["energy", "recurrent"]


@dataclass(frozen=True, slots=True)
class EquilibriumStep:
    state: torch.Tensor
    behavior_logits: torch.Tensor
    evidence_energy: torch.Tensor
    energy_gradient_rms: torch.Tensor
    energy_correction_rms: torch.Tensor
    recurrent_correction_rms: torch.Tensor


@dataclass(frozen=True, slots=True)
class EquilibriumTrajectory:
    initial_state: torch.Tensor
    steps: tuple[EquilibriumStep, ...]
    final_state: torch.Tensor
    final_behavior_logits: torch.Tensor
    final_evidence_energy: torch.Tensor


def evidence_energy(
    logits: torch.Tensor, outcomes: torch.Tensor, evidence_mask: torch.Tensor
) -> torch.Tensor:
    if logits.shape[:-1] != outcomes.shape or outcomes.shape != evidence_mask.shape:
        raise RevisionDynamicsError("evidence-energy geometry differs")
    classes = logits.shape[-1]
    per_item = F.cross_entropy(
        logits.reshape(-1, classes), outcomes.reshape(-1), reduction="none"
    ).view_as(outcomes)
    mask = evidence_mask.to(per_item.dtype)
    return (per_item * mask).sum(-1) / mask.sum(-1).clamp_min(1)


class PositiveEnergyPreconditioner(nn.Module):
    """A learned diagonal metric that cannot reverse the energy gradient."""

    def __init__(self, config: RevisionConfig, maximum_step: float = 0.5):
        super().__init__()
        if not 0.0 < maximum_step <= 1.0:
            raise RevisionDynamicsError("maximum energy step is outside (0, 1]")
        self.maximum_step = maximum_step
        self.state_norm = nn.RMSNorm(config.width)
        self.gradient_norm = nn.RMSNorm(config.width)
        self.scale = nn.Sequential(
            nn.Linear(config.width * 2, config.width),
            nn.SiLU(),
            nn.Linear(config.width, config.width),
            nn.Sigmoid(),
        )

    def forward(
        self, state: torch.Tensor, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state.shape != gradient.shape:
            raise RevisionDynamicsError("energy-gradient geometry differs")
        metric = self.maximum_step * self.scale(
            torch.cat((self.state_norm(state), self.gradient_norm(gradient)), -1)
        )
        global_rms = gradient.square().mean((-2, -1), keepdim=True).sqrt()
        normalized = gradient / global_rms.clamp_min(1e-6)
        correction = -metric * normalized
        return state + correction, correction.square().mean((-2, -1)).sqrt()


class DenseEvidenceUpdater(nn.Module):
    """Ordinary learned recurrent update used as the strongest local control."""

    def __init__(self, config: RevisionConfig):
        super().__init__()
        self.outcome_embedding = nn.Embedding(config.outcome_classes, config.width)
        self.challenge_projection = nn.Linear(config.width * 2, config.width)
        self.self_attention = nn.MultiheadAttention(
            config.width, config.heads, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            config.width, config.heads, batch_first=True
        )
        self.update = nn.Sequential(
            nn.RMSNorm(config.width),
            nn.Linear(config.width, config.width * config.ff_multiplier),
            nn.GELU(),
            nn.Linear(config.width * config.ff_multiplier, config.width),
        )
        self.gate = nn.Linear(config.width, config.width)

    def forward(
        self,
        state: torch.Tensor,
        probes: torch.Tensor,
        outcomes: torch.Tensor,
        evidence_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        challenge = self.challenge_projection(
            torch.cat((probes, self.outcome_embedding(outcomes)), -1)
        )
        normalized, _ = self.self_attention(state, state, state, need_weights=False)
        evidence_update, _ = self.cross_attention(
            normalized,
            challenge,
            challenge,
            key_padding_mask=~evidence_mask,
            need_weights=False,
        )
        proposal = evidence_update + self.update(normalized + evidence_update)
        correction = torch.sigmoid(self.gate(proposal)) * proposal
        return state + correction, correction.square().mean((-2, -1)).sqrt()


class CounterfactualEnergyEquilibriumCore(nn.Module):
    """Infer state by descending prompt-owned consequence energy."""

    def __init__(
        self,
        config: RevisionConfig,
        arm: EquilibriumArm = "energy",
        maximum_step: float = 0.5,
    ):
        super().__init__()
        config.validate()
        if arm not in ("energy", "recurrent"):
            raise RevisionDynamicsError("unknown equilibrium arm")
        self.config = config
        self.arm = arm
        self.initializer = RevisionInitializer(config)
        self.consequence = ConsequenceHead(config)
        self.energy_preconditioner = PositiveEnergyPreconditioner(
            config, maximum_step
        )
        self.recurrent_updater = DenseEvidenceUpdater(config)
        self.query_projection = nn.Linear(config.width, config.width, bias=False)
        self.query_attention = nn.MultiheadAttention(
            config.width, config.heads, batch_first=True
        )
        self.answer_head = nn.Linear(config.width, config.answer_classes)

    def deliberate(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        probes: torch.Tensor,
        outcomes: torch.Tensor,
        evidence_mask: torch.Tensor,
        *,
        shuffle_outcomes: bool = False,
        zero_energy_gradient: bool = False,
    ) -> EquilibriumTrajectory:
        if source.shape != probes.shape:
            raise RevisionDynamicsError("source and probe features must align")
        if outcomes.shape != source_mask.shape or evidence_mask.shape != source_mask.shape:
            raise RevisionDynamicsError("evidence geometry differs")
        used_outcomes = outcomes.roll(1, 0) if shuffle_outcomes else outcomes
        initial = self.initializer(source, source_mask)
        state = initial
        steps = []
        for _ in range(self.config.rounds):
            behavior_logits = self.consequence(state, probes)
            energy = evidence_energy(behavior_logits, used_outcomes, evidence_mask)
            gradient = torch.autograd.grad(
                energy.sum(),
                state,
                create_graph=self.training,
                retain_graph=True,
            )[0]
            if zero_energy_gradient:
                gradient = torch.zeros_like(gradient)
            energy_state, energy_correction_rms = self.energy_preconditioner(
                state, gradient
            )
            recurrent_state, recurrent_correction_rms = self.recurrent_updater(
                state, probes, used_outcomes, evidence_mask
            )
            state = energy_state if self.arm == "energy" else recurrent_state
            steps.append(
                EquilibriumStep(
                    state=state,
                    behavior_logits=behavior_logits,
                    evidence_energy=energy,
                    energy_gradient_rms=gradient.square()
                    .mean((-2, -1))
                    .sqrt(),
                    energy_correction_rms=energy_correction_rms,
                    recurrent_correction_rms=recurrent_correction_rms,
                )
            )
        final_behavior_logits = self.consequence(state, probes)
        final_energy = evidence_energy(
            final_behavior_logits, used_outcomes, evidence_mask
        )
        return EquilibriumTrajectory(
            initial_state=initial,
            steps=tuple(steps),
            final_state=state,
            final_behavior_logits=final_behavior_logits,
            final_evidence_energy=final_energy,
        )

    def read_answer(self, state: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        query_state = self.query_projection(query).unsqueeze(1)
        read, _ = self.query_attention(query_state, state, state, need_weights=False)
        return self.answer_head(read.squeeze(1))

    def forward(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        probes: torch.Tensor,
        outcomes: torch.Tensor,
        evidence_mask: torch.Tensor,
        query: torch.Tensor,
        *,
        shuffle_outcomes: bool = False,
        zero_energy_gradient: bool = False,
    ) -> tuple[torch.Tensor, EquilibriumTrajectory]:
        trajectory = self.deliberate(
            source,
            source_mask,
            probes,
            outcomes,
            evidence_mask,
            shuffle_outcomes=shuffle_outcomes,
            zero_energy_gradient=zero_energy_gradient,
        )
        return self.read_answer(trajectory.final_state, query), trajectory
