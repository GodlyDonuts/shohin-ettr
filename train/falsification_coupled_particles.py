"""Falsification-coupled whole-hypothesis particle deliberation.

This module implements the architecture mechanics only. Evidence outcomes are
part of the source context; query and answer targets never enter recurrent
proposal or particle-selection APIs. Full FCPT always selects complete states
and preserves lineage. Soft state aggregation exists only as a matched control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


ParticleArm = Literal["fcpt", "independent", "soft", "selection"]


class ParticleDynamicsError(ValueError):
    """A particle tensor or ownership contract was violated."""


@dataclass(frozen=True, slots=True)
class ParticleConfig:
    width: int = 64
    slots: int = 4
    particles: int = 4
    branches: int = 2
    rounds: int = 4
    heads: int = 4
    ff_multiplier: int = 2
    outcome_classes: int = 8
    answer_classes: int = 8
    probes_per_round: int = 2
    temperature: float = 0.5

    def validate(self) -> None:
        dimensions = (
            self.width,
            self.slots,
            self.particles,
            self.branches,
            self.rounds,
            self.heads,
            self.ff_multiplier,
            self.outcome_classes,
            self.answer_classes,
            self.probes_per_round,
        )
        if any(value <= 0 for value in dimensions):
            raise ParticleDynamicsError("all particle dimensions must be positive")
        if self.width % self.heads:
            raise ParticleDynamicsError("particle width must divide across heads")
        if self.temperature <= 0:
            raise ParticleDynamicsError("particle temperature must be positive")


@dataclass(frozen=True, slots=True)
class ParticlePopulation:
    state: torch.Tensor
    log_weight: torch.Tensor
    lineage: torch.Tensor


@dataclass(frozen=True, slots=True)
class ParticleRound:
    state: torch.Tensor
    log_weight: torch.Tensor
    lineage: torch.Tensor
    selected_candidate: torch.Tensor
    selected_probe: torch.Tensor
    disagreement: torch.Tensor
    behavior_logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class ParticleTrajectory:
    initial: ParticlePopulation
    rounds: tuple[ParticleRound, ...]
    final_state: torch.Tensor
    final_lineage: torch.Tensor
    final_log_weight: torch.Tensor


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise ParticleDynamicsError("masked source geometry differs")
    if mask.dtype != torch.bool or (~mask.any(-1)).any():
        raise ParticleDynamicsError("every source row needs valid evidence")
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(1) / weights.sum(1)


def gather_complete_states(states: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather whole states along the candidate axis without field mixing."""

    if states.ndim != 4 or indices.ndim != 2 or states.shape[0] != indices.shape[0]:
        raise ParticleDynamicsError("whole-state gather geometry differs")
    slots, width = states.shape[-2:]
    expanded = indices[..., None, None].expand(-1, -1, slots, width)
    return torch.gather(states, 1, expanded)


def gather_candidates(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if values.ndim < 2 or indices.ndim != 2 or values.shape[0] != indices.shape[0]:
        raise ParticleDynamicsError("candidate gather geometry differs")
    expansion = indices[(...,) + (None,) * (values.ndim - 2)].expand(
        -1, -1, *values.shape[2:]
    )
    return torch.gather(values, 1, expansion)


class ParticleInitializer(nn.Module):
    def __init__(self, config: ParticleConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.source_norm = nn.RMSNorm(config.width)
        self.slot_seed = nn.Parameter(torch.empty(config.slots, config.width))
        self.particle_seed = nn.Parameter(
            torch.empty(config.particles, config.slots, config.width)
        )
        self.source_projection = nn.Linear(config.width, config.width, bias=False)
        nn.init.normal_(self.slot_seed, std=0.02)
        nn.init.normal_(self.particle_seed, std=0.02)

    def forward(
        self, source_features: torch.Tensor, source_mask: torch.Tensor
    ) -> ParticlePopulation:
        summary = self.source_projection(
            self.source_norm(_masked_mean(source_features, source_mask))
        )
        state = (
            self.slot_seed[None, None]
            + self.particle_seed[None]
            + summary[:, None, None]
        )
        batch = source_features.shape[0]
        return ParticlePopulation(
            state=state,
            log_weight=torch.zeros(
                batch,
                self.config.particles,
                device=state.device,
                dtype=state.dtype,
            ),
            lineage=torch.arange(
                self.config.particles, device=state.device, dtype=torch.long
            )[None].expand(batch, -1),
        )


class SharedBranchProposal(nn.Module):
    """One proposal operator shared across particles, branches, and rounds."""

    def __init__(self, config: ParticleConfig):
        super().__init__()
        self.config = config
        self.branch_seed = nn.Parameter(torch.empty(config.branches, config.width))
        self.state_norm = nn.RMSNorm(config.width)
        self.attention = nn.MultiheadAttention(
            config.width, config.heads, batch_first=True
        )
        self.context_projection = nn.Linear(config.width, config.width, bias=False)
        self.update = nn.Sequential(
            nn.RMSNorm(config.width),
            nn.Linear(config.width, config.width * config.ff_multiplier),
            nn.GELU(),
            nn.Linear(config.width * config.ff_multiplier, config.width),
        )
        self.gate = nn.Linear(config.width, config.width)
        nn.init.normal_(self.branch_seed, std=0.02)

    def forward(
        self,
        state: torch.Tensor,
        source_summary: torch.Tensor,
        challenge: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim != 4 or source_summary.shape != challenge.shape:
            raise ParticleDynamicsError("proposal geometry differs")
        batch, particles, slots, width = state.shape
        flat = state.flatten(0, 1)
        normalized = self.state_norm(flat)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        context = self.context_projection(source_summary + challenge)
        base = flat + attended + context[:, None].repeat_interleave(particles, 0)
        branch_input = (
            base[:, None] + self.branch_seed[None, :, None, :]
        ).flatten(0, 1)
        proposal = branch_input + torch.sigmoid(self.gate(branch_input)) * self.update(
            branch_input
        )
        return proposal.view(batch, particles, self.config.branches, slots, width)


class BehavioralPredictionHead(nn.Module):
    def __init__(self, config: ParticleConfig):
        super().__init__()
        self.state_projection = nn.Linear(config.width, config.width, bias=False)
        self.probe_projection = nn.Linear(config.width, config.width, bias=False)
        self.output = nn.Linear(config.width, config.outcome_classes)

    def forward(self, states: torch.Tensor, probes: torch.Tensor) -> torch.Tensor:
        if states.ndim != 4 or probes.ndim != 3 or states.shape[0] != probes.shape[0]:
            raise ParticleDynamicsError("behavior prediction geometry differs")
        summary = self.state_projection(states.mean(-2))
        probe = self.probe_projection(probes)
        joint = torch.tanh(summary[:, :, None] + probe[:, None])
        return self.output(joint)


class StructuredContradictionBus(nn.Module):
    """Admit a fixed number of high-disagreement source evidence probes."""

    def __init__(self, config: ParticleConfig):
        super().__init__()
        self.config = config
        self.probe_prior = nn.Sequential(
            nn.RMSNorm(config.width), nn.Linear(config.width, 1)
        )
        self.outcome_embedding = nn.Embedding(config.outcome_classes, config.width)
        self.message = nn.Linear(config.width * 2, config.width)

    def select(
        self,
        behavior_logits: torch.Tensor,
        probe_features: torch.Tensor,
        evidence_mask: torch.Tensor,
        *,
        use_disagreement: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if behavior_logits.ndim != 4:
            raise ParticleDynamicsError("behavior logits rank differs")
        probabilities = behavior_logits.softmax(-1)
        disagreement = probabilities.var(1, unbiased=False).sum(-1)
        prior = self.probe_prior(probe_features).squeeze(-1)
        if use_disagreement:
            score = disagreement.clamp_min(1e-8).log() + 0.1 * prior
        else:
            positions = torch.arange(
                evidence_mask.shape[1],
                device=evidence_mask.device,
                dtype=probe_features.dtype,
            )
            score = -positions[None].expand_as(prior)
        score = score.masked_fill(~evidence_mask, torch.finfo(score.dtype).min)
        count = min(self.config.probes_per_round, score.shape[1])
        selected = score.topk(count, dim=1).indices
        return selected, disagreement

    def challenge(
        self,
        probe_features: torch.Tensor,
        evidence_outcomes: torch.Tensor,
        selected_probe: torch.Tensor,
    ) -> torch.Tensor:
        width = probe_features.shape[-1]
        selected_features = torch.gather(
            probe_features,
            1,
            selected_probe[..., None].expand(-1, -1, width),
        )
        selected_outcomes = torch.gather(evidence_outcomes, 1, selected_probe)
        embedded_outcomes = self.outcome_embedding(selected_outcomes)
        return self.message(
            torch.cat((selected_features, embedded_outcomes), -1)
        ).mean(1)


def evidence_log_likelihood(
    behavior_logits: torch.Tensor,
    evidence_outcomes: torch.Tensor,
    selected_probe: torch.Tensor,
) -> torch.Tensor:
    if behavior_logits.ndim != 4:
        raise ParticleDynamicsError("evidence score rank differs")
    batch, candidates, _, _ = behavior_logits.shape
    log_probability = behavior_logits.log_softmax(-1)
    labels = evidence_outcomes[:, None, :, None].expand(batch, candidates, -1, 1)
    per_probe = torch.gather(log_probability, -1, labels).squeeze(-1)
    admitted = torch.zeros_like(per_probe, dtype=torch.bool)
    admitted.scatter_(
        2, selected_probe[:, None].expand(batch, candidates, -1), True
    )
    return (per_probe * admitted).sum(-1) / admitted.sum(-1).clamp_min(1)


class FalsificationCoupledParticleCore(nn.Module):
    """Shared particle proposal plus arm-specific evidence/selection dynamics."""

    def __init__(self, config: ParticleConfig, arm: ParticleArm = "fcpt"):
        super().__init__()
        config.validate()
        if arm not in ("fcpt", "independent", "soft", "selection"):
            raise ParticleDynamicsError("unknown particle arm")
        self.config = config
        self.arm = arm
        self.initializer = ParticleInitializer(config)
        self.proposer = SharedBranchProposal(config)
        self.behavior = BehavioralPredictionHead(config)
        self.bus = StructuredContradictionBus(config)
        self.query_projection = nn.Linear(config.width, config.width, bias=False)
        self.query_attention = nn.MultiheadAttention(
            config.width, config.heads, batch_first=True
        )
        self.answer_head = nn.Linear(config.width, config.answer_classes)

    def _whole_select(
        self,
        candidate_state: torch.Tensor,
        candidate_score: torch.Tensor,
        candidate_lineage: torch.Tensor,
        count: int,
    ) -> tuple[ParticlePopulation, torch.Tensor]:
        selected = candidate_score.topk(count, dim=1).indices
        return (
            ParticlePopulation(
                state=gather_complete_states(candidate_state, selected),
                log_weight=torch.gather(candidate_score, 1, selected),
                lineage=torch.gather(candidate_lineage, 1, selected),
            ),
            selected,
        )

    def deliberate(
        self,
        source_features: torch.Tensor,
        source_mask: torch.Tensor,
        probe_features: torch.Tensor,
        evidence_outcomes: torch.Tensor,
        evidence_mask: torch.Tensor,
    ) -> ParticleTrajectory:
        if source_features.shape != probe_features.shape:
            raise ParticleDynamicsError("source and probe features must align")
        if evidence_outcomes.shape != source_mask.shape or evidence_mask.shape != source_mask.shape:
            raise ParticleDynamicsError("evidence geometry differs")
        if evidence_outcomes.min() < 0 or evidence_outcomes.max() >= self.config.outcome_classes:
            raise ParticleDynamicsError("evidence outcome exceeds vocabulary")

        initial = self.initializer(source_features, source_mask)
        population = initial
        source_summary = _masked_mean(source_features, source_mask)
        challenge = torch.zeros_like(source_summary)
        rounds = []
        for _ in range(self.config.rounds):
            proposed = self.proposer(population.state, source_summary, challenge)
            batch, particles, branches, slots, width = proposed.shape
            candidate_state = proposed.view(
                batch, particles * branches, slots, width
            )
            behavior_logits = self.behavior(candidate_state, probe_features)
            selected_probe, disagreement = self.bus.select(
                behavior_logits,
                probe_features,
                evidence_mask,
                use_disagreement=self.arm == "fcpt",
            )
            challenge = self.bus.challenge(
                probe_features, evidence_outcomes, selected_probe
            )
            score = evidence_log_likelihood(
                behavior_logits, evidence_outcomes, selected_probe
            )
            score = score + population.log_weight[:, :, None].expand(
                -1, -1, branches
            ).reshape(batch, particles * branches)
            lineage = (
                population.lineage[:, :, None] * branches
                + torch.arange(branches, device=proposed.device)[None, None]
            ).reshape(batch, particles * branches)

            if self.arm == "soft":
                mixture = torch.einsum(
                    "bn,bnsd->bsd",
                    (score / self.config.temperature).softmax(-1),
                    candidate_state,
                )
                state = mixture[:, None].expand(-1, particles, -1, -1)
                population = ParticlePopulation(
                    state=state,
                    log_weight=torch.zeros_like(population.log_weight),
                    lineage=torch.full_like(population.lineage, -1),
                )
                selected = torch.full_like(population.lineage, -1)
            elif self.arm == "independent":
                per_parent_score = score.view(batch, particles, branches)
                branch = per_parent_score.argmax(-1)
                selected = (
                    torch.arange(particles, device=proposed.device)[None] * branches
                    + branch
                )
                population = ParticlePopulation(
                    state=gather_complete_states(candidate_state, selected),
                    log_weight=torch.gather(score, 1, selected),
                    lineage=torch.gather(lineage, 1, selected),
                )
                challenge = torch.zeros_like(challenge)
            else:
                population, selected = self._whole_select(
                    candidate_state,
                    score,
                    lineage,
                    self.config.particles,
                )

            rounds.append(
                ParticleRound(
                    state=population.state,
                    log_weight=population.log_weight,
                    lineage=population.lineage,
                    selected_candidate=selected,
                    selected_probe=selected_probe,
                    disagreement=disagreement,
                    behavior_logits=behavior_logits,
                )
            )

        if self.arm == "soft":
            final_state = population.state[:, 0]
            final_lineage = population.lineage[:, 0]
            final_weight = population.log_weight[:, 0]
        else:
            winner = population.log_weight.argmax(-1, keepdim=True)
            final_state = gather_complete_states(population.state, winner).squeeze(1)
            final_lineage = torch.gather(population.lineage, 1, winner).squeeze(1)
            final_weight = torch.gather(population.log_weight, 1, winner).squeeze(1)
        return ParticleTrajectory(
            initial=initial,
            rounds=tuple(rounds),
            final_state=final_state,
            final_lineage=final_lineage,
            final_log_weight=final_weight,
        )

    def read_answer(
        self, final_state: torch.Tensor, query_features: torch.Tensor
    ) -> torch.Tensor:
        if final_state.ndim != 3 or query_features.ndim != 2:
            raise ParticleDynamicsError("late query geometry differs")
        query = self.query_projection(query_features).unsqueeze(1)
        read, _ = self.query_attention(query, final_state, final_state, need_weights=False)
        return self.answer_head(read.squeeze(1))

    def forward(
        self,
        source_features: torch.Tensor,
        source_mask: torch.Tensor,
        probe_features: torch.Tensor,
        evidence_outcomes: torch.Tensor,
        evidence_mask: torch.Tensor,
        query_features: torch.Tensor,
    ) -> tuple[torch.Tensor, ParticleTrajectory]:
        trajectory = self.deliberate(
            source_features,
            source_mask,
            probe_features,
            evidence_outcomes,
            evidence_mask,
        )
        return self.read_answer(trajectory.final_state, query_features), trajectory


def behaviorally_equivalent(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    probe_mask: torch.Tensor,
) -> torch.Tensor:
    """Certify equivalence only by predicted consequences on admitted probes."""

    if first_logits.shape != second_logits.shape or first_logits.ndim != 3:
        raise ParticleDynamicsError("behavioral equivalence geometry differs")
    if probe_mask.shape != first_logits.shape[:2] or probe_mask.dtype != torch.bool:
        raise ParticleDynamicsError("behavioral equivalence mask differs")
    equal = first_logits.argmax(-1).eq(second_logits.argmax(-1)) | ~probe_mask
    return equal.all(-1)
