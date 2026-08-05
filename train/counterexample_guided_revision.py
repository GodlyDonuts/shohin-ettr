"""Counterexample-guided sparse global revision over one coherent state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn


RevisionArm = Literal["guided", "fixed", "dense"]


class RevisionDynamicsError(ValueError):
    """The revision state or evidence ownership contract was violated."""


@dataclass(frozen=True, slots=True)
class RevisionConfig:
    width: int = 64
    slots: int = 8
    rounds: int = 4
    heads: int = 4
    ff_multiplier: int = 2
    outcome_classes: int = 11
    answer_classes: int = 11
    probes_per_round: int = 2
    revision_slots: int = 2

    def validate(self) -> None:
        dimensions = (
            self.width,
            self.slots,
            self.rounds,
            self.heads,
            self.ff_multiplier,
            self.outcome_classes,
            self.answer_classes,
            self.probes_per_round,
            self.revision_slots,
        )
        if any(value <= 0 for value in dimensions):
            raise RevisionDynamicsError("all revision dimensions must be positive")
        if self.width % self.heads:
            raise RevisionDynamicsError("revision width must divide across heads")
        if self.revision_slots > self.slots:
            raise RevisionDynamicsError("revision slot count exceeds state slots")


@dataclass(frozen=True, slots=True)
class RevisionStep:
    state: torch.Tensor
    behavior_logits: torch.Tensor
    selected_probe: torch.Tensor
    contradiction: torch.Tensor
    slot_mask: torch.Tensor
    correction_rms: torch.Tensor


@dataclass(frozen=True, slots=True)
class RevisionTrajectory:
    initial_state: torch.Tensor
    steps: tuple[RevisionStep, ...]
    final_state: torch.Tensor


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise RevisionDynamicsError("source geometry differs")
    if mask.dtype != torch.bool or (~mask.any(-1)).any():
        raise RevisionDynamicsError("every source row needs visible evidence")
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(1) / weights.sum(1)


class RevisionInitializer(nn.Module):
    def __init__(self, config: RevisionConfig):
        super().__init__()
        self.slot_seed = nn.Parameter(torch.empty(config.slots, config.width))
        self.source_projection = nn.Linear(config.width, config.width, bias=False)
        self.norm = nn.RMSNorm(config.width)
        nn.init.normal_(self.slot_seed, std=0.02)

    def forward(self, source: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        summary = self.source_projection(self.norm(_masked_mean(source, source_mask)))
        return self.slot_seed[None] + summary[:, None]


class ConsequenceHead(nn.Module):
    def __init__(self, config: RevisionConfig):
        super().__init__()
        self.state_projection = nn.Linear(config.width, config.width, bias=False)
        self.probe_projection = nn.Linear(config.width, config.width, bias=False)
        self.output = nn.Linear(config.width, config.outcome_classes)

    def forward(self, state: torch.Tensor, probes: torch.Tensor) -> torch.Tensor:
        if state.ndim != 3 or probes.ndim != 3 or state.shape[0] != probes.shape[0]:
            raise RevisionDynamicsError("consequence geometry differs")
        state_summary = self.state_projection(state.mean(1))
        probe = self.probe_projection(probes)
        return self.output(torch.tanh(state_summary[:, None] + probe))


class SparseRevisionOperator(nn.Module):
    def __init__(self, config: RevisionConfig):
        super().__init__()
        self.config = config
        self.outcome_embedding = nn.Embedding(config.outcome_classes, config.width)
        self.challenge_projection = nn.Linear(config.width * 2 + 1, config.width)
        self.slot_key = nn.Linear(config.width, config.width, bias=False)
        self.challenge_key = nn.Linear(config.width, config.width, bias=False)
        self.cross_attention = nn.MultiheadAttention(
            config.width, config.heads, batch_first=True
        )
        self.self_attention = nn.MultiheadAttention(
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
        contradiction: torch.Tensor,
        selected_probe: torch.Tensor,
        *,
        dense: bool,
        shuffle_outcomes: bool,
        selection_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, width = probes.shape
        selected_features = torch.gather(
            probes,
            1,
            selected_probe[..., None].expand(-1, -1, width),
        )
        selected_outcomes = torch.gather(outcomes, 1, selected_probe)
        if shuffle_outcomes:
            selected_outcomes = selected_outcomes.roll(1, 0)
        selected_residual = torch.gather(contradiction, 1, selected_probe)
        challenge = self.challenge_projection(
            torch.cat(
                (
                    selected_features,
                    self.outcome_embedding(selected_outcomes),
                    selected_residual.unsqueeze(-1),
                ),
                -1,
            )
        )
        if selection_weight is not None:
            if selection_weight.shape != selected_probe.shape:
                raise RevisionDynamicsError("selection-weight geometry differs")
            challenge = challenge * selection_weight.unsqueeze(-1)
        affinity = torch.einsum(
            "bsd,bpd->bsp",
            self.slot_key(state),
            self.challenge_key(challenge),
        ).amax(-1)
        if dense:
            slot_mask = torch.ones_like(affinity, dtype=torch.bool)
        else:
            selected_slots = affinity.topk(self.config.revision_slots, dim=1).indices
            slot_mask = torch.zeros_like(affinity, dtype=torch.bool)
            slot_mask.scatter_(1, selected_slots, True)
        normalized, _ = self.self_attention(state, state, state, need_weights=False)
        evidence_update, _ = self.cross_attention(
            normalized, challenge, challenge, need_weights=False
        )
        proposal = evidence_update + self.update(normalized + evidence_update)
        correction = torch.sigmoid(self.gate(proposal)) * proposal
        correction = correction * slot_mask.unsqueeze(-1)
        next_state = state + correction
        correction_rms = correction.square().mean((-2, -1)).sqrt()
        return next_state, slot_mask, correction_rms


class CounterexampleGuidedRevisionCore(nn.Module):
    """Predict, find a source-owned counterexample, and revise sparse slots."""

    def __init__(self, config: RevisionConfig, arm: RevisionArm = "guided"):
        super().__init__()
        config.validate()
        if arm not in ("guided", "fixed", "dense"):
            raise RevisionDynamicsError("unknown revision arm")
        self.config = config
        self.arm = arm
        self.initializer = RevisionInitializer(config)
        self.consequence = ConsequenceHead(config)
        self.revision = SparseRevisionOperator(config)
        self.query_projection = nn.Linear(config.width, config.width, bias=False)
        self.query_attention = nn.MultiheadAttention(
            config.width, config.heads, batch_first=True
        )
        self.answer_head = nn.Linear(config.width, config.answer_classes)

    def _select_probe(
        self,
        contradiction: torch.Tensor,
        evidence_mask: torch.Tensor,
        round_index: int,
    ) -> torch.Tensor:
        if self.arm in ("guided", "dense"):
            score = contradiction
        else:
            positions = torch.arange(
                evidence_mask.shape[1],
                device=evidence_mask.device,
                dtype=contradiction.dtype,
            )
            offset = (round_index * self.config.probes_per_round) % positions.numel()
            score = -((positions - offset) % positions.numel())[None].expand_as(
                contradiction
            )
        score = score.masked_fill(~evidence_mask, torch.finfo(score.dtype).min)
        count = min(self.config.probes_per_round, score.shape[1])
        return score.topk(count, dim=1).indices

    def deliberate(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        probes: torch.Tensor,
        outcomes: torch.Tensor,
        evidence_mask: torch.Tensor,
        *,
        shuffle_outcomes: bool = False,
    ) -> RevisionTrajectory:
        if source.shape != probes.shape:
            raise RevisionDynamicsError("source and probe features must align")
        if outcomes.shape != source_mask.shape or evidence_mask.shape != source_mask.shape:
            raise RevisionDynamicsError("evidence geometry differs")
        initial = self.initializer(source, source_mask)
        state = initial
        steps = []
        for round_index in range(self.config.rounds):
            behavior_logits = self.consequence(state, probes)
            log_probability = behavior_logits.log_softmax(-1)
            contradiction = -torch.gather(
                log_probability, -1, outcomes.unsqueeze(-1)
            ).squeeze(-1)
            selected_probe = self._select_probe(
                contradiction, evidence_mask, round_index
            )
            state, slot_mask, correction_rms = self.revision(
                state,
                probes,
                outcomes,
                contradiction,
                selected_probe,
                dense=self.arm == "dense",
                shuffle_outcomes=shuffle_outcomes,
            )
            steps.append(
                RevisionStep(
                    state=state,
                    behavior_logits=behavior_logits,
                    selected_probe=selected_probe,
                    contradiction=contradiction,
                    slot_mask=slot_mask,
                    correction_rms=correction_rms,
                )
            )
        return RevisionTrajectory(
            initial_state=initial,
            steps=tuple(steps),
            final_state=state,
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
    ) -> tuple[torch.Tensor, RevisionTrajectory]:
        trajectory = self.deliberate(
            source,
            source_mask,
            probes,
            outcomes,
            evidence_mask,
            shuffle_outcomes=shuffle_outcomes,
        )
        return self.read_answer(trajectory.final_state, query), trajectory
