"""Query-valued sparse revision over one coherent recurrent state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from counterexample_guided_revision import (
    ConsequenceHead,
    RevisionConfig,
    RevisionDynamicsError,
    RevisionInitializer,
    SparseRevisionOperator,
)


ValueArm = Literal["utility", "fixed", "residual"]


@dataclass(frozen=True, slots=True)
class ValueRevisionStep:
    state: torch.Tensor
    behavior_logits: torch.Tensor
    selected_probe: torch.Tensor
    contradiction: torch.Tensor
    utility_logits: torch.Tensor
    utility_probability: torch.Tensor
    slot_mask: torch.Tensor
    correction_rms: torch.Tensor


@dataclass(frozen=True, slots=True)
class ValueRevisionTrajectory:
    initial_state: torch.Tensor
    steps: tuple[ValueRevisionStep, ...]
    final_state: torch.Tensor


class ValueOfEvidenceSelector(nn.Module):
    """Estimate final-answer value for each prompt-owned evidence item."""

    def __init__(self, config: RevisionConfig):
        super().__init__()
        self.state_norm = nn.RMSNorm(config.width)
        self.state_projection = nn.Linear(config.width, config.width, bias=False)
        self.probe_projection = nn.Linear(config.width, config.width, bias=False)
        self.query_projection = nn.Linear(config.width, config.width, bias=False)
        self.contradiction_projection = nn.Linear(1, config.width, bias=False)
        self.novelty_projection = nn.Linear(1, config.width, bias=False)
        self.output = nn.Linear(config.width, 1)

    def forward(
        self,
        state: torch.Tensor,
        probes: torch.Tensor,
        query: torch.Tensor,
        contradiction: torch.Tensor,
        visit_count: torch.Tensor,
        evidence_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.shape != (state.shape[0], state.shape[-1]):
            raise RevisionDynamicsError("selector query geometry differs")
        state_context = self.state_projection(self.state_norm(state).mean(1))
        novelty = visit_count.add(1).reciprocal().unsqueeze(-1)
        hidden = torch.tanh(
            state_context[:, None]
            + self.probe_projection(probes)
            + self.query_projection(query)[:, None]
            + self.contradiction_projection(contradiction.unsqueeze(-1))
            + self.novelty_projection(novelty)
        )
        logits = self.output(hidden).squeeze(-1)
        logits = logits.masked_fill(~evidence_mask, torch.finfo(logits.dtype).min)
        probability = logits.softmax(-1)
        return logits, probability


class QueryValuedRevisionCore(nn.Module):
    """Learn which source evidence has value for the requested answer."""

    def __init__(self, config: RevisionConfig, arm: ValueArm = "utility"):
        super().__init__()
        config.validate()
        if arm not in ("utility", "fixed", "residual"):
            raise RevisionDynamicsError("unknown value-revision arm")
        self.config = config
        self.arm = arm
        self.initializer = RevisionInitializer(config)
        self.consequence = ConsequenceHead(config)
        self.selector = ValueOfEvidenceSelector(config)
        self.revision = SparseRevisionOperator(config)
        self.query_projection = nn.Linear(config.width, config.width, bias=False)
        self.query_attention = nn.MultiheadAttention(
            config.width, config.heads, batch_first=True
        )
        self.answer_head = nn.Linear(config.width, config.answer_classes)

    def _select_probe(
        self,
        utility_logits: torch.Tensor,
        contradiction: torch.Tensor,
        evidence_mask: torch.Tensor,
        round_index: int,
    ) -> torch.Tensor:
        if self.arm == "utility":
            score = utility_logits
        elif self.arm == "residual":
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

    @staticmethod
    def _straight_through_weight(
        probability: torch.Tensor, selected_probe: torch.Tensor
    ) -> torch.Tensor:
        hard = torch.zeros_like(probability)
        hard.scatter_(1, selected_probe, 1.0)
        straight_through = hard + probability - probability.detach()
        return torch.gather(straight_through, 1, selected_probe)

    def deliberate(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        probes: torch.Tensor,
        outcomes: torch.Tensor,
        evidence_mask: torch.Tensor,
        query: torch.Tensor,
        *,
        shuffle_outcomes: bool = False,
        shuffle_selector_query: bool = False,
    ) -> ValueRevisionTrajectory:
        if source.shape != probes.shape:
            raise RevisionDynamicsError("source and probe features must align")
        if outcomes.shape != source_mask.shape or evidence_mask.shape != source_mask.shape:
            raise RevisionDynamicsError("evidence geometry differs")
        selector_query = query.roll(1, 0) if shuffle_selector_query else query
        initial = self.initializer(source, source_mask)
        state = initial
        visit_count = torch.zeros_like(outcomes, dtype=source.dtype)
        steps = []
        for round_index in range(self.config.rounds):
            behavior_logits = self.consequence(state, probes)
            log_probability = behavior_logits.log_softmax(-1)
            contradiction = -torch.gather(
                log_probability, -1, outcomes.unsqueeze(-1)
            ).squeeze(-1)
            utility_logits, utility_probability = self.selector(
                state,
                probes,
                selector_query,
                contradiction,
                visit_count,
                evidence_mask,
            )
            selected_probe = self._select_probe(
                utility_logits,
                contradiction,
                evidence_mask,
                round_index,
            )
            selection_weight = self._straight_through_weight(
                utility_probability, selected_probe
            )
            state, slot_mask, correction_rms = self.revision(
                state,
                probes,
                outcomes,
                contradiction,
                selected_probe,
                dense=False,
                shuffle_outcomes=shuffle_outcomes,
                selection_weight=selection_weight,
            )
            visit_count.scatter_add_(
                1,
                selected_probe,
                torch.ones_like(selected_probe, dtype=visit_count.dtype),
            )
            steps.append(
                ValueRevisionStep(
                    state=state,
                    behavior_logits=behavior_logits,
                    selected_probe=selected_probe,
                    contradiction=contradiction,
                    utility_logits=utility_logits,
                    utility_probability=utility_probability,
                    slot_mask=slot_mask,
                    correction_rms=correction_rms,
                )
            )
        return ValueRevisionTrajectory(
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
        shuffle_selector_query: bool = False,
    ) -> tuple[torch.Tensor, ValueRevisionTrajectory]:
        trajectory = self.deliberate(
            source,
            source_mask,
            probes,
            outcomes,
            evidence_mask,
            query,
            shuffle_outcomes=shuffle_outcomes,
            shuffle_selector_query=shuffle_selector_query,
        )
        return self.read_answer(trajectory.final_state, query), trajectory
