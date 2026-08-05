"""Prompt-conditioned law induction from finite determining witnesses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


LawArm = Literal["law", "dense"]


class DeterminingLawError(RuntimeError):
    """The determining-law interface violated its fixed geometry."""


@dataclass(frozen=True, slots=True)
class DeterminingLawConfig:
    width: int = 64
    rank: int = 8
    heads: int = 4
    ff_multiplier: int = 2
    outcome_classes: int = 11
    ridge: float = 0.1

    def validate(self) -> None:
        if min(
            self.width,
            self.rank,
            self.heads,
            self.ff_multiplier,
            self.outcome_classes,
        ) <= 0:
            raise DeterminingLawError("law dimensions must be positive")
        if self.width % self.heads:
            raise DeterminingLawError("law width must divide attention heads")
        if self.rank < 2:
            raise DeterminingLawError("law rank must include a constant feature")
        if self.ridge <= 0:
            raise DeterminingLawError("ridge must be positive")


@dataclass(frozen=True, slots=True)
class DeterminingLawOutput:
    selected_logits: torch.Tensor
    law_logits: torch.Tensor
    dense_logits: torch.Tensor
    context_logits: torch.Tensor
    law_coefficients: torch.Tensor
    basis_rms: torch.Tensor
    coefficient_rms: torch.Tensor


class SharedProbeBasis(nn.Module):
    """Map every source probe and late query into the same law basis."""

    def __init__(self, config: DeterminingLawConfig):
        super().__init__()
        self.rank = config.rank
        self.network = nn.Sequential(
            nn.RMSNorm(config.width),
            nn.Linear(config.width, config.width * config.ff_multiplier),
            nn.SiLU(),
            nn.Linear(config.width * config.ff_multiplier, config.rank - 1),
            nn.Tanh(),
        )

    def forward(self, probes: torch.Tensor) -> torch.Tensor:
        learned = self.network(probes)
        constant = torch.ones_like(learned[..., :1])
        return torch.cat((constant, learned), -1)


class DeterminingLawSolver(nn.Module):
    """Solve one reusable class-valued law from prompt-owned witnesses."""

    def __init__(self, config: DeterminingLawConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.basis = SharedProbeBasis(config)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))
        self.class_bias = nn.Parameter(torch.zeros(config.outcome_classes))

    def solve(
        self,
        probes: torch.Tensor,
        outcomes: torch.Tensor,
        evidence_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if probes.ndim != 3 or probes.shape[:2] != outcomes.shape:
            raise DeterminingLawError("probe and outcome geometry differs")
        if evidence_mask.shape != outcomes.shape:
            raise DeterminingLawError("evidence mask geometry differs")
        basis = self.basis(probes).float()
        mask = evidence_mask.to(basis.dtype).unsqueeze(-1)
        weighted = basis * mask
        targets = F.one_hot(
            outcomes, num_classes=self.config.outcome_classes
        ).to(basis.dtype)
        gram = torch.einsum("ber,bes->brs", weighted, basis)
        identity = torch.eye(
            self.config.rank, dtype=gram.dtype, device=gram.device
        ).unsqueeze(0)
        rhs = torch.einsum("ber,bec->brc", weighted, targets)
        coefficients = torch.linalg.solve(gram + self.config.ridge * identity, rhs)
        return basis, coefficients

    def evaluate(
        self, basis: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(max=100.0)
        if basis.ndim == 3:
            logits = torch.einsum("ber,brc->bec", basis.float(), coefficients)
        elif basis.ndim == 2:
            logits = torch.einsum("br,brc->bc", basis.float(), coefficients)
        else:
            raise DeterminingLawError("law basis must be batched probes or queries")
        return logits.mul(scale).add(self.class_bias.float())

    def forward(
        self,
        probes: torch.Tensor,
        outcomes: torch.Tensor,
        evidence_mask: torch.Tensor,
        query: torch.Tensor,
        *,
        destroy_law: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        basis, coefficients = self.solve(probes, outcomes, evidence_mask)
        context_logits = self.evaluate(basis, coefficients)
        query_basis = self.basis(query).float()
        used_coefficients = coefficients.roll(1, 0) if destroy_law else coefficients
        query_logits = self.evaluate(query_basis, used_coefficients)
        return query_logits, context_logits, coefficients, basis


class DenseSetReader(nn.Module):
    """Standard direct set-to-query reader used as the matched control."""

    def __init__(self, config: DeterminingLawConfig):
        super().__init__()
        self.outcome_embedding = nn.Embedding(
            config.outcome_classes, config.width
        )
        self.evidence_projection = nn.Sequential(
            nn.Linear(config.width * 2, config.width),
            nn.SiLU(),
            nn.Linear(config.width, config.width),
        )
        self.query_norm = nn.RMSNorm(config.width)
        self.cross_attention = nn.MultiheadAttention(
            config.width, config.heads, batch_first=True
        )
        self.answer = nn.Sequential(
            nn.RMSNorm(config.width),
            nn.Linear(config.width, config.width * config.ff_multiplier),
            nn.GELU(),
            nn.Linear(config.width * config.ff_multiplier, config.outcome_classes),
        )

    def forward(
        self,
        probes: torch.Tensor,
        outcomes: torch.Tensor,
        evidence_mask: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        evidence = self.evidence_projection(
            torch.cat((probes, self.outcome_embedding(outcomes)), -1)
        )
        read, _ = self.cross_attention(
            self.query_norm(query).unsqueeze(1),
            evidence,
            evidence,
            key_padding_mask=~evidence_mask,
            need_weights=False,
        )
        return self.answer(read.squeeze(1))


class PromptConditionedDeterminingLaw(nn.Module):
    """Execute both law induction and dense reading; select only one output."""

    def __init__(self, config: DeterminingLawConfig, arm: LawArm):
        super().__init__()
        config.validate()
        if arm not in ("law", "dense"):
            raise DeterminingLawError("unknown determining-law arm")
        self.arm = arm
        self.law = DeterminingLawSolver(config)
        self.dense = DenseSetReader(config)

    def forward(
        self,
        probes: torch.Tensor,
        outcomes: torch.Tensor,
        evidence_mask: torch.Tensor,
        query: torch.Tensor,
        *,
        shuffle_outcomes: bool = False,
        destroy_law: bool = False,
    ) -> DeterminingLawOutput:
        used_outcomes = outcomes.roll(1, 0) if shuffle_outcomes else outcomes
        law_logits, context_logits, coefficients, basis = self.law(
            probes,
            used_outcomes,
            evidence_mask,
            query,
            destroy_law=destroy_law,
        )
        dense_logits = self.dense(
            probes, used_outcomes, evidence_mask, query
        )
        selected = law_logits if self.arm == "law" else dense_logits
        return DeterminingLawOutput(
            selected_logits=selected,
            law_logits=law_logits,
            dense_logits=dense_logits,
            context_logits=context_logits,
            law_coefficients=coefficients,
            basis_rms=basis.square().mean((-2, -1)).sqrt(),
            coefficient_rms=coefficients.square().mean((-2, -1)).sqrt(),
        )
