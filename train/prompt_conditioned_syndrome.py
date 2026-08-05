"""Prompt-conditioned syndrome correction for model-owned latent reasoning.

The source compiles one sticky parity geometry. A tied recurrent proposer may
then update the workspace, but every commit is projected back onto the affine
constraint manifold defined by the initial workspace syndrome. Query and
answer tensors are deliberately absent from the transition API.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn


class SyndromeDynamicsError(ValueError):
    """The syndrome workspace violated a tensor or ownership contract."""


@dataclass(frozen=True, slots=True)
class SyndromeConfig:
    input_width: int = 128
    state_width: int = 128
    slots: int = 16
    checks: int = 4
    heads: int = 4
    steps: int = 8
    min_steps: int = 1
    ff_multiplier: int = 4
    epsilon: float = 1e-4
    use_step_embedding: bool = True

    def validate(self) -> None:
        dimensions = (
            self.input_width,
            self.state_width,
            self.slots,
            self.checks,
            self.heads,
            self.steps,
            self.ff_multiplier,
        )
        if any(value <= 0 for value in dimensions):
            raise SyndromeDynamicsError("all syndrome dimensions must be positive")
        if self.checks > min(self.slots, self.state_width):
            raise SyndromeDynamicsError("checks exceed factorized state rank")
        if self.state_width % self.heads:
            raise SyndromeDynamicsError("state width must divide evenly across heads")
        if not 0 <= self.min_steps <= self.steps:
            raise SyndromeDynamicsError("minimum steps differ")
        if self.epsilon <= 0:
            raise SyndromeDynamicsError("epsilon must be positive")


@dataclass(frozen=True, slots=True)
class StickyChecks:
    slot_factors: torch.Tensor
    feature_factors: torch.Tensor
    reference_syndrome: torch.Tensor


@dataclass(frozen=True, slots=True)
class SyndromeStep:
    state: torch.Tensor
    pre_syndrome_rms: torch.Tensor
    post_syndrome_rms: torch.Tensor
    correction_rms: torch.Tensor
    gate_mean: torch.Tensor
    stop_probability: torch.Tensor
    alive_before: torch.Tensor


@dataclass(frozen=True, slots=True)
class SyndromeTrajectory:
    initial_state: torch.Tensor
    final_state: torch.Tensor
    checks: StickyChecks
    steps: tuple[SyndromeStep, ...]
    stop_step: torch.Tensor


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or mask.shape != values.shape[:2]:
        raise SyndromeDynamicsError("source feature or mask geometry differs")
    if mask.dtype != torch.bool:
        raise SyndromeDynamicsError("source mask must be boolean")
    if (~mask.any(-1)).any():
        raise SyndromeDynamicsError("every source row must expose one token")
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(1) / weights.sum(1)


def _orthonormal_rows(values: torch.Tensor) -> torch.Tensor:
    """Return orthonormal rows while retaining gradients through QR."""

    if values.ndim not in (2, 3):
        raise SyndromeDynamicsError("orthonormal input rank differs")
    rows, columns = values.shape[-2:]
    if rows > columns:
        raise SyndromeDynamicsError("cannot orthonormalize more rows than columns")
    q, _ = torch.linalg.qr(values.transpose(-2, -1).float(), mode="reduced")
    return q.transpose(-2, -1).to(values.dtype)


def syndrome(
    state: torch.Tensor,
    slot_factors: torch.Tensor,
    feature_factors: torch.Tensor,
) -> torch.Tensor:
    if state.ndim != 3 or slot_factors.ndim != 3 or feature_factors.ndim != 2:
        raise SyndromeDynamicsError("syndrome tensor rank differs")
    batch, slots, width = state.shape
    if slot_factors.shape[0] != batch or slot_factors.shape[2] != slots:
        raise SyndromeDynamicsError("slot check geometry differs")
    if feature_factors.shape != (slot_factors.shape[1], width):
        raise SyndromeDynamicsError("feature check geometry differs")
    return torch.einsum("bcs,cd,bsd->bc", slot_factors, feature_factors, state)


class PromptConditionedCheckCompiler(nn.Module):
    """Compile one source-owned, sticky factorized parity geometry."""

    def __init__(self, config: SyndromeConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.source_norm = nn.RMSNorm(config.input_width)
        self.slot_logits = nn.Linear(
            config.input_width,
            config.checks * config.slots,
        )
        self.feature_logits = nn.Parameter(
            torch.empty(config.checks, config.state_width)
        )
        nn.init.normal_(self.feature_logits, std=0.02)

    def factors(
        self,
        source_features: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        summary = self.source_norm(_masked_mean(source_features, source_mask))
        raw_slots = self.slot_logits(summary).view(
            summary.shape[0], self.config.checks, self.config.slots
        )
        return _orthonormal_rows(raw_slots), _orthonormal_rows(self.feature_logits)

    def forward(
        self,
        source_features: torch.Tensor,
        source_mask: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> StickyChecks:
        slots, features = self.factors(source_features, source_mask)
        return StickyChecks(
            slot_factors=slots,
            feature_factors=features,
            reference_syndrome=syndrome(initial_state, slots, features),
        )


class MinimumNormSyndromeProjector(nn.Module):
    """Project a proposed state onto the prompt's affine check manifold."""

    def __init__(self, config: SyndromeConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.correction_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        proposed_state: torch.Tensor,
        checks: StickyChecks,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        observed = syndrome(
            proposed_state,
            checks.slot_factors,
            checks.feature_factors,
        )
        error = observed - checks.reference_syndrome
        slot_gram = torch.einsum(
            "bcs,bds->bcd", checks.slot_factors, checks.slot_factors
        )
        feature_gram = torch.einsum(
            "cw,dw->cd", checks.feature_factors, checks.feature_factors
        )
        gram = slot_gram * feature_gram.unsqueeze(0)
        identity = torch.eye(
            self.config.checks,
            device=gram.device,
            dtype=gram.dtype,
        ).unsqueeze(0)
        coefficients = torch.linalg.solve(
            (gram + self.config.epsilon * identity).float(),
            error.float().unsqueeze(-1),
        ).squeeze(-1).to(proposed_state.dtype)
        correction = torch.einsum(
            "bc,bcs,cw->bsw",
            coefficients,
            checks.slot_factors,
            checks.feature_factors,
        )
        scale = self.correction_scale.clamp(0.0, 1.0).to(proposed_state.dtype)
        corrected = proposed_state - scale * correction
        residual = syndrome(
            corrected,
            checks.slot_factors,
            checks.feature_factors,
        ) - checks.reference_syndrome
        return corrected, error, residual, correction


class SourceStateInitializer(nn.Module):
    def __init__(self, config: SyndromeConfig):
        super().__init__()
        self.config = config
        self.source_projection = nn.Linear(
            config.input_width, config.state_width, bias=False
        )
        self.slot_queries = nn.Parameter(torch.empty(config.slots, config.state_width))
        self.attention = nn.MultiheadAttention(
            config.state_width,
            config.heads,
            batch_first=True,
        )
        self.norm = nn.RMSNorm(config.state_width)
        nn.init.normal_(self.slot_queries, std=0.02)

    def forward(
        self,
        source_features: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        source = self.source_projection(source_features)
        queries = self.slot_queries.unsqueeze(0).expand(source.shape[0], -1, -1)
        attended, _ = self.attention(
            queries,
            source,
            source,
            key_padding_mask=~source_mask,
            need_weights=False,
        )
        return self.norm(queries + attended)


class TiedStateProposer(nn.Module):
    """A standard tied recurrent proposal used before syndrome correction."""

    def __init__(self, config: SyndromeConfig):
        super().__init__()
        self.config = config
        width = config.state_width
        self.source_projection = nn.Linear(config.input_width, width, bias=False)
        self.step_embedding = (
            nn.Parameter(torch.empty(config.steps, width))
            if config.use_step_embedding
            else None
        )
        self.cross_attention = nn.MultiheadAttention(
            width, config.heads, batch_first=True
        )
        self.state_norm = nn.RMSNorm(width)
        self.update = nn.Sequential(
            nn.Linear(2 * width, config.ff_multiplier * width),
            nn.GELU(),
            nn.Linear(config.ff_multiplier * width, width),
        )
        self.gate = nn.Linear(2 * width, 1)
        self.stop = nn.Linear(width, 1)
        if self.step_embedding is not None:
            nn.init.normal_(self.step_embedding, std=0.02)
        nn.init.constant_(self.gate.bias, -1.0)
        nn.init.constant_(self.stop.bias, -2.0)

    def forward(
        self,
        state: torch.Tensor,
        source_features: torch.Tensor,
        source_mask: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = self.source_projection(source_features)
        step_bias = (
            self.step_embedding[step].view(1, 1, -1)
            if self.step_embedding is not None
            else 0.0
        )
        query = self.state_norm(state + step_bias)
        context, _ = self.cross_attention(
            query,
            source,
            source,
            key_padding_mask=~source_mask,
            need_weights=False,
        )
        joined = torch.cat((query, context), dim=-1)
        update = self.update(joined) / math.sqrt(self.config.state_width)
        gate = torch.sigmoid(self.gate(joined))
        stop = torch.sigmoid(self.stop((query + context).mean(1))).squeeze(-1)
        return update, gate, stop


class PromptConditionedSyndromeCore(nn.Module):
    """A complete source-to-corrected-state trajectory with adaptive halt."""

    def __init__(self, config: SyndromeConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.initializer = SourceStateInitializer(config)
        self.compiler = PromptConditionedCheckCompiler(config)
        self.proposer = TiedStateProposer(config)
        self.projector = MinimumNormSyndromeProjector(config)

    def forward(
        self,
        source_features: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> SyndromeTrajectory:
        if source_features.shape[-1] != self.config.input_width:
            raise SyndromeDynamicsError("source width differs")
        initial = self.initializer(source_features, source_mask)
        checks = self.compiler(source_features, source_mask, initial)
        state = initial
        alive = torch.ones(
            state.shape[0], device=state.device, dtype=state.dtype
        )
        stop_step = torch.full(
            (state.shape[0],),
            self.config.steps,
            device=state.device,
            dtype=torch.long,
        )
        trace: list[SyndromeStep] = []
        for step in range(self.config.steps):
            update, gate, stop_probability = self.proposer(
                state, source_features, source_mask, step
            )
            proposed = state + gate * update
            corrected, pre, post, correction = self.projector(proposed, checks)
            alive_before = alive
            state = state + alive.view(-1, 1, 1) * (corrected - state)
            if step + 1 >= self.config.min_steps:
                hard_stop = stop_probability.ge(0.5).to(state.dtype)
                straight_through_stop = (
                    hard_stop + stop_probability - stop_probability.detach()
                )
                newly_stopped = alive_before.bool() & hard_stop.bool()
                stop_step = torch.where(
                    newly_stopped,
                    torch.full_like(stop_step, step + 1),
                    stop_step,
                )
                alive = alive * (1.0 - straight_through_stop)
            trace.append(
                SyndromeStep(
                    state=state,
                    pre_syndrome_rms=pre.square().mean(-1).sqrt(),
                    post_syndrome_rms=post.square().mean(-1).sqrt(),
                    correction_rms=correction.square().mean((-2, -1)).sqrt(),
                    gate_mean=gate.mean((-2, -1)),
                    stop_probability=stop_probability,
                    alive_before=alive_before,
                )
            )
        return SyndromeTrajectory(
            initial_state=initial,
            final_state=state,
            checks=checks,
            steps=tuple(trace),
            stop_step=stop_step,
        )


__all__ = [
    "MinimumNormSyndromeProjector",
    "PromptConditionedCheckCompiler",
    "PromptConditionedSyndromeCore",
    "StickyChecks",
    "SyndromeConfig",
    "SyndromeDynamicsError",
    "SyndromeStep",
    "SyndromeTrajectory",
    "syndrome",
]
