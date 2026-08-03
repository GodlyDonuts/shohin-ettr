"""Prompt-conditioned recurrent workspace for practical reasoning models.

The module converts frozen or jointly trained language-model prompt features
into a short sequence of soft-prefix states.  A single tied cell owns every
internal step, so longer deliberation changes compute without changing the
number of reasoning parameters.  It has no symbolic executor, host arithmetic,
or answer channel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntegratedWorkspaceError(ValueError):
    """The integrated workspace tensor or configuration contract differs."""


@dataclass(frozen=True, slots=True)
class IntegratedWorkspaceConfig:
    backbone_width: int
    workspace_width: int = 512
    workspace_slots: int = 16
    recurrent_steps: int = 8
    attention_heads: int = 8
    ff_multiplier: int = 4
    dropout: float = 0.0

    def validate(self) -> None:
        dimensions = (
            self.backbone_width,
            self.workspace_width,
            self.workspace_slots,
            self.recurrent_steps,
            self.attention_heads,
            self.ff_multiplier,
        )
        if any(value <= 0 for value in dimensions):
            raise IntegratedWorkspaceError("workspace dimensions must be positive")
        if self.workspace_width % self.attention_heads:
            raise IntegratedWorkspaceError(
                "workspace width must divide evenly across attention heads"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise IntegratedWorkspaceError("dropout is outside [0, 1)")


@dataclass(frozen=True, slots=True)
class IntegratedWorkspaceOutput:
    prefix_states: torch.Tensor
    workspace_states: torch.Tensor
    stop_logits: torch.Tensor
    step_deltas: torch.Tensor


class TiedWorkspaceCell(nn.Module):
    """One shared prompt-conditioned update used for every reasoning step."""

    def __init__(self, config: IntegratedWorkspaceConfig) -> None:
        super().__init__()
        config.validate()
        width = config.workspace_width
        self.state_norm = nn.LayerNorm(width)
        self.prompt_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(
            width,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(
            width,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.candidate_norm = nn.LayerNorm(width)
        self.candidate = nn.Sequential(
            nn.Linear(width, width * config.ff_multiplier),
            nn.SiLU(),
            nn.Linear(width * config.ff_multiplier, width),
        )
        self.update_gate = nn.Linear(width * 2, width)
        self.stop_head = nn.Linear(width, 1)

    def forward(
        self,
        state: torch.Tensor,
        prompt: torch.Tensor,
        prompt_padding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_state = self.state_norm(state)
        prompt_delta, _ = self.cross_attention(
            normalized_state,
            self.prompt_norm(prompt),
            self.prompt_norm(prompt),
            key_padding_mask=prompt_padding,
            need_weights=False,
        )
        attended = state + prompt_delta
        self_delta, _ = self.self_attention(
            self.self_norm(attended),
            self.self_norm(attended),
            self.self_norm(attended),
            need_weights=False,
        )
        attended = attended + self_delta
        candidate = self.candidate(self.candidate_norm(attended))
        gate = torch.sigmoid(self.update_gate(torch.cat((state, candidate), dim=-1)))
        delta = gate * candidate
        updated = attended + delta
        stop_logit = self.stop_head(updated.mean(dim=1)).squeeze(-1)
        return updated, stop_logit, delta


class IntegratedReasoningWorkspace(nn.Module):
    """Create soft-prefix states by recurrently deliberating over a prompt."""

    def __init__(self, config: IntegratedWorkspaceConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.prompt_projection = nn.Linear(
            config.backbone_width, config.workspace_width, bias=False
        )
        self.initial_slots = nn.Parameter(
            torch.empty(config.workspace_slots, config.workspace_width)
        )
        self.cell = TiedWorkspaceCell(config)
        self.output_norm = nn.LayerNorm(config.workspace_width)
        self.output_projection = nn.Linear(
            config.workspace_width, config.backbone_width, bias=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.initial_slots, mean=0.0, std=0.02)

    def forward(
        self,
        prompt_features: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> IntegratedWorkspaceOutput:
        if prompt_features.ndim != 3:
            raise IntegratedWorkspaceError("prompt features must have rank three")
        batch, tokens, width = prompt_features.shape
        if width != self.config.backbone_width:
            raise IntegratedWorkspaceError("prompt feature width differs")
        if prompt_attention_mask.shape != (batch, tokens):
            raise IntegratedWorkspaceError("prompt attention mask geometry differs")
        if not torch.isfinite(prompt_features).all():
            raise IntegratedWorkspaceError("prompt features contain nonfinite values")
        active = prompt_attention_mask.to(dtype=torch.bool)
        if not active.any(dim=1).all():
            raise IntegratedWorkspaceError("every prompt must contain an active token")

        prompt = self.prompt_projection(prompt_features)
        state = self.initial_slots.unsqueeze(0).expand(batch, -1, -1)
        stop_logits: list[torch.Tensor] = []
        step_deltas: list[torch.Tensor] = []
        for _ in range(self.config.recurrent_steps):
            state, stop_logit, delta = self.cell(state, prompt, ~active)
            stop_logits.append(stop_logit)
            step_deltas.append(delta.square().mean(dim=(1, 2)).sqrt())
        prefix = self.output_projection(self.output_norm(state))
        return IntegratedWorkspaceOutput(
            prefix_states=prefix,
            workspace_states=state,
            stop_logits=torch.stack(stop_logits, dim=1),
            step_deltas=torch.stack(step_deltas, dim=1),
        )

    def halting_regularizer(self, output: IntegratedWorkspaceOutput) -> torch.Tensor:
        """Encourage confident, monotone late halting without forcing an early stop."""

        probabilities = output.stop_logits.sigmoid()
        monotone_penalty = F.relu(probabilities[:, :-1] - probabilities[:, 1:]).mean()
        final_penalty = (1.0 - probabilities[:, -1]).mean()
        return monotone_penalty + final_penalty

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class DenseReasoningWorkspace(nn.Module):
    """Capacity-matched untied control for the recurrent workspace."""

    def __init__(self, config: IntegratedWorkspaceConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.prompt_projection = nn.Linear(
            config.backbone_width, config.workspace_width, bias=False
        )
        self.initial_slots = nn.Parameter(
            torch.empty(config.workspace_slots, config.workspace_width)
        )
        self.cells = nn.ModuleList(
            TiedWorkspaceCell(config) for _ in range(config.recurrent_steps)
        )
        self.output_norm = nn.LayerNorm(config.workspace_width)
        self.output_projection = nn.Linear(
            config.workspace_width, config.backbone_width, bias=False
        )
        nn.init.normal_(self.initial_slots, mean=0.0, std=0.02)

    def forward(
        self,
        prompt_features: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> IntegratedWorkspaceOutput:
        if prompt_features.ndim != 3:
            raise IntegratedWorkspaceError("prompt features must have rank three")
        batch, tokens, width = prompt_features.shape
        if width != self.config.backbone_width:
            raise IntegratedWorkspaceError("prompt feature width differs")
        if prompt_attention_mask.shape != (batch, tokens):
            raise IntegratedWorkspaceError("prompt attention mask geometry differs")
        if not torch.isfinite(prompt_features).all():
            raise IntegratedWorkspaceError("prompt features contain nonfinite values")
        active = prompt_attention_mask.to(dtype=torch.bool)
        if not active.any(dim=1).all():
            raise IntegratedWorkspaceError("every prompt must contain an active token")

        prompt = self.prompt_projection(prompt_features)
        state = self.initial_slots.unsqueeze(0).expand(batch, -1, -1)
        stop_logits: list[torch.Tensor] = []
        step_deltas: list[torch.Tensor] = []
        for cell in self.cells:
            state, stop_logit, delta = cell(state, prompt, ~active)
            stop_logits.append(stop_logit)
            step_deltas.append(delta.square().mean(dim=(1, 2)).sqrt())
        prefix = self.output_projection(self.output_norm(state))
        return IntegratedWorkspaceOutput(
            prefix_states=prefix,
            workspace_states=state,
            stop_logits=torch.stack(stop_logits, dim=1),
            step_deltas=torch.stack(step_deltas, dim=1),
        )

    def halting_regularizer(self, output: IntegratedWorkspaceOutput) -> torch.Tensor:
        probabilities = output.stop_logits.sigmoid()
        monotone_penalty = F.relu(probabilities[:, :-1] - probabilities[:, 1:]).mean()
        final_penalty = (1.0 - probabilities[:, -1]).mean()
        return monotone_penalty + final_penalty

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def workspace_architecture_sha256(config: IntegratedWorkspaceConfig) -> str:
    config.validate()
    payload = {
        "schema": "shohin-integrated-reasoning-workspace-v1",
        "config": asdict(config),
        "mechanism": (
            "prompt-projection+learned-slots+tied-cross-self-gated-cell+soft-prefix"
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def dense_workspace_architecture_sha256(config: IntegratedWorkspaceConfig) -> str:
    config.validate()
    payload = {
        "schema": "shohin-dense-reasoning-workspace-control-v1",
        "config": asdict(config),
        "mechanism": (
            "prompt-projection+learned-slots+untied-cross-self-gated-cells+soft-prefix"
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
