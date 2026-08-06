"""Question/draft fault-line workspace for verified temporal correction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class VCR1WorkspaceError(RuntimeError):
    """The temporal-correction workspace contract was violated."""


@dataclass(frozen=True)
class TemporalCorrectionConfig:
    backbone_width: int
    workspace_width: int = 384
    workspace_slots: int = 8
    recurrent_steps: int = 4
    attention_heads: int = 8
    ff_multiplier: int = 4

    def __post_init__(self) -> None:
        positive = (
            self.backbone_width,
            self.workspace_width,
            self.workspace_slots,
            self.recurrent_steps,
            self.attention_heads,
            self.ff_multiplier,
        )
        if any(value <= 0 for value in positive):
            raise VCR1WorkspaceError("correction dimensions must be positive")
        if self.workspace_width % self.attention_heads:
            raise VCR1WorkspaceError("workspace width must divide attention heads")


@dataclass
class TemporalCorrectionOutput:
    prefix_states: torch.Tensor
    validity_logits: torch.Tensor
    correction_strength: torch.Tensor
    step_delta_norms: torch.Tensor


class TemporalCorrectionReactor(nn.Module):
    """Tied correction dynamics over separate question and draft channels."""

    def __init__(self, config: TemporalCorrectionConfig) -> None:
        super().__init__()
        self.config = config
        width = config.workspace_width
        self.memory_projection = nn.Linear(config.backbone_width, width)
        self.initial_slots = nn.Parameter(torch.empty(config.workspace_slots, width))
        nn.init.normal_(self.initial_slots, std=0.02)
        self.slot_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.question_attention = nn.MultiheadAttention(
            width, config.attention_heads, batch_first=True
        )
        self.draft_attention = nn.MultiheadAttention(
            width, config.attention_heads, batch_first=True
        )
        self.self_attention = nn.MultiheadAttention(
            width, config.attention_heads, batch_first=True
        )
        self.discrepancy = nn.Sequential(
            nn.Linear(3 * width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.candidate_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, config.ff_multiplier * width),
            nn.SiLU(),
            nn.Linear(config.ff_multiplier * width, width),
        )
        self.update_gate = nn.Linear(2 * width, width)
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, config.backbone_width)
        self.validity_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    @staticmethod
    def _validate_masks(
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
        question_mask: torch.Tensor,
        draft_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if memory.ndim != 3:
            raise VCR1WorkspaceError("memory must be [batch, tokens, width]")
        expected = memory.shape[:2]
        if (
            attention_mask.shape != expected
            or question_mask.shape != expected
            or draft_mask.shape != expected
        ):
            raise VCR1WorkspaceError("correction masks differ from memory")
        active = attention_mask.bool()
        question = question_mask.bool() & active
        draft = draft_mask.bool() & active
        if torch.any(question & draft):
            raise VCR1WorkspaceError("question and draft masks overlap")
        if torch.any(question.sum(dim=1) == 0) or torch.any(draft.sum(dim=1) == 0):
            raise VCR1WorkspaceError("every row needs question and draft tokens")
        return active, question, draft

    def forward(
        self,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
        question_mask: torch.Tensor,
        draft_mask: torch.Tensor,
        *,
        role_blind: bool = False,
        swap_roles: bool = False,
        reset_prefix: bool = False,
    ) -> TemporalCorrectionOutput:
        active, question, draft = self._validate_masks(
            memory, attention_mask, question_mask, draft_mask
        )
        if role_blind:
            question = active
            draft = active
        elif swap_roles:
            question, draft = draft, question

        projected = self.memory_norm(self.memory_projection(memory.float()))
        slots = self.initial_slots.unsqueeze(0).expand(memory.shape[0], -1, -1)
        deltas: list[torch.Tensor] = []
        for _ in range(self.config.recurrent_steps):
            query = self.slot_norm(slots)
            question_context, _ = self.question_attention(
                query,
                projected,
                projected,
                key_padding_mask=~question,
                need_weights=False,
            )
            draft_context, _ = self.draft_attention(
                query,
                projected,
                projected,
                key_padding_mask=~draft,
                need_weights=False,
            )
            self_context, _ = self.self_attention(
                query, query, query, need_weights=False
            )
            discrepancy = self.discrepancy(
                torch.cat(
                    (
                        question_context,
                        draft_context,
                        question_context - draft_context,
                    ),
                    dim=-1,
                )
            )
            candidate = slots + self_context + discrepancy
            candidate = candidate + self.feed_forward(self.candidate_norm(candidate))
            gate = torch.sigmoid(
                self.update_gate(torch.cat((slots, candidate), dim=-1))
            )
            updated = gate * candidate + (1.0 - gate) * slots
            deltas.append((updated - slots).float().square().mean(dim=(1, 2)).sqrt())
            slots = updated

        pooled = self.output_norm(slots).mean(dim=1)
        validity_logits = self.validity_head(pooled).squeeze(-1)
        correction_strength = torch.sigmoid(-validity_logits)
        prefix = self.output_projection(self.output_norm(slots))
        prefix = prefix * (0.05 + 0.95 * correction_strength[:, None, None])
        if reset_prefix:
            prefix = torch.zeros_like(prefix)
        return TemporalCorrectionOutput(
            prefix_states=prefix,
            validity_logits=validity_logits,
            correction_strength=correction_strength,
            step_delta_norms=torch.stack(deltas, dim=1),
        )


__all__ = [
    "TemporalCorrectionConfig",
    "TemporalCorrectionOutput",
    "TemporalCorrectionReactor",
    "VCR1WorkspaceError",
]
