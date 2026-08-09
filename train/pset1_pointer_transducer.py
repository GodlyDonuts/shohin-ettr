"""PSET1 two-stream pointer/edit head and generic byte-preserving executor."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


KEEP = 0
REPLACE = 1


class PSET1Error(RuntimeError):
    """A PSET1 edit program or tensor geometry is invalid."""


@dataclass(frozen=True)
class PSET1Config:
    host_hidden_size: int = 2048
    width: int = 256
    attention_heads: int = 8
    ff_width: int = 1024
    replacement_layers: int = 2
    max_replacement_tokens: int = 16


@dataclass(frozen=True)
class EditProgram:
    action: int
    start: int | None = None
    end: int | None = None
    replacement: str = ""


def execute_program(draft: str, offsets: list[list[int]], program: EditProgram) -> str:
    if program.action == KEEP:
        if program.start is not None or program.end is not None or program.replacement:
            raise PSET1Error("KEEP carries edit fields")
        return draft
    if program.action != REPLACE or program.start is None or program.end is None:
        raise PSET1Error("PSET1 action or pointers are absent")
    if not 0 <= program.start <= program.end < len(draft):
        raise PSET1Error("PSET1 pointer is outside the draft")
    left, right = program.start, program.end + 1
    if not program.replacement:
        raise PSET1Error("PSET1 character span or replacement is invalid")
    return draft[:left] + program.replacement + draft[right:]


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class CrossFusion(nn.Module):
    def __init__(self, config: PSET1Config) -> None:
        super().__init__()
        self.cross = nn.MultiheadAttention(
            config.width, config.attention_heads, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(config.width)
        self.ff = nn.Sequential(
            nn.Linear(config.width, config.ff_width),
            nn.GELU(),
            nn.Linear(config.ff_width, config.width),
        )
        self.ff_norm = nn.LayerNorm(config.width)

    def forward(
        self,
        draft: torch.Tensor,
        source: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.cross(
            draft,
            source,
            source,
            key_padding_mask=~source_mask.bool(),
            need_weights=False,
        )
        hidden = self.cross_norm(draft + attended)
        return self.ff_norm(hidden + self.ff(hidden))


class PSET1PointerHead(nn.Module):
    def __init__(self, config: PSET1Config) -> None:
        super().__init__()
        self.config = config
        self.source_projection = nn.Linear(config.host_hidden_size, config.width)
        self.draft_projection = nn.Linear(config.host_hidden_size, config.width)
        self.fusion = CrossFusion(config)
        self.character_embedding = nn.Embedding(256, config.width)
        self.action_head = nn.Linear(2 * config.width, 2)
        self.start_head = nn.Linear(config.width, 1)
        self.end_head = nn.Linear(config.width, 1)
        self.replacement_input = nn.Embedding(257, config.width)
        self.replacement_position = nn.Embedding(config.max_replacement_tokens + 1, config.width)
        layer = nn.TransformerDecoderLayer(
            d_model=config.width,
            nhead=config.attention_heads,
            dim_feedforward=config.ff_width,
            batch_first=True,
            norm_first=True,
        )
        self.replacement_decoder = nn.TransformerDecoder(
            layer, num_layers=config.replacement_layers
        )
        self.replacement_output = nn.Linear(config.width, 257)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def trainable_parameter_name_sha256(self) -> str:
        names = sorted(name for name, parameter in self.named_parameters() if parameter.requires_grad)
        return hashlib.sha256("\n".join(names).encode()).hexdigest()

    def encode(
        self,
        source_hidden: torch.Tensor,
        source_mask: torch.Tensor,
        draft_hidden: torch.Tensor,
        draft_mask: torch.Tensor,
        character_to_token: torch.Tensor,
        character_ids: torch.Tensor,
        character_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source = self.source_projection(source_hidden)
        draft = self.draft_projection(draft_hidden)
        fused = self.fusion(draft, source, source_mask)
        action = self.action_head(
            torch.cat((masked_mean(source, source_mask), masked_mean(fused, draft_mask)), dim=-1)
        )
        gather = character_to_token.unsqueeze(-1).expand(-1, -1, fused.shape[-1])
        characters = fused.gather(1, gather) + self.character_embedding(character_ids)
        start = self.start_head(characters).squeeze(-1).masked_fill(~character_mask.bool(), -torch.inf)
        end = self.end_head(characters).squeeze(-1).masked_fill(~character_mask.bool(), -torch.inf)
        return source, characters, action, torch.stack((start, end), dim=1)

    def replacement_logits(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor,
        character_states: torch.Tensor,
        pointers: torch.Tensor,
        replacement_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, length = replacement_ids.shape
        if length > self.config.max_replacement_tokens + 1:
            raise PSET1Error("PSET1 replacement input exceeds budget")
        row = torch.arange(batch, device=character_states.device)
        start = character_states[row, pointers[:, 0]]
        end = character_states[row, pointers[:, 1]]
        edit_state = ((start + end) * 0.5).unsqueeze(1)
        memory = torch.cat((source, edit_state), dim=1)
        memory_mask = torch.cat(
            (source_mask.bool(), torch.ones((batch, 1), device=source_mask.device, dtype=torch.bool)),
            dim=1,
        )
        positions = torch.arange(length, device=source.device)
        target = self.replacement_input(replacement_ids) + self.replacement_position(positions)[None]
        causal = torch.triu(
            torch.ones((length, length), device=source.device, dtype=torch.bool), diagonal=1
        )
        decoded = self.replacement_decoder(
            target,
            memory,
            tgt_mask=causal,
            memory_key_padding_mask=~memory_mask,
        )
        return self.replacement_output(decoded)


def pointer_loss(
    action_logits: torch.Tensor,
    pointer_logits: torch.Tensor,
    replacement_logits: torch.Tensor,
    actions: torch.Tensor,
    pointers: torch.Tensor,
    replacement_labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    action = F.cross_entropy(action_logits, actions)
    active = actions == REPLACE
    if not bool(active.any()):
        raise PSET1Error("PSET1 training batch has no edit")
    start = F.cross_entropy(pointer_logits[active, 0], pointers[active, 0])
    end = F.cross_entropy(pointer_logits[active, 1], pointers[active, 1])
    replacement = F.cross_entropy(
        replacement_logits[active].reshape(-1, replacement_logits.shape[-1]),
        replacement_labels[active].reshape(-1),
        ignore_index=-100,
    )
    total = (action + start + end + replacement) * 0.25
    return total, {
        "action_ce": action.detach(),
        "start_ce": start.detach(),
        "end_ce": end.detach(),
        "replacement_ce": replacement.detach(),
    }
