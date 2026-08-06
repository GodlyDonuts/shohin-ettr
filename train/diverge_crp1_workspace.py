"""Guarded whole-candidate workspace for causal trace revision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class CRP1WorkspaceError(RuntimeError):
    """The causal-revision packet contract was violated."""


@dataclass(frozen=True)
class CausalRevisionConfig:
    backbone_width: int
    workspace_width: int = 256
    workspace_slots: int = 6
    recurrent_steps: int = 4
    attention_heads: int = 8
    ff_multiplier: int = 4
    max_trace_steps: int = 12

    def __post_init__(self) -> None:
        positive = (
            self.backbone_width,
            self.workspace_width,
            self.workspace_slots,
            self.recurrent_steps,
            self.attention_heads,
            self.ff_multiplier,
            self.max_trace_steps,
        )
        if any(value <= 0 for value in positive):
            raise CRP1WorkspaceError("revision dimensions must be positive")
        if self.workspace_width % self.attention_heads:
            raise CRP1WorkspaceError("workspace width must divide attention heads")


@dataclass
class CausalRevisionOutput:
    prefix_states: torch.Tensor
    candidate_logits: torch.Tensor
    selected_candidates: torch.Tensor
    candidate_active: torch.Tensor
    step_delta_norms: torch.Tensor
    all_candidate_prefixes: torch.Tensor


class CausalRevisionPacket(nn.Module):
    """Maintain one coherent recurrent packet per possible first-error step."""

    def __init__(self, config: CausalRevisionConfig) -> None:
        super().__init__()
        self.config = config
        width = config.workspace_width
        candidates = config.max_trace_steps + 1
        self.memory_projection = nn.Linear(config.backbone_width, width)
        self.initial_slots = nn.Parameter(torch.empty(config.workspace_slots, width))
        self.candidate_identity = nn.Parameter(torch.empty(candidates, width))
        nn.init.normal_(self.initial_slots, std=0.02)
        nn.init.normal_(self.candidate_identity, std=0.02)
        self.slot_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.problem_attention = nn.MultiheadAttention(
            width, config.attention_heads, batch_first=True
        )
        self.preserved_attention = nn.MultiheadAttention(
            width, config.attention_heads, batch_first=True
        )
        self.fault_attention = nn.MultiheadAttention(
            width, config.attention_heads, batch_first=True
        )
        self.replay_attention = nn.MultiheadAttention(
            width, config.attention_heads, batch_first=True
        )
        self.self_attention = nn.MultiheadAttention(
            width, config.attention_heads, batch_first=True
        )
        self.revision = nn.Sequential(
            nn.Linear(6 * width, width),
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
        self.candidate_head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def _validate(
        self,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
        problem_mask: torch.Tensor,
        step_mask: torch.Tensor,
        final_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if memory.ndim != 3:
            raise CRP1WorkspaceError("memory must be [batch, tokens, width]")
        expected = memory.shape[:2]
        if (
            attention_mask.shape != expected
            or problem_mask.shape != expected
            or final_mask.shape != expected
            or step_mask.ndim != 3
            or step_mask.shape[0] != expected[0]
            or step_mask.shape[2] != expected[1]
            or step_mask.shape[1] != self.config.max_trace_steps
        ):
            raise CRP1WorkspaceError("revision masks differ from memory")
        active = attention_mask.bool()
        problem = problem_mask.bool() & active
        steps = step_mask.bool() & active[:, None, :]
        final = final_mask.bool() & active
        if torch.any(problem.sum(dim=1) == 0) or torch.any(final.sum(dim=1) == 0):
            raise CRP1WorkspaceError("every row needs problem and final tokens")
        step_active = steps.any(dim=2)
        if torch.any(step_active.sum(dim=1) < 4):
            raise CRP1WorkspaceError("every row needs at least four trace steps")
        expected_prefix = (
            torch.arange(self.config.max_trace_steps, device=memory.device)[None, :]
            < step_active.sum(dim=1)[:, None]
        )
        if not torch.equal(step_active, expected_prefix):
            raise CRP1WorkspaceError("trace step masks must form a dense prefix")
        segments = torch.cat(
            (problem[:, None, :], steps, final[:, None, :]), dim=1
        ).sum(dim=1)
        if torch.any(segments > 1):
            raise CRP1WorkspaceError("revision segments overlap")
        return active, problem, steps, final

    def _guards(
        self,
        problem: torch.Tensor,
        steps: torch.Tensor,
        final: torch.Tensor,
        *,
        unguarded: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, max_steps, tokens = steps.shape
        candidates = max_steps + 1
        step_active = steps.any(dim=2)
        counts = step_active.sum(dim=1)
        candidate_active = (
            torch.arange(candidates, device=steps.device)[None, :] <= counts[:, None]
        )
        trace = steps.any(dim=1) | final
        if unguarded:
            full = trace[:, None, :].expand(batch, candidates, tokens)
            return full, full, full, candidate_active

        preserved = torch.zeros(
            batch, candidates, tokens, dtype=torch.bool, device=steps.device
        )
        fault = torch.zeros_like(preserved)
        replay = torch.zeros_like(preserved)
        preserved[:, 0] = trace
        fault[:, 0] = final
        replay[:, 0] = final
        for candidate in range(1, candidates):
            prior = steps[:, : candidate - 1].any(dim=1)
            preserved[:, candidate] = prior | (
                (~prior.any(dim=1))[:, None] & problem
            )
            fault[:, candidate] = steps[:, candidate - 1]
            later = steps[:, candidate:].any(dim=1)
            replay[:, candidate] = later | final
        fallback = final[:, None, :].expand_as(preserved)
        preserved = torch.where(candidate_active[:, :, None], preserved, fallback)
        fault = torch.where(candidate_active[:, :, None], fault, fallback)
        replay = torch.where(candidate_active[:, :, None], replay, fallback)
        if any(
            torch.any(mask.sum(dim=2) == 0) for mask in (preserved, fault, replay)
        ):
            raise CRP1WorkspaceError("a revision guard is empty")
        return preserved, fault, replay, candidate_active

    def forward(
        self,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
        problem_mask: torch.Tensor,
        step_mask: torch.Tensor,
        final_mask: torch.Tensor,
        *,
        unguarded: bool = False,
        selection_targets: torch.Tensor | None = None,
        ablation: str = "normal",
    ) -> CausalRevisionOutput:
        if ablation not in {"normal", "reset", "force_no_error", "shift", "packet_swap"}:
            raise CRP1WorkspaceError("revision ablation differs")
        _, problem, steps, final = self._validate(
            memory, attention_mask, problem_mask, step_mask, final_mask
        )
        preserved, fault, replay, candidate_active = self._guards(
            problem, steps, final, unguarded=unguarded
        )
        batch, candidates, tokens = preserved.shape
        width = self.config.workspace_width
        projected = self.memory_norm(self.memory_projection(memory.float()))
        expanded_memory = projected[:, None].expand(
            batch, candidates, tokens, width
        ).reshape(batch * candidates, tokens, width)
        problem_keys = problem[:, None, :].expand(
            batch, candidates, tokens
        ).reshape(batch * candidates, tokens)
        preserved_keys = preserved.reshape(batch * candidates, tokens)
        fault_keys = fault.reshape(batch * candidates, tokens)
        replay_keys = replay.reshape(batch * candidates, tokens)
        identity = self.candidate_identity[:candidates]
        slots = self.initial_slots[None, None] + identity[None, :, None, :]
        slots = slots.expand(batch, candidates, -1, -1).reshape(
            batch * candidates, self.config.workspace_slots, width
        )
        deltas: list[torch.Tensor] = []
        for _ in range(self.config.recurrent_steps):
            query = self.slot_norm(slots)
            problem_context, _ = self.problem_attention(
                query,
                expanded_memory,
                expanded_memory,
                key_padding_mask=~problem_keys,
                need_weights=False,
            )
            preserved_context, _ = self.preserved_attention(
                query,
                expanded_memory,
                expanded_memory,
                key_padding_mask=~preserved_keys,
                need_weights=False,
            )
            fault_context, _ = self.fault_attention(
                query,
                expanded_memory,
                expanded_memory,
                key_padding_mask=~fault_keys,
                need_weights=False,
            )
            replay_context, _ = self.replay_attention(
                query,
                expanded_memory,
                expanded_memory,
                key_padding_mask=~replay_keys,
                need_weights=False,
            )
            self_context, _ = self.self_attention(
                query, query, query, need_weights=False
            )
            revision = self.revision(
                torch.cat(
                    (
                        problem_context,
                        preserved_context,
                        fault_context,
                        replay_context,
                        fault_context - preserved_context,
                        replay_context - fault_context,
                    ),
                    dim=-1,
                )
            )
            candidate = slots + self_context + revision
            candidate = candidate + self.feed_forward(self.candidate_norm(candidate))
            gate = torch.sigmoid(
                self.update_gate(torch.cat((slots, candidate), dim=-1))
            )
            updated = gate * candidate + (1.0 - gate) * slots
            deltas.append(
                (updated - slots).float().square().mean(dim=(1, 2)).sqrt()
            )
            slots = updated

        normalized = self.output_norm(slots)
        pooled = normalized.mean(dim=1)
        logits = self.candidate_head(pooled).reshape(batch, candidates)
        logits = logits.masked_fill(~candidate_active, -1.0e4)
        if selection_targets is not None:
            if selection_targets.shape != (batch,):
                raise CRP1WorkspaceError("selection targets differ")
            if torch.any(selection_targets < 0) or torch.any(
                selection_targets >= candidates
            ):
                raise CRP1WorkspaceError("selection target is outside packet")
            if torch.any(
                ~candidate_active.gather(1, selection_targets[:, None]).squeeze(1)
            ):
                raise CRP1WorkspaceError("selection target is inactive")
            selected = selection_targets
        else:
            selected = logits.argmax(dim=1)
        counts = candidate_active.sum(dim=1) - 1
        if ablation == "force_no_error":
            selected = torch.zeros_like(selected)
        elif ablation == "shift":
            selected = selected.remainder(counts) + 1
        prefixes = self.output_projection(normalized).reshape(
            batch,
            candidates,
            self.config.workspace_slots,
            self.config.backbone_width,
        )
        gather = selected[:, None, None, None].expand(
            batch, 1, self.config.workspace_slots, self.config.backbone_width
        )
        prefix = prefixes.gather(1, gather).squeeze(1)
        if ablation == "reset":
            prefix = torch.zeros_like(prefix)
        elif ablation == "packet_swap":
            prefix = prefix.roll(1, dims=0) if batch > 1 else torch.zeros_like(prefix)
        return CausalRevisionOutput(
            prefix_states=prefix,
            candidate_logits=logits,
            selected_candidates=selected,
            candidate_active=candidate_active,
            step_delta_norms=torch.stack(deltas, dim=1).reshape(
                batch, candidates, self.config.recurrent_steps
            ),
            all_candidate_prefixes=prefixes,
        )


__all__ = [
    "CRP1WorkspaceError",
    "CausalRevisionConfig",
    "CausalRevisionOutput",
    "CausalRevisionPacket",
]
