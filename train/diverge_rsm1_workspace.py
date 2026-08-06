"""Hard persistent byte-state replay conditioned by one selected CRP1 packet."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from diverge_rsm1_data import MAX_STATE_SLOTS, STATE_VOCAB_SIZE


class RSM1WorkspaceError(RuntimeError):
    """The persistent state replay contract was violated."""


@dataclass(frozen=True)
class PersistentReplayConfig:
    backbone_width: int
    state_width: int = 256
    state_slots: int = MAX_STATE_SLOTS
    packet_slots: int = 6
    max_trace_steps: int = 12
    attention_heads: int = 8
    ff_multiplier: int = 4
    state_vocab_size: int = STATE_VOCAB_SIZE

    def __post_init__(self) -> None:
        values = (
            self.backbone_width,
            self.state_width,
            self.state_slots,
            self.packet_slots,
            self.max_trace_steps,
            self.attention_heads,
            self.ff_multiplier,
            self.state_vocab_size,
        )
        if any(value <= 0 for value in values):
            raise RSM1WorkspaceError("replay dimensions must be positive")
        if self.state_width % self.attention_heads:
            raise RSM1WorkspaceError("state width must divide attention heads")


@dataclass
class PersistentReplayOutput:
    initial_logits: torch.Tensor
    transition_logits: torch.Tensor
    state_trace_tokens: torch.Tensor
    terminal_tokens: torch.Tensor
    replay_active: torch.Tensor
    step_delta_norms: torch.Tensor


class PersistentStateReplay(nn.Module):
    """Decode, execute, and re-encode one hard complete state at every step."""

    def __init__(self, config: PersistentReplayConfig) -> None:
        super().__init__()
        self.config = config
        width = config.state_width
        self.packet_projection = nn.Linear(config.backbone_width, width, bias=False)
        self.memory_projection = nn.Linear(config.backbone_width, width, bias=False)
        self.packet_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.initial_queries = nn.Parameter(torch.empty(config.state_slots, width))
        self.state_positions = nn.Parameter(torch.empty(config.state_slots, width))
        self.step_identity = nn.Parameter(torch.empty(config.max_trace_steps, width))
        self.byte_embedding = nn.Embedding(config.state_vocab_size, width)
        nn.init.normal_(self.initial_queries, std=0.02)
        nn.init.normal_(self.state_positions, std=0.02)
        nn.init.normal_(self.step_identity, std=0.02)

        self.initial_attention = nn.MultiheadAttention(
            width, config.attention_heads, batch_first=True
        )
        self.initial_norm = nn.LayerNorm(width)
        self.initial_ff = nn.Sequential(
            nn.Linear(width, config.ff_multiplier * width),
            nn.SiLU(),
            nn.Linear(config.ff_multiplier * width, width),
        )
        self.state_norm = nn.LayerNorm(width)
        self.state_attention = nn.MultiheadAttention(
            width, config.attention_heads, batch_first=True
        )
        self.operation_attention = nn.MultiheadAttention(
            width, config.attention_heads, batch_first=True
        )
        self.transition_norm = nn.LayerNorm(width)
        self.transition_ff = nn.Sequential(
            nn.Linear(3 * width, config.ff_multiplier * width),
            nn.SiLU(),
            nn.Linear(config.ff_multiplier * width, width),
        )
        self.transition_gate = nn.Linear(2 * width, width)
        self.output_norm = nn.LayerNorm(width)
        self.state_head = nn.Linear(width, config.state_vocab_size)

    def _validate(
        self,
        packet_prefix: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
        problem_mask: torch.Tensor,
        step_mask: torch.Tensor,
        selected_candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = memory.shape[0] if memory.ndim == 3 else -1
        if (
            memory.ndim != 3
            or memory.shape[2] != self.config.backbone_width
            or packet_prefix.shape
            != (batch, self.config.packet_slots, self.config.backbone_width)
            or attention_mask.shape != memory.shape[:2]
            or problem_mask.shape != memory.shape[:2]
            or step_mask.shape
            != (batch, self.config.max_trace_steps, memory.shape[1])
            or selected_candidates.shape != (batch,)
            or selected_candidates.dtype != torch.long
        ):
            raise RSM1WorkspaceError("persistent replay tensor interface differs")
        attention = attention_mask.bool()
        problem = problem_mask.bool() & attention
        steps = step_mask.bool() & attention[:, None, :]
        if torch.any(problem.sum(dim=1) == 0):
            raise RSM1WorkspaceError("every replay row needs problem tokens")
        step_active = steps.any(dim=2)
        depth = step_active.sum(dim=1)
        dense = (
            torch.arange(self.config.max_trace_steps, device=memory.device)[None, :]
            < depth[:, None]
        )
        if not torch.equal(step_active, dense):
            raise RSM1WorkspaceError("replay step spans must form a dense prefix")
        if torch.any(selected_candidates < 0) or torch.any(selected_candidates > depth):
            raise RSM1WorkspaceError("selected replay boundary is outside the trace")
        replay_active = (
            selected_candidates[:, None].gt(0)
            & (
                torch.arange(1, self.config.max_trace_steps + 1, device=memory.device)[
                    None, :
                ]
                >= selected_candidates[:, None]
            )
            & dense
        )
        return attention, problem, steps, replay_active

    def _hard_feedback(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = logits.float().softmax(-1)
        tokens = probabilities.argmax(-1)
        hard = F.one_hot(tokens, self.config.state_vocab_size).to(probabilities.dtype)
        straight_through = hard + probabilities - probabilities.detach()
        embeddings = straight_through @ self.byte_embedding.weight.float()
        embeddings = embeddings + self.state_positions[None]
        return embeddings, tokens

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if (
            tokens.ndim != 2
            or tokens.shape[1] != self.config.state_slots
            or tokens.dtype != torch.long
            or (tokens.numel() and int(tokens.min()) < 0)
            or (tokens.numel() and int(tokens.max()) >= self.config.state_vocab_size)
        ):
            raise RSM1WorkspaceError("hard state tokens differ")
        return self.byte_embedding(tokens).float() + self.state_positions[None]

    def _transition(
        self,
        state: torch.Tensor,
        memory_state: torch.Tensor,
        operation_mask: torch.Tensor,
        problem_mask: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        safe_mask = operation_mask | (
            (~operation_mask.any(dim=1))[:, None] & problem_mask
        )
        normalized = self.state_norm(state)
        self_context, _ = self.state_attention(
            normalized, normalized, normalized, need_weights=False
        )
        operation_context, _ = self.operation_attention(
            normalized,
            memory_state,
            memory_state,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        identity = self.step_identity[step_index][None, None].expand_as(state)
        proposal = state + self_context + self.transition_ff(
            torch.cat((normalized, operation_context, identity), dim=-1)
        )
        gate = torch.sigmoid(
            self.transition_gate(torch.cat((state, proposal), dim=-1))
        )
        proposal = gate * proposal + (1.0 - gate) * state
        return self.state_head(self.output_norm(proposal)).float()

    def oracle_transition_logits(
        self,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
        problem_mask: torch.Tensor,
        step_mask: torch.Tensor,
        predecessor_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Supervise the same transition from exact predecessor states."""

        if (
            memory.ndim != 3
            or memory.shape[2] != self.config.backbone_width
            or attention_mask.shape != memory.shape[:2]
            or problem_mask.shape != memory.shape[:2]
            or step_mask.shape
            != (memory.shape[0], self.config.max_trace_steps, memory.shape[1])
            or predecessor_tokens.shape
            != (
                memory.shape[0],
                self.config.max_trace_steps,
                self.config.state_slots,
            )
            or predecessor_tokens.dtype != torch.long
        ):
            raise RSM1WorkspaceError("oracle transition tensors differ")
        active = attention_mask.bool()
        problem = problem_mask.bool() & active
        steps = step_mask.bool() & active[:, None, :]
        if torch.any(problem.sum(dim=1) == 0):
            raise RSM1WorkspaceError("oracle transition lacks problem tokens")
        memory_state = self.memory_norm(self.memory_projection(memory.float()))
        outputs: list[torch.Tensor] = []
        for index in range(self.config.max_trace_steps):
            state = self._embed_tokens(predecessor_tokens[:, index])
            outputs.append(
                self._transition(state, memory_state, steps[:, index], problem, index)
            )
        return torch.stack(outputs, dim=1)

    def forward(
        self,
        packet_prefix: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
        problem_mask: torch.Tensor,
        step_mask: torch.Tensor,
        selected_candidates: torch.Tensor,
    ) -> PersistentReplayOutput:
        _, problem, steps, replay_active = self._validate(
            packet_prefix,
            memory,
            attention_mask,
            problem_mask,
            step_mask,
            selected_candidates,
        )
        packet = self.packet_norm(self.packet_projection(packet_prefix.float()))
        memory_state = self.memory_norm(self.memory_projection(memory.float()))
        queries = self.initial_queries[None].expand(memory.shape[0], -1, -1)
        initial_context, _ = self.initial_attention(
            queries, packet, packet, need_weights=False
        )
        initial_hidden = queries + initial_context
        initial_hidden = initial_hidden + self.initial_ff(
            self.initial_norm(initial_hidden)
        )
        initial_logits = self.state_head(self.output_norm(initial_hidden)).float()
        state, tokens = self._hard_feedback(initial_logits)
        traces = [tokens]
        logits_by_step: list[torch.Tensor] = []
        deltas: list[torch.Tensor] = []

        for index in range(self.config.max_trace_steps):
            active = replay_active[:, index]
            previous_state = state
            operation_mask = steps[:, index]
            step_logits = self._transition(
                state, memory_state, operation_mask, problem, index
            )
            next_state, next_tokens = self._hard_feedback(step_logits)
            selector = active[:, None, None]
            state = torch.where(selector, next_state, state)
            tokens = torch.where(active[:, None], next_tokens, tokens)
            logits_by_step.append(step_logits)
            traces.append(tokens)
            deltas.append(
                (next_state - previous_state)
                .float()
                .square()
                .mean(dim=(1, 2))
                .sqrt()
            )

        return PersistentReplayOutput(
            initial_logits=initial_logits,
            transition_logits=torch.stack(logits_by_step, dim=1),
            state_trace_tokens=torch.stack(traces, dim=1),
            terminal_tokens=tokens,
            replay_active=replay_active,
            step_delta_norms=torch.stack(deltas, dim=1),
        )


__all__ = [
    "PersistentReplayConfig",
    "PersistentReplayOutput",
    "PersistentStateReplay",
    "RSM1WorkspaceError",
]
