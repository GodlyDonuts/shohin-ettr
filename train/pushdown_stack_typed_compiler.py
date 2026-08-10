"""Hard pushdown-stack compiler mechanics for PSTC1."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from fixed_slot_typed_compiler import MAX_SOURCE_NUMBERS, NumberSpan


MAX_ACTIONS = 22
MAX_STACK = 6
ACTIONS = (
    "PUSH",
    "NEGATE",
    "APPLY_ADD",
    "APPLY_SUB",
    "APPLY_MUL",
    "APPLY_DIV",
    "STOP",
)
ACTION_TO_ID = {action: index for index, action in enumerate(ACTIONS)}
PUSH = ACTION_TO_ID["PUSH"]
NEGATE = ACTION_TO_ID["NEGATE"]
STOP = ACTION_TO_ID["STOP"]
APPLY_BEGIN = ACTION_TO_ID["APPLY_ADD"]
APPLY_END = ACTION_TO_ID["APPLY_DIV"] + 1


class PushdownCompilerError(ValueError):
    """Raised when PSTC1 action or stack geometry differs."""


@dataclass(frozen=True, slots=True)
class StackAction:
    action: int
    source_index: int = -1


@dataclass(frozen=True, slots=True)
class StackProgram:
    identity_sha256: str
    family: str
    question: str
    number_spans: tuple[NumberSpan, ...]
    actions: tuple[StackAction, ...]
    maximum_stack: int


@dataclass(slots=True)
class StackCompilerOutput:
    action_logits: torch.Tensor
    pointer_logits: torch.Tensor
    chosen_actions: torch.Tensor
    chosen_pointers: torch.Tensor
    stack_depths: torch.Tensor
    invalid_action_count: torch.Tensor


def load_stack_program(row: dict[str, Any]) -> StackProgram:
    identity = row.get("identity_sha256")
    family = row.get("family")
    question = row.get("question")
    raw_spans = row.get("number_spans")
    raw_actions = row.get("actions")
    maximum_stack = row.get("maximum_stack")
    if (
        not isinstance(identity, str)
        or len(identity) != 64
        or not isinstance(family, str)
        or not isinstance(question, str)
        or not isinstance(raw_spans, list)
        or not isinstance(raw_actions, list)
        or type(maximum_stack) is not int
    ):
        raise PushdownCompilerError("stack program metadata differs")
    spans = []
    for span in raw_spans:
        start, end, surface = span.get("start"), span.get("end"), span.get("surface")
        if type(start) is not int or type(end) is not int or not isinstance(surface, str):
            raise PushdownCompilerError("number span differs")
        if question[start:end] != surface:
            raise PushdownCompilerError("number span source ownership differs")
        spans.append(NumberSpan(start, end, surface, Fraction(span["magnitude"])))
    if not 1 <= len(spans) <= MAX_SOURCE_NUMBERS:
        raise PushdownCompilerError("number span count differs")
    actions = []
    for raw in raw_actions:
        name = raw.get("action")
        if name not in ACTION_TO_ID:
            raise PushdownCompilerError("stack action differs")
        source_index = int(raw.get("source_index", -1))
        if name == "PUSH" and not 0 <= source_index < len(spans):
            raise PushdownCompilerError("PUSH pointer differs")
        if name != "PUSH" and source_index != -1:
            raise PushdownCompilerError("non-PUSH action has a pointer")
        actions.append(StackAction(ACTION_TO_ID[name], source_index))
    if not 1 <= len(actions) <= MAX_ACTIONS or actions[-1].action != STOP:
        raise PushdownCompilerError("action sequence geometry differs")
    if not 1 <= maximum_stack <= MAX_STACK:
        raise PushdownCompilerError("stack depth exceeds schema")
    return StackProgram(identity, family, question, tuple(spans), tuple(actions), maximum_stack)


def stack_labels(programs: Sequence[StackProgram], device: torch.device) -> dict[str, torch.Tensor]:
    action = torch.full(
        (len(programs), MAX_ACTIONS), -100, dtype=torch.long, device=device
    )
    pointer = torch.full_like(action, -100)
    candidate_count = torch.tensor(
        [len(program.number_spans) for program in programs], dtype=torch.long, device=device
    )
    for row, program in enumerate(programs):
        for column, item in enumerate(program.actions):
            action[row, column] = item.action
            if item.action == PUSH:
                pointer[row, column] = item.source_index
    return {"action": action, "pointer": pointer, "candidate_count": candidate_count}


class PushdownStackCompiler(nn.Module):
    """A recurrent source reader whose hard actions own a bounded tensor stack."""

    def __init__(
        self,
        source_width: int,
        *,
        width: int = 512,
        encoder_layers: int = 4,
        heads: int = 8,
    ) -> None:
        super().__init__()
        if width % heads or encoder_layers <= 0:
            raise PushdownCompilerError("compiler geometry differs")
        self.width = width
        self.source_projection = nn.Linear(source_width, width, bias=False)
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            4 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.source_encoder = nn.TransformerEncoder(
            layer, encoder_layers, enable_nested_tensor=False
        )
        self.source_norm = nn.LayerNorm(width)
        self.initial_state = nn.Linear(width, width)
        self.cross_attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.stack_reader = nn.Sequential(
            nn.LayerNorm(3 * width),
            nn.Linear(3 * width, width),
            nn.GELU(),
        )
        self.recurrent_cell = nn.GRUCell(width, width)
        self.action_head = nn.Linear(width, len(ACTIONS))
        self.pointer_query = nn.Linear(width, width, bias=False)
        self.pointer_key = nn.Linear(width, width, bias=False)
        self.push_writer = nn.Sequential(
            nn.LayerNorm(2 * width), nn.Linear(2 * width, width), nn.GELU()
        )
        self.negate_writer = nn.Sequential(
            nn.LayerNorm(2 * width), nn.Linear(2 * width, width), nn.GELU()
        )
        self.apply_writer = nn.Sequential(
            nn.LayerNorm(4 * width),
            nn.Linear(4 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.action_embedding = nn.Embedding(len(ACTIONS), width)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(values.dtype).unsqueeze(-1)
        return (values * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def _candidate_states(
        self, memory: torch.Tensor, candidate_token_mask: torch.Tensor
    ) -> torch.Tensor:
        weights = candidate_token_mask.to(memory.dtype)
        states = torch.einsum("bcl,blh->bch", weights, memory)
        return states / weights.sum(-1, keepdim=True).clamp_min(1.0)

    @staticmethod
    def _top(stack: torch.Tensor, depth: torch.Tensor, offset: int) -> torch.Tensor:
        index = (depth - 1 - offset).clamp_min(0)
        gathered = stack[torch.arange(stack.shape[0], device=stack.device), index]
        return gathered * (depth > offset).to(stack.dtype).unsqueeze(-1)

    def forward(
        self,
        source_features: torch.Tensor,
        source_mask: torch.Tensor,
        candidate_token_mask: torch.Tensor,
        candidate_count: torch.Tensor,
        *,
        gold: dict[str, torch.Tensor] | None = None,
        feedback: str = "hard",
        reset_stack: bool = False,
        permute_stack_top: bool = False,
    ) -> StackCompilerOutput:
        if feedback not in {"hard", "gold"} or (feedback == "gold" and gold is None):
            raise PushdownCompilerError("feedback mode differs")
        memory = self.source_projection(source_features)
        memory = self.source_encoder(memory, src_key_padding_mask=~source_mask.bool())
        memory = self.source_norm(memory)
        global_state = self._masked_mean(memory, source_mask.bool())
        state = torch.tanh(self.initial_state(global_state))
        candidates = self._candidate_states(memory, candidate_token_mask)
        batch = source_features.shape[0]
        stack = memory.new_zeros(batch, MAX_STACK, self.width)
        depth = torch.zeros(batch, dtype=torch.long, device=memory.device)
        stopped = torch.zeros(batch, dtype=torch.bool, device=memory.device)
        action_outputs = []
        pointer_outputs = []
        depth_outputs = []
        chosen_action_outputs = []
        chosen_pointer_outputs = []
        invalid_count = torch.zeros((), dtype=torch.long, device=memory.device)
        rows = torch.arange(batch, device=memory.device)

        for action_index in range(MAX_ACTIONS):
            if reset_stack and action_index:
                stack.zero_()
                depth.zero_()
                stopped.zero_()
            top = self._top(stack, depth, 0)
            second = self._top(stack, depth, 1)
            if permute_stack_top and action_index and batch > 1:
                top = top.roll(1, 0)
            context, _ = self.cross_attention(
                state.unsqueeze(1),
                memory,
                memory,
                key_padding_mask=~source_mask.bool(),
                need_weights=False,
            )
            recurrent_input = self.stack_reader(
                torch.cat((context[:, 0], top, second), dim=-1)
            )
            state = self.recurrent_cell(recurrent_input, state)
            action_logits = self.action_head(state)
            pointer_logits = torch.einsum(
                "bh,bch->bc", self.pointer_query(state), self.pointer_key(candidates)
            ) / (self.width**0.5)
            valid_candidates = (
                torch.arange(MAX_SOURCE_NUMBERS, device=memory.device)[None, :]
                < candidate_count[:, None]
            )
            pointer_logits = pointer_logits.masked_fill(~valid_candidates, -1e9)

            valid_actions = torch.zeros(
                batch, len(ACTIONS), dtype=torch.bool, device=memory.device
            )
            valid_actions[:, PUSH] = depth < MAX_STACK
            valid_actions[:, NEGATE] = depth >= 1
            valid_actions[:, APPLY_BEGIN:APPLY_END] = (depth >= 2).unsqueeze(-1)
            valid_actions[:, STOP] = depth == 1
            valid_actions[stopped] = False
            valid_actions[stopped, STOP] = True
            masked_action_logits = action_logits.masked_fill(~valid_actions, -1e9)

            if feedback == "gold":
                raw_action = gold["action"][:, action_index]
                raw_pointer = gold["pointer"][:, action_index]
                chosen_action = torch.where(
                    raw_action == -100, torch.full_like(raw_action, STOP), raw_action
                )
                chosen_pointer = torch.where(
                    raw_pointer == -100, torch.zeros_like(raw_pointer), raw_pointer
                )
            else:
                chosen_action = masked_action_logits.argmax(-1)
                chosen_pointer = pointer_logits.argmax(-1)
            chosen_valid = valid_actions[rows, chosen_action]
            invalid_count = invalid_count + (~chosen_valid & ~stopped).sum()

            active = ~stopped
            push_rows = rows[active & (chosen_action == PUSH)]
            if len(push_rows):
                selected = candidates[push_rows, chosen_pointer[push_rows]]
                written = self.push_writer(
                    torch.cat((selected, state[push_rows]), dim=-1)
                )
                stack[push_rows, depth[push_rows]] = written
                depth[push_rows] += 1
            negate_rows = rows[active & (chosen_action == NEGATE) & (depth >= 1)]
            if len(negate_rows):
                positions = depth[negate_rows] - 1
                current = stack[negate_rows, positions]
                stack[negate_rows, positions] = self.negate_writer(
                    torch.cat((current, self.action_embedding(chosen_action[negate_rows])), dim=-1)
                )
            apply_rows = rows[
                active
                & (chosen_action >= APPLY_BEGIN)
                & (chosen_action < APPLY_END)
                & (depth >= 2)
            ]
            if len(apply_rows):
                right_position = depth[apply_rows] - 1
                left_position = depth[apply_rows] - 2
                right = stack[apply_rows, right_position]
                left = stack[apply_rows, left_position]
                result = self.apply_writer(
                    torch.cat(
                        (
                            left,
                            right,
                            self.action_embedding(chosen_action[apply_rows]),
                            state[apply_rows],
                        ),
                        dim=-1,
                    )
                )
                stack[apply_rows, left_position] = result
                stack[apply_rows, right_position] = 0
                depth[apply_rows] -= 1
            stopped |= active & (chosen_action == STOP) & (depth == 1)
            state = state + self.action_embedding(chosen_action)
            action_outputs.append(action_logits)
            pointer_outputs.append(pointer_logits)
            chosen_action_outputs.append(chosen_action)
            chosen_pointer_outputs.append(chosen_pointer)
            depth_outputs.append(depth.clone())

        return StackCompilerOutput(
            action_logits=torch.stack(action_outputs, dim=1),
            pointer_logits=torch.stack(pointer_outputs, dim=1),
            chosen_actions=torch.stack(chosen_action_outputs, dim=1),
            chosen_pointers=torch.stack(chosen_pointer_outputs, dim=1),
            stack_depths=torch.stack(depth_outputs, dim=1),
            invalid_action_count=invalid_count,
        )


def stack_loss(
    output: StackCompilerOutput, labels: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    active = labels["action"] != -100
    push = labels["action"] == PUSH
    action_loss = F.cross_entropy(output.action_logits[active], labels["action"][active])
    if not push.any():
        raise PushdownCompilerError("batch has no PUSH actions")
    pointer_loss = F.cross_entropy(output.pointer_logits[push], labels["pointer"][push])
    return action_loss + pointer_loss, {"action": action_loss, "pointer": pointer_loss}
