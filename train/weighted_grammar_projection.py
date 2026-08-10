"""Weighted complete-path grammar projection for frozen BTT1 logits."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from byte_tape_compiler import ROLE_TO_ID, ROLES


@dataclass(frozen=True, slots=True)
class GrammarState:
    score: float
    roles: tuple[int, ...]
    started: bool
    expecting_operand: bool
    parenthesis_depth: int
    in_number: bool
    number_digits: int
    number_dots: int


def _finish_number(state: GrammarState) -> GrammarState | None:
    if not state.in_number:
        return state
    if state.number_digits <= 0 or state.number_dots > 1:
        return None
    return GrammarState(
        state.score, state.roles, state.started, False, state.parenthesis_depth,
        False, 0, 0,
    )


def _transition(state: GrammarState, role: int, byte: int, score: float) -> GrammarState | None:
    name = ROLES[role]
    current = state
    if current.in_number and name != "NUM_CONT":
        current = _finish_number(current)
        if current is None:
            return None
    roles = current.roles + (role,)
    total = current.score + score
    if name == "IGNORE":
        return GrammarState(total, roles, current.started, current.expecting_operand, current.parenthesis_depth, False, 0, 0)
    if name == "NUM_CONT":
        if not current.in_number or not (byte == 46 or 48 <= byte <= 57):
            return None
        return GrammarState(
            total, roles, True, True, current.parenthesis_depth, True,
            current.number_digits + int(48 <= byte <= 57), current.number_dots + int(byte == 46),
        )
    if name == "NUM_BEGIN":
        if current.started and not current.expecting_operand:
            return None
        if not (byte == 46 or 48 <= byte <= 57):
            return None
        return GrammarState(total, roles, True, True, current.parenthesis_depth, True, int(48 <= byte <= 57), int(byte == 46))
    if name in {"NEGATE", "LPAREN"}:
        if current.started and not current.expecting_operand:
            return None
        depth = current.parenthesis_depth + int(name == "LPAREN")
        return GrammarState(total, roles, True, True, depth, False, 0, 0)
    if name in {"ADD", "SUB", "MUL", "DIV"}:
        if not current.started or current.expecting_operand:
            return None
        return GrammarState(total, roles, True, True, current.parenthesis_depth, False, 0, 0)
    if name == "RPAREN":
        if not current.started or current.expecting_operand or current.parenthesis_depth <= 0:
            return None
        return GrammarState(total, roles, True, False, current.parenthesis_depth - 1, False, 0, 0)
    return None


def _state_key(state: GrammarState) -> tuple[bool, bool, int, bool, bool, int]:
    """Future grammar legality depends only on this state quotient."""
    return (
        state.started,
        state.expecting_operand,
        state.parenthesis_depth,
        state.in_number,
        state.number_digits > 0,
        state.number_dots,
    )


def project_role_logits(logits: torch.Tensor, source_bytes: Sequence[int], *, beam_width: int = 64) -> list[int] | None:
    """Return the maximum-score complete grammatical role path."""
    if logits.ndim != 2 or logits.shape[0] != len(source_bytes) or logits.shape[1] != len(ROLES):
        raise ValueError("grammar projection geometry differs")
    log_probs = logits.float().log_softmax(-1).cpu()
    top1 = log_probs.argmax(-1).tolist()
    state = GrammarState(0.0, (), False, True, 0, False, 0, 0)
    for position, role in enumerate(top1):
        state = _transition(state, role, int(source_bytes[position]), float(log_probs[position, role]))
        if state is None:
            break
    if state is not None:
        final = _finish_number(state)
        if final is not None and final.started and not final.expecting_operand and final.parenthesis_depth == 0:
            return top1
    beam = [GrammarState(0.0, (), False, True, 0, False, 0, 0)]
    for position, byte in enumerate(source_bytes):
        candidates = []
        role_order = torch.argsort(log_probs[position], descending=True).tolist()
        for state in beam:
            for role in role_order:
                updated = _transition(state, role, int(byte), float(log_probs[position, role]))
                if updated is not None and math.isfinite(updated.score):
                    candidates.append(updated)
        best_by_state: dict[tuple[bool, bool, int, bool, bool, int], GrammarState] = {}
        for candidate in candidates:
            key = _state_key(candidate)
            incumbent = best_by_state.get(key)
            if incumbent is None or candidate.score > incumbent.score:
                best_by_state[key] = candidate
        beam = sorted(best_by_state.values(), key=lambda state: state.score, reverse=True)[:beam_width]
        if not beam:
            return None
    valid = []
    for state in beam:
        final = _finish_number(state)
        if final is not None and final.started and not final.expecting_operand and final.parenthesis_depth == 0:
            valid.append(final)
    if not valid:
        return None
    return list(max(valid, key=lambda state: state.score).roles)
