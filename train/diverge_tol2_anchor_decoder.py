"""Document-owned symbol table and anchor relations for DIVERGE-TOL2."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

from diverge_tol1_data import SourceCandidate, source_candidates
from diverge_tol1_ir import Action, Atom, DIRECT_OPS, Instruction, Predicate


class TOL2DecodeError(RuntimeError):
    """A source clause cannot form one typed anchor-relational packet."""


ACTION_ANCHORS = {
    "SET": ("set", "assign", "make"),
    "ADD": ("add", "increase"),
    "SUBTRACT": ("subtract", "decrease"),
    "MULTIPLY": ("multiply", "scale"),
    "SWAP": ("swap", "exchange"),
    "QUERY": ("report", "return"),
}


@dataclass(frozen=True, slots=True)
class GuardRegions:
    predicate: str
    true_action: str
    false_action: str


def _strip_clause(value: str) -> str:
    return value.strip().rstrip(".").strip()


def split_guard(text: str) -> GuardRegions:
    """Partition the three admitted source orders without reading semantics."""

    source = _strip_clause(text)
    if source.startswith("if ") and ", then " in source:
        predicate, remainder = source[3:].split(", then ", 1)
        marker = "; otherwise, " if "; otherwise, " in remainder else "; otherwise "
        if marker not in remainder:
            raise TOL2DecodeError("leading-if guard lacks otherwise")
        true_action, false_action = remainder.split(marker, 1)
    elif source.startswith("when ") and ": " in source and "; else " in source:
        predicate, remainder = source[5:].split(": ", 1)
        true_action, false_action = remainder.split("; else ", 1)
    elif " if " in source and "; otherwise " in source:
        true_action, remainder = source.split(" if ", 1)
        predicate, false_action = remainder.split("; otherwise ", 1)
    elif source.startswith("otherwise ") and "; if " in source and ", then " in source:
        false_action, remainder = source[len("otherwise ") :].split("; if ", 1)
        predicate, true_action = remainder.split(", then ", 1)
    else:
        raise TOL2DecodeError("unknown guard-region order")
    values = tuple(_strip_clause(value) for value in (predicate, true_action, false_action))
    if any(not value for value in values):
        raise TOL2DecodeError("empty guard region")
    return GuardRegions(*values)


def _word_position(text: str, words: Iterable[str]) -> tuple[int, int]:
    matches = [
        match
        for word in words
        for match in re.finditer(rf"\b{re.escape(word)}\b", text)
    ]
    if len(matches) != 1:
        raise TOL2DecodeError("operation anchor is missing or ambiguous")
    return matches[0].start(), matches[0].end()


def _relation_position(text: str, word: str) -> tuple[int, int] | None:
    matches = list(re.finditer(rf"\b{re.escape(word)}\b", text))
    if not matches:
        return None
    # The admitted action grammar contains at most one argument relation word.
    if len(matches) != 1:
        raise TOL2DecodeError("argument relation is ambiguous")
    return matches[0].start(), matches[0].end()


def _semantic_candidates(
    text: str,
    symbols: Sequence[str] | None,
) -> tuple[SourceCandidate, ...]:
    candidates = source_candidates(text)
    symbol_set = set(symbols or ())
    if symbols is None:
        return candidates
    return tuple(
        candidate
        for candidate in candidates
        if candidate.kind == "NUMBER" or candidate.text in symbol_set
    )


def _first_after(
    candidates: Sequence[SourceCandidate],
    position: int,
    *,
    words_only: bool,
) -> SourceCandidate:
    admitted = [
        candidate
        for candidate in candidates
        if candidate.start >= position and (not words_only or candidate.kind == "WORD")
    ]
    if not admitted:
        raise TOL2DecodeError("typed relation has no following argument")
    return min(admitted, key=lambda value: (value.start, value.end))


def _atom(candidate: SourceCandidate) -> Atom:
    return Atom("CONST" if candidate.kind == "NUMBER" else "REF", candidate.text)


def decode_direct_action(
    text: str,
    operation: str,
    symbols: Sequence[str] | None,
) -> Action:
    if operation not in DIRECT_OPS:
        raise TOL2DecodeError("direct action opcode differs")
    candidates = _semantic_candidates(text, symbols)
    _, anchor_end = _word_position(text, ACTION_ANCHORS[operation])
    relation = {
        "SET": "into",
        "ADD": "to",
        "SUBTRACT": "from",
        "MULTIPLY": None,
    }[operation]
    relation_span = _relation_position(text, relation) if relation else None
    target = _first_after(
        candidates,
        relation_span[1] if relation_span is not None else anchor_end,
        words_only=True,
    )
    remaining = [candidate for candidate in candidates if candidate is not target]
    if symbols is None:
        numeric = [candidate for candidate in remaining if candidate.kind == "NUMBER"]
        if len(numeric) == 1:
            operand = numeric[0]
        else:
            operand = _first_after(remaining, anchor_end, words_only=False)
    else:
        if len(remaining) != 1:
            raise TOL2DecodeError("direct action does not have two typed arguments")
        operand = remaining[0]
    action = Action(operation, target.text, _atom(operand))
    action.validate()
    return action


def decode_swap(text: str, symbols: Sequence[str]) -> Instruction:
    candidates = [
        value for value in _semantic_candidates(text, symbols) if value.kind == "WORD"
    ]
    if len(candidates) != 2 or candidates[0].text == candidates[1].text:
        raise TOL2DecodeError("swap does not name two distinct registers")
    left, right = sorted((candidates[0].text, candidates[1].text))
    return Instruction("SWAP", swap_left=left, swap_right=right)


def decode_query(text: str, symbols: Sequence[str]) -> Instruction:
    candidates = [
        value for value in _semantic_candidates(text, symbols) if value.kind == "WORD"
    ]
    if len(candidates) != 1:
        raise TOL2DecodeError("query does not identify exactly one register")
    return Instruction("QUERY", query=candidates[0].text)


def decode_predicate(
    text: str,
    comparator: str,
    symbols: Sequence[str],
) -> Predicate:
    candidates = _semantic_candidates(text, symbols)
    if len(candidates) != 2 or candidates[0].kind != "WORD":
        raise TOL2DecodeError("predicate does not expose left and right atoms")
    predicate = Predicate(comparator, candidates[0].text, _atom(candidates[1]))
    predicate.validate()
    return predicate


def canonical_instruction(instruction: Instruction) -> Instruction:
    if instruction.operation != "SWAP":
        return instruction
    assert instruction.swap_left and instruction.swap_right
    left, right = sorted((instruction.swap_left, instruction.swap_right))
    return Instruction("SWAP", swap_left=left, swap_right=right)


def semantic_instruction_equal(left: Instruction, right: Instruction) -> bool:
    return canonical_instruction(left) == canonical_instruction(right)


__all__ = [
    "ACTION_ANCHORS",
    "GuardRegions",
    "TOL2DecodeError",
    "canonical_instruction",
    "decode_direct_action",
    "decode_predicate",
    "decode_query",
    "decode_swap",
    "semantic_instruction_equal",
    "split_guard",
]
