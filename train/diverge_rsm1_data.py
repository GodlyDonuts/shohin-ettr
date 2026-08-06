"""Supervisor-only state trajectories for persistent CRP1 replay."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence

from diverge_crp1_data import render_revision_prompt, tokenize_revision_example


PAD_TOKEN = 0
EOS_TOKEN = 1
STATE_CHARACTERS = "-0123456789,abcdefghijklmnopqrstuvwxyz"
CHAR_TO_TOKEN = {value: index + 2 for index, value in enumerate(STATE_CHARACTERS)}
TOKEN_TO_CHAR = {value: key for key, value in CHAR_TO_TOKEN.items()}
STATE_VOCAB_SIZE = len(STATE_CHARACTERS) + 2
MAX_STATE_SLOTS = 24


class RSM1DataError(RuntimeError):
    """The replay supervisor or rendered-state contract differs."""


@dataclass(frozen=True)
class ReplayTokens:
    """Frozen prompt segmentation for packet selection and state replay."""

    prompt_ids: list[int]
    problem_mask: list[bool]
    packet_step_masks: list[list[bool]]
    operation_masks: list[list[bool]]
    final_mask: list[bool]


@dataclass(frozen=True)
class ReplaySupervision:
    """Canonical fixed-width targets; exact semantics stay assessor-owned."""

    selection: int
    initial: tuple[int, ...]
    free_targets: tuple[tuple[int, ...], ...]
    free_active: tuple[bool, ...]
    oracle_predecessors: tuple[tuple[int, ...], ...]
    oracle_targets: tuple[tuple[int, ...], ...]
    oracle_active: tuple[bool, ...]
    terminal: tuple[int, ...]


_SCALAR_STEP = re.compile(
    r"^Step\s+\d+:\s+-?\d+\s+([+\-*])\s+(\d+)\s+=\s+-?\d+\.$"
)
_NAMED_STEP = re.compile(r"^Step\s+\d+:\s+(.+?):\s+.+?\s+->\s+.+\.$")


def encode_state(value: str, *, slots: int = MAX_STATE_SLOTS) -> tuple[int, ...]:
    text = str(value)
    if not text or slots < 2 or len(text) + 1 > slots:
        raise RSM1DataError("state text does not fit the fixed replay packet")
    try:
        encoded = [CHAR_TO_TOKEN[character] for character in text]
    except KeyError as error:
        raise RSM1DataError(f"state contains unsupported character {error.args[0]!r}") from error
    return tuple(encoded + [EOS_TOKEN] + [PAD_TOKEN] * (slots - len(encoded) - 1))


def decode_state(tokens: Sequence[int]) -> str:
    output: list[str] = []
    saw_eos = False
    for raw in tokens:
        token = int(raw)
        if saw_eos:
            if token != PAD_TOKEN:
                raise RSM1DataError("non-padding token follows state EOS")
            continue
        if token == EOS_TOKEN:
            saw_eos = True
        elif token == PAD_TOKEN:
            raise RSM1DataError("state padding precedes EOS")
        elif token in TOKEN_TO_CHAR:
            output.append(TOKEN_TO_CHAR[token])
        else:
            raise RSM1DataError("state token is outside the replay vocabulary")
    if not saw_eos or not output:
        raise RSM1DataError("state packet is empty or unterminated")
    return "".join(output)


def _format_state(family: str, state: Any) -> str:
    if family == "scalar":
        return str(int(state))
    if family == "register":
        if not isinstance(state, (tuple, list)) or len(state) != 2:
            raise RSM1DataError("register state differs")
        return f"{int(state[0])},{int(state[1])}"
    if family == "symbolic":
        text = str(state)
        if not text.isalpha() or not text.islower():
            raise RSM1DataError("symbolic state differs")
        return text
    raise RSM1DataError("unknown replay family")


def _apply_scalar(state: int, operation: Sequence[Any]) -> int:
    kind, raw = operation
    value = int(raw)
    if kind == "add":
        return state + value
    if kind == "subtract":
        return state - value
    if kind == "multiply":
        return state * value
    raise RSM1DataError("unknown scalar operation")


def _apply_register(state: tuple[int, int], operation: str) -> tuple[int, int]:
    left, right = state
    if operation == "A+=B":
        return left + right, right
    if operation == "B-=A":
        return left, right - left
    if operation == "swap":
        return right, left
    if operation == "A*=2":
        return 2 * left, right
    if operation == "B+=A":
        return left, right + left
    raise RSM1DataError("unknown register operation")


def _apply_symbolic(state: str, operation: Sequence[Any]) -> str:
    kind, raw_left, raw_right = operation
    left, right = int(raw_left), int(raw_right)
    if kind == "reverse":
        return state[::-1]
    if kind == "rotate":
        return state[left:] + state[:left]
    if kind == "swap":
        values = list(state)
        values[left - 1], values[right - 1] = values[right - 1], values[left - 1]
        return "".join(values)
    raise RSM1DataError("unknown symbolic operation")


def supervisor_states(row: dict[str, Any]) -> tuple[str, ...]:
    """Return initial plus every correct successor; never used as model input."""

    family = str(row.get("family"))
    program = row.get("program")
    if not isinstance(program, list) or len(program) != int(row.get("depth", -1)):
        raise RSM1DataError("program/depth contract differs")
    raw_initial = row.get("initial_state")
    if family == "scalar":
        state: Any = int(raw_initial)
        apply = _apply_scalar
    elif family == "register":
        if not isinstance(raw_initial, list) or len(raw_initial) != 2:
            raise RSM1DataError("initial register state differs")
        state = (int(raw_initial[0]), int(raw_initial[1]))
        apply = _apply_register
    elif family == "symbolic":
        state = str(raw_initial)
        apply = _apply_symbolic
    else:
        raise RSM1DataError("unknown replay family")
    states = [_format_state(family, state)]
    for operation in program:
        state = apply(state, operation)
        states.append(_format_state(family, state))
    if states[-1] != str(row.get("answer")):
        raise RSM1DataError("supervisor terminal state differs from board answer")
    if len(states) != len(row.get("correct_steps", [])) + 1:
        raise RSM1DataError("supervisor state/step count differs")
    for value in states:
        if decode_state(encode_state(value)) != value:
            raise RSM1DataError("state-code round trip differs")
    return tuple(states)


def operation_surfaces(row: dict[str, Any]) -> tuple[str, ...]:
    """Extract rendered operation surfaces without exposing supervisor op IDs."""

    family = str(row.get("family"))
    steps = row.get("wrong_steps")
    if not isinstance(steps, list) or len(steps) != int(row.get("depth", -1)):
        raise RSM1DataError("wrong trace/depth contract differs")
    output: list[str] = []
    for raw in steps:
        text = str(raw)
        if family == "scalar":
            match = _SCALAR_STEP.fullmatch(text)
            if match is None:
                raise RSM1DataError("scalar step renderer differs")
            output.append(f"{match.group(1)} {match.group(2)}")
        else:
            match = _NAMED_STEP.fullmatch(text)
            if match is None:
                raise RSM1DataError("named step renderer differs")
            output.append(match.group(1))
    if any("->" in value or "=" in value for value in output):
        raise RSM1DataError("operation surface leaks a rendered successor")
    return tuple(output)


def _content_span(rendered: str, opening: str, closing: str) -> tuple[int, int]:
    if rendered.count(opening) != 1 or rendered.count(closing) != 1:
        raise RSM1DataError("replay markers are not unique")
    start = rendered.index(opening) + len(opening)
    if start < len(rendered) and rendered[start] == "\n":
        start += 1
    end = rendered.index(closing, start)
    while end > start and rendered[end - 1] == "\n":
        end -= 1
    if end <= start:
        raise RSM1DataError("replay step span is empty")
    return start, end


def _overlaps(offset: tuple[int, int], span: tuple[int, int]) -> bool:
    start, end = offset
    return end > span[0] and start < span[1]


def tokenize_replay_example(
    tokenizer: Any,
    row: dict[str, Any],
    trace_steps: Sequence[str],
    draft_final: str,
    *,
    max_sequence_length: int,
    packet_slots: int,
) -> ReplayTokens | None:
    """Tokenize once while exposing only operation phrases to the executor."""

    steps = tuple(map(str, trace_steps))
    if len(steps) != int(row.get("depth", -1)):
        raise RSM1DataError("replay trace/depth contract differs")
    base = tokenize_revision_example(
        tokenizer,
        str(row["problem"]),
        steps,
        str(draft_final),
        None,
        max_sequence_length=max_sequence_length,
        workspace_slots=packet_slots,
    )
    if base is None:
        return None
    rendered = render_revision_prompt(
        tokenizer,
        str(row["problem"]),
        steps,
        str(draft_final),
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    prompt_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [tuple(map(int, value)) for value in encoded["offset_mapping"]]
    if prompt_ids != base.prompt_ids or len(offsets) != len(prompt_ids):
        raise RSM1DataError("replay tokenizer pass differs from packet tokens")

    operation_masks: list[list[bool]] = []
    surfaces = operation_surfaces(row)
    for index, surface in enumerate(surfaces, start=1):
        step_span = _content_span(
            rendered,
            f"<draft_step_{index:02d}>",
            f"</draft_step_{index:02d}>",
        )
        step_text = rendered[step_span[0] : step_span[1]]
        if step_text.count(surface) != 1:
            raise RSM1DataError("operation surface is not unique inside its step")
        operation_start = step_span[0] + step_text.index(surface)
        operation_span = (operation_start, operation_start + len(surface))
        mask = [_overlaps(offset, operation_span) for offset in offsets]
        if not any(mask):
            raise RSM1DataError("tokenizer lost a replay operation surface")
        if any(
            selected and not base.step_masks[index - 1][position]
            for position, selected in enumerate(mask)
        ):
            raise RSM1DataError("operation mask escaped its packet step")
        operation_masks.append(mask)
    return ReplayTokens(
        prompt_ids=base.prompt_ids,
        problem_mask=base.problem_mask,
        packet_step_masks=base.step_masks,
        operation_masks=operation_masks,
        final_mask=base.final_mask,
    )


def replay_targets(
    row: dict[str, Any],
    selection: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    """Return selected start state and per-step targets for assessor/training."""

    states = supervisor_states(row)
    depth = int(row["depth"])
    if not 0 <= selection <= depth:
        raise RSM1DataError("selection is outside the trace")
    if selection == 0:
        terminal = encode_state(states[-1])
        return terminal, ()
    initial = encode_state(states[selection - 1])
    successors = tuple(encode_state(states[index]) for index in range(selection, depth + 1))
    return initial, successors


def build_replay_supervision(
    row: dict[str, Any],
    selection: int,
    *,
    max_trace_steps: int = 12,
) -> ReplaySupervision:
    """Build one unambiguous rollback/replay target tensor contract."""

    states = tuple(encode_state(value) for value in supervisor_states(row))
    depth = int(row["depth"])
    if depth > max_trace_steps or not 0 <= selection <= depth:
        raise RSM1DataError("replay supervision boundary differs")
    initial, successors = replay_targets(row, selection)
    terminal = states[-1]
    free_targets = [initial] * max_trace_steps
    free_active = [False] * max_trace_steps
    if selection > 0:
        if len(successors) != depth - selection + 1:
            raise RSM1DataError("replay successor count differs")
        for step_index in range(selection - 1, depth):
            free_targets[step_index] = states[step_index + 1]
            free_active[step_index] = True
    oracle_predecessors = [terminal] * max_trace_steps
    oracle_targets = [terminal] * max_trace_steps
    oracle_active = [False] * max_trace_steps
    for step_index in range(depth):
        oracle_predecessors[step_index] = states[step_index]
        oracle_targets[step_index] = states[step_index + 1]
        oracle_active[step_index] = True
    return ReplaySupervision(
        selection=selection,
        initial=initial,
        free_targets=tuple(free_targets),
        free_active=tuple(free_active),
        oracle_predecessors=tuple(oracle_predecessors),
        oracle_targets=tuple(oracle_targets),
        oracle_active=tuple(oracle_active),
        terminal=terminal,
    )


__all__ = [
    "CHAR_TO_TOKEN",
    "EOS_TOKEN",
    "MAX_STATE_SLOTS",
    "PAD_TOKEN",
    "RSM1DataError",
    "ReplayTokens",
    "ReplaySupervision",
    "STATE_CHARACTERS",
    "STATE_VOCAB_SIZE",
    "decode_state",
    "build_replay_supervision",
    "encode_state",
    "operation_surfaces",
    "replay_targets",
    "supervisor_states",
    "tokenize_replay_example",
]
