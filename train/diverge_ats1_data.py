"""Supervisor-only byte roles for DIVERGE-ATS1 source compilation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Sequence


class ATS1DataError(RuntimeError):
    """A source segment cannot satisfy the frozen ATS1 contract."""


PAD_ID = 0
CLS_ID = 1
BYTE_OFFSET = 2
BYTE_VOCAB_SIZE = 130
MAX_SEGMENT_BYTES = 80

ROLE_NAMES = (
    "OTHER",
    "LHS_A",
    "LHS_B",
    "LHS_SYMBOL",
    "RHS_A",
    "RHS_B",
    "RHS_SYMBOL",
    "ARG1",
    "ARG2",
)
ROLE_TO_ID = {name: index for index, name in enumerate(ROLE_NAMES)}

OPERATION_NAMES = (
    "SCALAR_ADD",
    "SCALAR_SUBTRACT",
    "SCALAR_MULTIPLY",
    "REGISTER_A_ADD_B",
    "REGISTER_B_SUB_A",
    "REGISTER_SWAP",
    "REGISTER_A_DOUBLE",
    "REGISTER_B_ADD_A",
    "SYMBOL_REVERSE",
    "SYMBOL_ROTATE",
    "SYMBOL_SWAP",
)
OPERATION_TO_ID = {name: index for index, name in enumerate(OPERATION_NAMES)}

FAMILY_NAMES = ("scalar", "register", "symbolic")
FAMILY_TO_ID = {name: index for index, name in enumerate(FAMILY_NAMES)}


@dataclass(frozen=True, slots=True)
class SegmentTarget:
    text: str
    byte_ids: tuple[int, ...]
    role_ids: tuple[int, ...]
    operation_id: int
    family_id: int
    lhs_a: str | None
    lhs_b: str | None
    lhs_symbol: str | None
    rhs_a: str | None
    rhs_b: str | None
    rhs_symbol: str | None
    arguments: tuple[str, ...]
    identity_sha256: str
    step_index: int
    trace_kind: str


_SCALAR = re.compile(
    r"^Step\s+\d+:\s+(-?\d+)\s+([+\-*])\s+(\d+)\s+=\s+(-?\d+)\.$"
)
_REGISTER = re.compile(
    r"^Step\s+\d+:\s+(.+?):\s+\(A=(-?\d+), B=(-?\d+)\)\s+->\s+"
    r"\(A=(-?\d+), B=(-?\d+)\)\.$"
)
_SYMBOLIC = re.compile(
    r"^Step\s+\d+:\s+(.+?):\s+([a-z]+)\s+->\s+([a-z]+)\.$"
)


def encode_bytes(text: str) -> tuple[int, ...]:
    if not text or len(text.encode("ascii")) + 1 > MAX_SEGMENT_BYTES:
        raise ATS1DataError("source segment is empty, non-ASCII, or too long")
    return (CLS_ID, *(ord(character) + BYTE_OFFSET for character in text))


def operation_id(family: str, operation: Any) -> int:
    if family == "scalar":
        kind, _ = operation
        names = {
            "add": "SCALAR_ADD",
            "subtract": "SCALAR_SUBTRACT",
            "multiply": "SCALAR_MULTIPLY",
        }
    elif family == "register":
        names = {
            "A+=B": "REGISTER_A_ADD_B",
            "B-=A": "REGISTER_B_SUB_A",
            "swap": "REGISTER_SWAP",
            "A*=2": "REGISTER_A_DOUBLE",
            "B+=A": "REGISTER_B_ADD_A",
        }
        kind = str(operation)
    elif family == "symbolic":
        kind, _, _ = operation
        names = {
            "reverse": "SYMBOL_REVERSE",
            "rotate": "SYMBOL_ROTATE",
            "swap": "SYMBOL_SWAP",
        }
    else:
        raise ATS1DataError("unknown ATS1 family")
    try:
        return OPERATION_TO_ID[names[kind]]
    except KeyError as error:
        raise ATS1DataError(f"unknown operation {operation!r}") from error


def _mark(roles: list[int], span: tuple[int, int], role: str) -> None:
    start, end = span
    if not 0 <= start < end <= len(roles) - 1:
        raise ATS1DataError("role span escaped the source segment")
    role_id = ROLE_TO_ID[role]
    for offset in range(start, end):
        position = offset + 1
        if roles[position] != ROLE_TO_ID["OTHER"]:
            raise ATS1DataError("source role spans overlap")
        roles[position] = role_id


def _argument_spans(text: str, span: tuple[int, int]) -> list[tuple[int, int]]:
    start, end = span
    return [
        (start + match.start(), start + match.end())
        for match in re.finditer(r"\d+", text[start:end])
    ]


def segment_target(
    row: dict[str, Any],
    step_index: int,
    *,
    trace_kind: str,
) -> SegmentTarget:
    if trace_kind not in {"wrong", "correct"}:
        raise ATS1DataError("trace kind differs")
    family = str(row.get("family"))
    steps = row.get(f"{trace_kind}_steps")
    program = row.get("program")
    if (
        not isinstance(steps, list)
        or not isinstance(program, list)
        or len(steps) != len(program)
        or not 0 <= step_index < len(steps)
    ):
        raise ATS1DataError("row step/program contract differs")
    text = str(steps[step_index])
    byte_ids = encode_bytes(text)
    roles = [ROLE_TO_ID["OTHER"]] * len(byte_ids)
    lhs_a = lhs_b = lhs_symbol = None
    rhs_a = rhs_b = rhs_symbol = None
    arguments: tuple[str, ...]

    if family == "scalar":
        match = _SCALAR.fullmatch(text)
        if match is None:
            raise ATS1DataError("scalar renderer differs")
        lhs_a, rhs_a = match.group(1), match.group(4)
        arguments = (match.group(3),)
        _mark(roles, match.span(1), "LHS_A")
        _mark(roles, match.span(4), "RHS_A")
        _mark(roles, match.span(3), "ARG1")
    elif family == "register":
        match = _REGISTER.fullmatch(text)
        if match is None:
            raise ATS1DataError("register renderer differs")
        lhs_a, lhs_b = match.group(2), match.group(3)
        rhs_a, rhs_b = match.group(4), match.group(5)
        arguments = ()
        for group, role in ((2, "LHS_A"), (3, "LHS_B"), (4, "RHS_A"), (5, "RHS_B")):
            _mark(roles, match.span(group), role)
    elif family == "symbolic":
        match = _SYMBOLIC.fullmatch(text)
        if match is None:
            raise ATS1DataError("symbolic renderer differs")
        lhs_symbol, rhs_symbol = match.group(2), match.group(3)
        _mark(roles, match.span(2), "LHS_SYMBOL")
        _mark(roles, match.span(3), "RHS_SYMBOL")
        spans = _argument_spans(text, match.span(1))
        operation = program[step_index]
        kind = str(operation[0])
        needed = 0 if kind == "reverse" else (1 if kind == "rotate" else 2)
        if len(spans) != needed:
            raise ATS1DataError("symbolic argument renderer differs")
        for index, span in enumerate(spans):
            _mark(roles, span, f"ARG{index + 1}")
        arguments = tuple(text[start:end] for start, end in spans)
    else:
        raise ATS1DataError("unknown ATS1 family")

    expected_arguments: tuple[str, ...]
    operation = program[step_index]
    if family == "scalar":
        expected_arguments = (str(int(operation[1])),)
    elif family == "symbolic" and operation[0] == "rotate":
        expected_arguments = (str(int(operation[1])),)
    elif family == "symbolic" and operation[0] == "swap":
        expected_arguments = (str(int(operation[1])), str(int(operation[2])))
    else:
        expected_arguments = ()
    if arguments != expected_arguments:
        raise ATS1DataError("rendered operation arguments differ from program")

    return SegmentTarget(
        text=text,
        byte_ids=byte_ids,
        role_ids=tuple(roles),
        operation_id=operation_id(family, operation),
        family_id=FAMILY_TO_ID[family],
        lhs_a=lhs_a,
        lhs_b=lhs_b,
        lhs_symbol=lhs_symbol,
        rhs_a=rhs_a,
        rhs_b=rhs_b,
        rhs_symbol=rhs_symbol,
        arguments=arguments,
        identity_sha256=str(row["identity_sha256"]),
        step_index=step_index,
        trace_kind=trace_kind,
    )


def build_segments(
    rows: Iterable[dict[str, Any]],
    *,
    trace_kinds: Sequence[str] = ("wrong", "correct"),
) -> list[SegmentTarget]:
    output: list[SegmentTarget] = []
    for row in rows:
        depth = int(row.get("depth", -1))
        for trace_kind in trace_kinds:
            for step_index in range(depth):
                output.append(segment_target(row, step_index, trace_kind=trace_kind))
    if not output:
        raise ATS1DataError("ATS1 segment board is empty")
    return output


__all__ = [
    "ATS1DataError",
    "BYTE_OFFSET",
    "BYTE_VOCAB_SIZE",
    "CLS_ID",
    "FAMILY_NAMES",
    "FAMILY_TO_ID",
    "MAX_SEGMENT_BYTES",
    "OPERATION_NAMES",
    "OPERATION_TO_ID",
    "PAD_ID",
    "ROLE_NAMES",
    "ROLE_TO_ID",
    "SegmentTarget",
    "build_segments",
    "encode_bytes",
    "operation_id",
    "segment_target",
]
