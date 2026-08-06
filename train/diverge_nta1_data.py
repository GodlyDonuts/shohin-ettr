"""Corpus-derived natural arithmetic segments for DIVERGE-NTA1."""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

from diverge_ats1_data import (
    FAMILY_TO_ID,
    OPERATION_TO_ID,
    ROLE_TO_ID,
    SegmentTarget,
    encode_bytes,
)


class NTA1DataError(RuntimeError):
    """A natural arithmetic transaction cannot satisfy the frozen grammar."""


NATURAL_SCALAR = re.compile(
    r"^\s*(?P<lhs>-?\d+)\s*(?P<operator>[+\-*])\s*"
    r"(?P<argument>\d+)\s*=\s*(?P<rhs>-?\d+)\s*$"
)


def natural_segment_target(
    row: dict[str, Any],
    step_index: int,
    *,
    trace_kind: str,
) -> SegmentTarget:
    if trace_kind not in {"wrong", "correct"}:
        raise NTA1DataError("natural trace kind differs")
    steps = row.get(f"{trace_kind}_steps")
    if not isinstance(steps, list) or not 0 <= step_index < len(steps):
        raise NTA1DataError("natural step index differs")
    text = str(steps[step_index])
    match = NATURAL_SCALAR.fullmatch(text)
    if match is None:
        raise NTA1DataError("natural arithmetic transaction differs")
    byte_ids = encode_bytes(text)
    roles = [ROLE_TO_ID["OTHER"]] * len(byte_ids)
    for group, role in (
        ("lhs", "LHS_A"),
        ("argument", "ARG1"),
        ("rhs", "RHS_A"),
    ):
        start, end = match.span(group)
        for position in range(start + 1, end + 1):
            roles[position] = ROLE_TO_ID[role]
    operation = {
        "+": "SCALAR_ADD",
        "-": "SCALAR_SUBTRACT",
        "*": "SCALAR_MULTIPLY",
    }[match.group("operator")]
    return SegmentTarget(
        text=text,
        byte_ids=byte_ids,
        role_ids=tuple(roles),
        operation_id=OPERATION_TO_ID[operation],
        family_id=FAMILY_TO_ID["scalar"],
        lhs_a=match.group("lhs"),
        lhs_b=None,
        lhs_symbol=None,
        rhs_a=match.group("rhs"),
        rhs_b=None,
        rhs_symbol=None,
        arguments=(match.group("argument"),),
        identity_sha256=str(row["identity_sha256"]),
        step_index=step_index,
        trace_kind=trace_kind,
    )


def build_nta1_segments(
    rows: Iterable[dict[str, Any]],
    *,
    trace_kinds: Sequence[str] = ("wrong",),
) -> list[SegmentTarget]:
    output: list[SegmentTarget] = []
    for row in rows:
        for trace_kind in trace_kinds:
            for step_index in range(int(row["depth"])):
                output.append(
                    natural_segment_target(row, step_index, trace_kind=trace_kind)
                )
    if not output:
        raise NTA1DataError("natural arithmetic segment board is empty")
    return output


__all__ = [
    "NATURAL_SCALAR",
    "NTA1DataError",
    "build_nta1_segments",
    "natural_segment_target",
]
