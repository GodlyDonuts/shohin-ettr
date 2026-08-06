#!/usr/bin/env python3
"""Deterministic supervisor boards for the bounded DIVERGE-MEI1 gate.

This file is assessor-side.  It may use exact DIVERGE mechanics to construct
labels, but no function from this module is imported by the candidate runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Sequence

from diverge_mei1_runtime import REGISTER_COUNT, VALUE_COUNT
from diverge_v0 import TypedCell, TypedState, apply_transaction
from diverge_v0_neural_pilot import PROGRAMS


EVIDENCE_COHORTS = (
    "train",
    "lexical_shift",
    "renderer_shift",
    "composition_shift",
)


@dataclass(frozen=True, slots=True)
class ProbeEvidence:
    evidence_id: str
    cohort: str
    words: tuple[str, ...]
    before: tuple[int, ...]
    after: tuple[int, ...]
    program: int | None
    renderer: int
    noise: int


def exact_program(
    values: Sequence[int],
    program: int | None,
) -> tuple[int, ...]:
    if len(values) != REGISTER_COUNT:
        raise ValueError("probe state has the wrong register count")
    state = TypedState(
        tuple(TypedCell(slot, 0, int(value)) for slot, value in enumerate(values))
    )
    transactions = () if program is None else PROGRAMS[int(program)]
    for transaction in transactions:
        state = apply_transaction(state, transaction)
    output = tuple(cell.value for cell in state.cells)
    if len(output) != REGISTER_COUNT or any(not 0 <= value < VALUE_COUNT for value in output):
        raise ValueError("supervisor program leaves the MEI1 value domain")
    return output


def _numbered(prefix: str, values: Sequence[int]) -> str:
    return " ".join(f"{prefix}{index} {value}" for index, value in enumerate(values))


def render_probe(
    before: Sequence[int],
    after: Sequence[int],
    *,
    cohort: str,
    renderer: int,
    noise: int,
) -> tuple[str, ...]:
    if cohort not in EVIDENCE_COHORTS:
        raise ValueError("unknown evidence cohort")
    if len(before) != REGISTER_COUNT or len(after) != REGISTER_COUNT:
        raise ValueError("evidence state has the wrong register count")
    b = tuple(int(value) for value in before)
    a = tuple(int(value) for value in after)
    if cohort == "train":
        templates = (
            "delayed audit before register zero {b0} register one {b1} register two {b2} register three {b3} register four {b4} after register zero {a0} register one {a1} register two {a2} register three {a3} register four {a4}",
            "probe began with slot 0 {b0} slot 1 {b1} slot 2 {b2} slot 3 {b3} slot 4 {b4} and ended with slot 0 {a0} slot 1 {a1} slot 2 {a2} slot 3 {a3} slot 4 {a4}",
            "input cells c0 {b0} c1 {b1} c2 {b2} c3 {b3} c4 {b4} output cells c0 {a0} c1 {a1} c2 {a2} c3 {a3} c4 {a4}",
            "witness start r0 {b0} r1 {b1} r2 {b2} r3 {b3} r4 {b4} witness finish r0 {a0} r1 {a1} r2 {a2} r3 {a3} r4 {a4}",
        )
    elif cohort == "lexical_shift":
        templates = (
            "inspection antecedent ledger alpha {b0} beta {b1} gamma {b2} delta {b3} epsilon {b4} consequent ledger alpha {a0} beta {a1} gamma {a2} delta {a3} epsilon {a4}",
            "diagnostic entry positions first {b0} second {b1} third {b2} fourth {b3} fifth {b4} exit positions first {a0} second {a1} third {a2} fourth {a3} fifth {a4}",
        )
    elif cohort == "renderer_shift":
        templates = (
            "after four {a4} three {a3} two {a2} one {a1} zero {a0} separator before four {b4} three {b3} two {b2} one {b1} zero {b0}",
            "observation table final 0 colon {a0} 1 colon {a1} 2 colon {a2} 3 colon {a3} 4 colon {a4} initial 0 colon {b0} 1 colon {b1} 2 colon {b2} 3 colon {b3} 4 colon {b4}",
        )
    else:
        templates = (
            "archive batch {noise} is irrelevant final readings second {a1} fourth {a3} first {a0} fifth {a4} third {a2} while initial readings third {b2} first {b0} fifth {b4} second {b1} fourth {b3} audit complete",
            "ignore checksum {noise} the terminal register vector has index 4 {a4} index 2 {a2} index 0 {a0} index 3 {a3} index 1 {a1} whereas the starting vector has index 1 {b1} index 3 {b3} index 0 {b0} index 4 {b4} index 2 {b2}",
        )
    template = templates[renderer % len(templates)]
    text = template.format(
        b0=b[0], b1=b[1], b2=b[2], b3=b[3], b4=b[4],
        a0=a[0], a1=a[1], a2=a[2], a3=a[3], a4=a[4],
        noise=noise,
    )
    return tuple(text.split())


def generate_probe_evidence(
    *,
    seed: int,
    cohort: str,
    program: int | None = None,
    sample_program: bool = True,
) -> ProbeEvidence:
    rng = random.Random(seed)
    if cohort not in EVIDENCE_COHORTS:
        raise ValueError("unknown evidence cohort")
    if sample_program and program is None and rng.randrange(5):
        program = rng.randrange(4)
    values = rng.sample(range(5, 91), REGISTER_COUNT)
    before = tuple(values)
    after = exact_program(before, program)
    renderer_counts = {
        "train": 4,
        "lexical_shift": 2,
        "renderer_shift": 2,
        "composition_shift": 2,
    }
    renderer = rng.randrange(renderer_counts[cohort])
    noise = rng.randrange(1000, 9999)
    words = render_probe(
        before,
        after,
        cohort=cohort,
        renderer=renderer,
        noise=noise,
    )
    return ProbeEvidence(
        f"mei1-probe-{cohort}-{seed}",
        cohort,
        words,
        before,
        after,
        program,
        renderer,
        noise,
    )


def exact_action_batch(
    states: Sequence[Sequence[int]],
    actions: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    if len(states) != len(actions):
        raise ValueError("supervisor action batch differs")
    action_to_program = {0: 4, 1: 5, 2: 2, 3: 3}
    # Program IDs 4 and 5 are virtual one-action programs handled below.
    output = []
    for values, action in zip(states, actions, strict=True):
        if action == 0:
            state = TypedState(
                tuple(TypedCell(slot, 0, int(value)) for slot, value in enumerate(values))
            )
            state = apply_transaction(state, PROGRAMS[0][0])
            output.append(tuple(cell.value for cell in state.cells))
        elif action == 1:
            state = TypedState(
                tuple(TypedCell(slot, 0, int(value)) for slot, value in enumerate(values))
            )
            state = apply_transaction(state, PROGRAMS[0][1])
            output.append(tuple(cell.value for cell in state.cells))
        else:
            output.append(exact_program(values, action_to_program[action]))
    return tuple(output)


def random_register_states(seed: int, count: int) -> tuple[tuple[int, ...], ...]:
    rng = random.Random(seed)
    return tuple(
        tuple(rng.randrange(5, 121) for _ in range(REGISTER_COUNT))
        for _ in range(count)
    )


def random_action_program(seed: int, depth: int) -> tuple[int, ...]:
    rng = random.Random(seed)
    return tuple(rng.randrange(4) for _ in range(depth))


def exact_action_program(
    values: Sequence[int],
    actions: Iterable[int],
) -> tuple[int, ...]:
    state = tuple(int(value) for value in values)
    for action in actions:
        state = exact_action_batch((state,), (action,))[0]
    return state
