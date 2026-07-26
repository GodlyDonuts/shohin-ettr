"""Independent exact auditor for the guarded resource-process ETTR board."""

from __future__ import annotations

from functools import lru_cache
from itertools import product

from cross_ontology_resource_board import (
    Demonstration,
    Marking,
    OPERATOR_LIBRARY,
    OPERATOR_SYMBOL_COUNT,
    PLACE_SPECS,
    ProcessOutcome,
    ProcessStatus,
    ResourceTheory,
    THEORIES,
    challenge_cases,
    execute_sequence,
    heldout_programs,
    input_markings,
    single_step_cases,
)


@lru_cache(maxsize=None)
def _independent_transition(
    operator_index: int,
    multiplicities: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Compile one operator application as bounded vector algebra."""

    operator = OPERATOR_LIBRARY[operator_index]
    guards = [0] * len(PLACE_SPECS)
    debits = [0] * len(PLACE_SPECS)
    credits = [0] * len(PLACE_SPECS)
    for quantity in operator.guards:
        guards[quantity.place] = quantity.multiplicity
    for quantity in operator.consumes:
        debits[quantity.place] = quantity.multiplicity
    for quantity in operator.produces:
        credits[quantity.place] = quantity.multiplicity
    if any(
        available < max(guard, debit)
        for available, guard, debit in zip(
            multiplicities,
            guards,
            debits,
            strict=True,
        )
    ):
        return None
    successor = tuple(
        available - debit + credit
        for available, debit, credit in zip(
            multiplicities,
            debits,
            credits,
            strict=True,
        )
    )
    if any(
        value < 0 or value > place.capacity
        for value, place in zip(successor, PLACE_SPECS, strict=True)
    ):
        return None
    return successor


@lru_cache(maxsize=1)
def independent_transition_table() -> dict[
    tuple[int, tuple[int, ...]],
    tuple[int, ...] | None,
]:
    """Materialize the finite transition relation before any sequence replay."""

    return {
        (operator_index, tuple(marking)): _independent_transition(
            operator_index,
            tuple(marking),
        )
        for operator_index in range(len(OPERATOR_LIBRARY))
        for marking in product(range(4), repeat=len(PLACE_SPECS))
    }


def independent_execute_sequence(
    theory: ResourceTheory,
    initial: Marking,
    sequence: tuple[int, ...],
) -> ProcessOutcome:
    """Replay only precompiled relation rows, unlike the board executor."""

    table = independent_transition_table()
    marking = initial.multiplicities
    for cursor, symbol in enumerate(sequence):
        if not 0 <= symbol < OPERATOR_SYMBOL_COUNT:
            raise ValueError("resource sequence differs")
        successor = table[(theory.operator_indices[symbol], marking)]
        if successor is None:
            return ProcessOutcome(
                Marking(marking),
                cursor,
                ProcessStatus.DEADLOCK,
            )
        marking = successor
    return ProcessOutcome(
        Marking(marking),
        len(sequence),
        ProcessStatus.HALT,
    )


def independent_consistent_theories(
    evidence: tuple[Demonstration, ...],
) -> tuple[int, ...]:
    return tuple(
        theory_index
        for theory_index, theory in enumerate(THEORIES)
        if all(
            independent_execute_sequence(
                theory,
                demo.initial,
                demo.sequence,
            )
            == demo.outcome
            for demo in evidence
        )
    )


def build_exact_audit() -> dict[str, int | bool]:
    """Exhaustively compare both executors on the frozen bounded board."""

    agreement = True
    halt_count = 0
    deadlock_count = 0
    for theory in THEORIES:
        for marking, sequence in challenge_cases():
            production = execute_sequence(theory, marking, sequence)
            reference = independent_execute_sequence(
                theory,
                marking,
                sequence,
            )
            agreement = agreement and production == reference
            if production.status == ProcessStatus.HALT:
                halt_count += 1
            else:
                deadlock_count += 1
    nominal_operator_slots = (
        len(THEORIES)
        * len(input_markings())
        * sum(len(sequence) for sequence in heldout_programs())
    )
    return {
        "theory_count": len(THEORIES),
        "operator_law_count": len(OPERATOR_LIBRARY),
        "input_marking_count": len(input_markings()),
        "single_step_evidence_case_count": len(single_step_cases()),
        "heldout_program_count": len(heldout_programs()),
        "challenge_case_count": len(challenge_cases()),
        "oracle_execution_comparisons": (len(THEORIES) * len(challenge_cases())),
        "nominal_operator_slots": nominal_operator_slots,
        "halt_outcome_count": halt_count,
        "deadlock_outcome_count": deadlock_count,
        "transition_table_rows": len(independent_transition_table()),
        "exact_oracle_agreement": agreement,
    }


__all__ = [
    "build_exact_audit",
    "independent_consistent_theories",
    "independent_execute_sequence",
    "independent_transition_table",
]
