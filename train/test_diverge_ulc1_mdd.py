#!/usr/bin/env python3
"""Exact tests for the ULC1 multi-valued decision-DAG runtime."""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import replace

from diverge_ulc1_mdd import (
    RuntimeChoice,
    execute_choice_path,
    execute_mdd,
    k_best_product_paths,
    query_mdd,
    support_contains,
)
from diverge_v0 import (
    ABSTAIN,
    ANSWER,
    Query,
    TypedCell,
    TypedState,
    TypedTransaction,
    named_commitment,
    read_query,
)


def _choice(record: int, value: int, mass: int, transactions, witness: int):
    return RuntimeChoice(
        record,
        value,
        mass,
        tuple(transactions),
        witness,
        f"r{record}-v{value}",
        named_commitment("ulc1-mdd-test-choice", f"{record}:{value}"),
        (("record", record), ("value", value)),
    )


def _board():
    state = TypedState(
        (
            TypedCell(0, 0, 2),
            TypedCell(1, 0, 7),
            TypedCell(2, 1, 0),
            TypedCell(3, 1, 0),
            TypedCell(4, 1, 0),
        )
    )
    rows = []
    for record in range(3):
        witness = 2 + record
        rows.append(
            (
                _choice(
                    record,
                    0,
                    5,
                    (TypedTransaction("SET_VALUE", (witness, 0)),),
                    0,
                ),
                _choice(
                    record,
                    1,
                    2,
                    (
                        TypedTransaction("ADD_VALUE", (0, record + 1)),
                        TypedTransaction("SWAP_VALUE", (0, 1)),
                        TypedTransaction("SET_VALUE", (witness, 1)),
                    ),
                    1,
                ),
                _choice(
                    record,
                    2,
                    1,
                    (
                        TypedTransaction("SWAP_VALUE", (0, 1)),
                        TypedTransaction("ADD_VALUE", (0, record + 1)),
                        TypedTransaction("SET_VALUE", (witness, 2)),
                    ),
                    2,
                ),
            )
        )
    return state, tuple(rows)


def test_mdd_matches_complete_enumeration_and_exact_masses() -> None:
    state, rows = _board()
    execution = execute_mdd(state, rows)
    assert not execution.overflow
    assert execution.represented_worlds == 27
    expected_states = defaultdict(int)
    for assignment in itertools.product(range(3), repeat=3):
        terminal = execute_choice_path(state, rows, assignment)
        assert terminal is not None
        mass = 1
        for record, value in enumerate(assignment):
            mass *= rows[record][value].mass
        expected_states[str(terminal.record())] += mass
        assert support_contains(execution, assignment)
    observed_states = {
        str(group.state.record()): execution.arena.total_mass(group.expression)
        for group in execution.groups
        if group.state is not None
    }
    assert observed_states == dict(expected_states)
    assert (
        sum(
            execution.arena.assignment_count(group.expression)
            for group in execution.groups
        )
        == 27
    )


def test_delayed_constraints_recover_one_coherent_lineage() -> None:
    state, rows = _board()
    execution = execute_mdd(state, rows)
    query = Query("READ_VALUE", (0,))
    assert query_mdd(execution, query).disposition == ABSTAIN
    allowed = {record: frozenset({2}) for record in range(3)}
    expected_state = execute_choice_path(state, rows, (2, 2, 2))
    assert expected_state is not None
    decision = query_mdd(execution, query, allowed=allowed)
    assert decision.disposition == ANSWER
    assert decision.answer == read_query(expected_state, query)
    assert decision.total_mass == 1


def test_merging_preserves_lineage_for_late_evidence() -> None:
    state, rows = _board()
    rows = list(rows)
    for record in (1, 2):
        witness = 2 + record
        rows[record] = tuple(
            replace(
                choice,
                transactions=(TypedTransaction("SET_VALUE", (witness, 0)),),
            )
            for choice in rows[record]
        )
    rows = tuple(rows)
    # Background choices for the final two records merge many prefixes into
    # identical states, but evidence must still distinguish their lineages.
    execution = execute_mdd(state, rows)
    assert len(execution.groups) < execution.represented_worlds
    left = query_mdd(
        execution,
        Query("READ_VALUE", (0,)),
        allowed={0: frozenset({0}), 1: frozenset({0}), 2: frozenset({0})},
    )
    right = query_mdd(
        execution,
        Query("READ_VALUE", (0,)),
        allowed={0: frozenset({2}), 1: frozenset({2}), 2: frozenset({2})},
    )
    assert left.disposition == right.disposition == ANSWER
    assert left.answer != right.answer


def test_k_best_product_is_exact_and_does_not_materialize_the_product() -> None:
    _, rows = _board()
    paths = k_best_product_paths(rows, 5)
    assert paths[0] == (0, 0, 0)
    masses = [
        rows[0][path[0]].mass * rows[1][path[1]].mass * rows[2][path[2]].mass
        for path in paths
    ]
    assert masses == sorted(masses, reverse=True)
    assert len(set(paths)) == 5


def test_caps_fail_closed_without_partial_groups() -> None:
    state, rows = _board()
    execution = execute_mdd(state, rows, max_nodes=2)
    assert execution.overflow
    assert not execution.groups and execution.represented_worlds == 0


def main() -> None:
    test_mdd_matches_complete_enumeration_and_exact_masses()
    test_delayed_constraints_recover_one_coherent_lineage()
    test_merging_preserves_lineage_for_late_evidence()
    test_k_best_product_is_exact_and_does_not_materialize_the_product()
    test_caps_fail_closed_without_partial_groups()
    print("DIVERGE-ULC1 MDD tests: passed")


if __name__ == "__main__":
    main()
