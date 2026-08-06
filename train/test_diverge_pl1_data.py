#!/usr/bin/env python3
"""Tests for the DIVERGE-PL1 episode generator."""

from __future__ import annotations

from diverge_pl1_data import (
    OP_NAMES,
    apply_operation,
    build_episode,
    episode_from_assessor_record,
    execute_mapping,
    iter_program_identities,
    operation_outputs_are_unique,
    verify_trace,
)


def main() -> None:
    episode = build_episode(split="development", seed=2026080702, serial=0)
    assert len(episode.aliases) == len(OP_NAMES)
    assert len(set(episode.aliases)) == len(OP_NAMES)
    assert sorted(episode.symbol_to_operation) == list(range(len(OP_NAMES)))
    assert len(episode.acquisition) == 12
    assert len(episode.transfer) == 16

    for program in (*episode.acquisition, *episode.transfer):
        assert len(program.trace) == len(program.symbols) + 1
        for state in program.trace[:-1]:
            assert operation_outputs_are_unique(state)
        candidate = execute_mapping(episode.symbol_to_operation, program)
        assert candidate == program.trace
        result = verify_trace(episode, program, candidate)
        assert result.passed
        assert result.first_error is None

        wrong_mapping = list(episode.symbol_to_operation)
        first_symbol = program.symbols[0]
        other_symbol = (first_symbol + 1) % len(OP_NAMES)
        wrong_mapping[first_symbol], wrong_mapping[other_symbol] = (
            wrong_mapping[other_symbol],
            wrong_mapping[first_symbol],
        )
        wrong_trace = execute_mapping(tuple(wrong_mapping), program)
        wrong = verify_trace(episode, program, wrong_trace)
        assert not wrong.passed
        assert wrong.first_error == 1

    repeat = build_episode(split="development", seed=2026080702, serial=0)
    assert repeat == episode
    assert episode_from_assessor_record(episode.assessor_record()) == episode
    other = build_episode(split="confirmation", seed=2026080711, serial=0)
    assert set(episode.aliases).isdisjoint(other.aliases)
    assert set(iter_program_identities((episode,))).isdisjoint(
        iter_program_identities((other,))
    )

    state = (11, 37)
    assert len({apply_operation(index, state) for index in range(len(OP_NAMES))}) == len(
        OP_NAMES
    )
    print("DIVERGE-PL1 data tests passed")


if __name__ == "__main__":
    main()
