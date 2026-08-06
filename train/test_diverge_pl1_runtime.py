#!/usr/bin/env python3
"""Tests for DIVERGE-PL1 policy-state mechanics."""

from __future__ import annotations

from diverge_pl1_data import build_episode
from diverge_pl1_runtime import (
    SIZE,
    evaluate_policy_state,
    matrix_hash,
    maximum_assignment,
    poison_and_rollback_probe,
    run_episode,
    zero_matrix,
)


def main() -> None:
    matrix = zero_matrix()
    assert maximum_assignment(matrix) == tuple(range(SIZE))
    for index in range(SIZE):
        matrix[index][SIZE - index - 1] = 1.0
    assert maximum_assignment(matrix) == tuple(reversed(range(SIZE)))

    episode = build_episode(split="oracle_development", seed=2026080702, serial=7)
    static = run_episode(episode, arm="STATIC", seed=11)
    pl1 = run_episode(episode, arm="PL1", seed=11)
    reset = run_episode(episode, arm="PL1", seed=11, reset_before_transfer=True)
    repeat = run_episode(episode, arm="PL1", seed=11)
    assert pl1 == repeat
    assert len(pl1.write_receipts) == len(episode.acquisition)
    assert len(pl1.probe_transfer_exact) == len(episode.acquisition)
    assert pl1.policy_hash != matrix_hash(zero_matrix())
    assert reset.policy_hash == matrix_hash(zero_matrix())
    assert static.policy_hash == matrix_hash(zero_matrix())
    assert evaluate_policy_state(episode, pl1.policy_state)
    rollback = poison_and_rollback_probe(episode, pl1)
    assert rollback.exact
    assert rollback.pre_hash != rollback.poisoned_hash
    assert pl1.transfer_total == len(episode.transfer)
    assert 0 <= pl1.transfer_exact <= pl1.transfer_total
    assert 0 <= reset.transfer_exact <= reset.transfer_total
    try:
        run_episode(episode, arm="PL1", seed=11, inject_protected_mutation=True)
    except RuntimeError as error:
        assert "protected owner changed" in str(error)
    else:
        raise AssertionError("protected mutation did not fail closed")
    print("DIVERGE-PL1 runtime tests passed")


if __name__ == "__main__":
    main()
