#!/usr/bin/env python3
"""Renderer identifiability and oracle-mechanics tests for DIVERGE-EAL2."""

from __future__ import annotations

from diverge_eal1_data import (
    DEMONSTRATIONS_PER_OPERATION,
    DEVELOPMENT_MATRICES,
    OPERATIONS,
    TRAIN_MATRICES,
    TRANSFER_DEPTHS,
    apply_matrix,
)
from diverge_eal2_data import (
    DEVELOPMENT_EPISODES,
    build_development_episode,
    build_training_record,
    overlap_report,
)
from diverge_mze1_runtime import PRIME, ROW_CANDIDATES


def _compatible_rows(
    demonstrations: list[tuple[tuple[int, ...], tuple[int, ...]]], output: int
):
    return tuple(
        row
        for row in ROW_CANDIDATES
        if all(
            (row[0] * before[0] + row[1] * before[1]) % PRIME == after[output]
            for before, after in demonstrations
        )
    )


def main() -> None:
    training = [build_training_record(index) for index in range(128)]
    paired = [build_development_episode(index) for index in range(DEVELOPMENT_EPISODES)]
    public = [value[0] for value in paired]
    assessor = [value[1] for value in paired]
    train_pairs = {tuple(row["renderer"][:2]) for row in training}
    development_pairs = {
        tuple(item["renderer"][:2])
        for episode in public
        for item in episode["evidence"]
    }
    assert train_pairs.isdisjoint(development_pairs)
    for pairs in (train_pairs, development_pairs):
        assert {value[0] for value in pairs} == set(range(4))
        assert {value[1] for value in pairs} == set(range(4))
    assert set(TRAIN_MATRICES).isdisjoint(DEVELOPMENT_MATRICES)
    audit = overlap_report(training, public)
    assert audit["source_overlap"] == 0
    assert audit["name_overlap"] == 0
    assert audit["matrix_overlap"] == 0
    assert audit["development_source_unique"]
    assert build_training_record(0) == training[0]
    assert build_development_episode(0) == paired[0]

    visible = public[0]
    hidden = assessor[0]
    matrices = tuple(
        tuple(tuple(int(coefficient) for coefficient in row) for row in matrix)
        for matrix in hidden["matrices"]
    )
    for operation in range(OPERATIONS):
        demonstrations = [
            (tuple(item["before"]), tuple(item["after"]))
            for item in hidden["evidence"]
            if int(item["operation_index"]) == operation
        ]
        assert len(demonstrations) == DEMONSTRATIONS_PER_OPERATION
        for output in range(2):
            assert len(_compatible_rows(demonstrations[:1], output)) == 5
            assert _compatible_rows(demonstrations, output) == (
                matrices[operation][output],
            )
    assert tuple(item["depth"] for item in visible["transfer"]) == TRANSFER_DEPTHS
    for program, target in zip(visible["transfer"], hidden["transfer"], strict=True):
        state = tuple(program["initial_state"])
        for operation in target["symbol_indices"]:
            state = apply_matrix(matrices[int(operation)], state)
        assert list(state) == target["terminal_state"]
    print("diverge EAL2 data tests passed")


if __name__ == "__main__":
    main()
