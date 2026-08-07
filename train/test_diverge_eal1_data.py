#!/usr/bin/env python3
"""Deterministic data and oracle-mechanics tests for DIVERGE-EAL1."""

from __future__ import annotations

import json

from diverge_eal1_data import (
    DEMONSTRATIONS_PER_OPERATION,
    DEVELOPMENT_EPISODES,
    DEVELOPMENT_MATRICES,
    OPERATIONS,
    TRAIN_MATRICES,
    TRANSFER_DEPTHS,
    TRANSFER_PROGRAMS,
    apply_matrix,
    build_development_episode,
    build_training_record,
    canonical_sha256,
    overlap_report,
    scan_integer_spans,
)
from diverge_mze1_runtime import PRIME, ROW_CANDIDATES


def _integers(text: str) -> tuple[int, ...]:
    return tuple(int(text[start:end]) for start, end in scan_integer_spans(text))


def _compatible_rows(
    demonstrations: list[tuple[tuple[int, int], tuple[int, int]]],
    output: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        row
        for row in ROW_CANDIDATES
        if all(
            (row[0] * before[0] + row[1] * before[1]) % PRIME == after[output]
            for before, after in demonstrations
        )
    )


def main() -> None:
    assert set(TRAIN_MATRICES).isdisjoint(DEVELOPMENT_MATRICES)
    assert len(TRANSFER_DEPTHS) == TRANSFER_PROGRAMS
    assert min(TRANSFER_DEPTHS) == 12 and max(TRANSFER_DEPTHS) == 32
    assert all(
        (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) % PRIME
        for matrix in (*TRAIN_MATRICES, *DEVELOPMENT_MATRICES)
    )

    training = [build_training_record(index) for index in range(128)]
    for row in training:
        assert _integers(row["source_text"]) == _integers(row["counterfactual_text"])
        assert _integers(row["source_text"]) == _integers(row["scrubbed_text"])
        assert tuple(row["counterfactual_role_ids"]) == tuple(
            role + 2 if role < 2 else role - 2 for role in row["numeric_role_ids"]
        )
        assert "Before " not in row["scrubbed_text"]
        assert "After " not in row["scrubbed_text"]
        assert "Prior to " not in row["scrubbed_text"]
        assert "Following " not in row["scrubbed_text"]

    public = []
    assessor = []
    for serial in range(DEVELOPMENT_EPISODES):
        visible, hidden = build_development_episode(serial)
        public.append(visible)
        assessor.append(hidden)
        if serial == 0:
            repeated = build_development_episode(serial)
            assert canonical_sha256(repeated) == canonical_sha256((visible, hidden))
        assert len(visible["evidence"]) == OPERATIONS * DEMONSTRATIONS_PER_OPERATION
        assert len(visible["transfer"]) == TRANSFER_PROGRAMS
        assert len(visible["queries"]) == 2 * TRANSFER_PROGRAMS
        serialized = json.dumps(visible, sort_keys=True)
        for forbidden in ("matrices", "terminal_state", "before", "after"):
            assert forbidden not in serialized

        matrices = tuple(
            tuple(tuple(int(value) for value in row) for row in matrix)
            for matrix in hidden["matrices"]
        )
        for operation in range(OPERATIONS):
            demonstrations = [
                (
                    tuple(int(value) for value in item["before"]),
                    tuple(int(value) for value in item["after"]),
                )
                for item in hidden["evidence"]
                if int(item["operation_index"]) == operation
            ]
            assert len(demonstrations) == DEMONSTRATIONS_PER_OPERATION
            for output in range(2):
                assert len(_compatible_rows(demonstrations[:1], output)) == 5
                assert _compatible_rows(demonstrations, output) == (
                    matrices[operation][output],
                )

        for visible_transfer, hidden_transfer in zip(
            visible["transfer"], hidden["transfer"], strict=True
        ):
            state = tuple(int(value) for value in visible_transfer["initial_state"])
            for operation in hidden_transfer["symbol_indices"]:
                state = apply_matrix(matrices[int(operation)], state)
            assert list(state) == hidden_transfer["terminal_state"]
            assert visible_transfer["symbols"] == [
                visible["aliases"][int(operation)]
                for operation in hidden_transfer["symbol_indices"]
            ]

    audit = overlap_report(training, public)
    assert audit["source_overlap"] == 0
    assert audit["name_overlap"] == 0
    assert audit["matrix_overlap"] == 0
    assert audit["training_source_unique"]
    assert audit["development_source_unique"]
    assert len({row["identity_sha256"] for row in public}) == DEVELOPMENT_EPISODES
    assert len({row["identity_sha256"] for row in assessor}) == DEVELOPMENT_EPISODES
    print("diverge EAL1 data tests passed")


if __name__ == "__main__":
    main()
