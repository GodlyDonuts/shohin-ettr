#!/usr/bin/env python3
"""Tests for contradiction-guided FTA1 autonomous replay."""

from diverge_ats1_runtime import CompiledSegment, typed_state_from_surfaces
from diverge_fta1_autonomous import evaluate_autonomous_replay


def _state(value: int):
    return typed_state_from_surfaces("scalar", str(value))


def _packet(lhs: int, rhs: int, operation: int, argument: int) -> CompiledSegment:
    return CompiledSegment(
        operation_id=operation,
        lhs=_state(lhs),
        rhs_claim=_state(rhs),
        arguments=(argument,),
        provenance_positions=(),
    )


def main() -> None:
    rows = [
        {
            "identity_sha256": "a" * 64,
            "family": "scalar",
            "depth": 2,
            "error_index": 1,
            "initial_state": 1,
            "program": [["add", 2], ["multiply", 3]],
            "answer": "9",
        },
        {
            "identity_sha256": "b" * 64,
            "family": "scalar",
            "depth": 2,
            "error_index": 1,
            "initial_state": 5,
            "program": [["add", 1], ["multiply", 2]],
            "answer": "12",
        },
    ]
    compiled = {
        ("a" * 64, 0, "wrong"): _packet(1, 4, 0, 2),
        ("a" * 64, 1, "wrong"): _packet(4, 12, 2, 3),
        ("b" * 64, 0, "wrong"): _packet(5, 8, 0, 1),
        ("b" * 64, 1, "wrong"): _packet(8, 16, 2, 2),
    }
    normal = evaluate_autonomous_replay(rows, compiled)
    assert normal["counts"]["selection_exact"] == 2
    assert normal["counts"]["terminal_exact"] == 2
    assert normal["counts"]["trajectory_exact"] == 2
    for ablation in (
        "trust_source",
        "ignore_first_conflict",
        "initial_swap",
        "operation_shift",
    ):
        result = evaluate_autonomous_replay(rows, compiled, ablation=ablation)
        assert result["counts"]["terminal_exact"] == 0
    print("diverge FTA1 autonomous tests passed")


if __name__ == "__main__":
    main()
