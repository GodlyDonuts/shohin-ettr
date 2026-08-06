#!/usr/bin/env python3
"""Tests for the NTA1 zero-shot transfer reducer."""

from score_diverge_nta1 import score


def main() -> None:
    normal = {"rows": 279, "selection_exact": 220, "terminal_exact": 220, "trajectory_exact": 220, "invalid": 10}
    zero = {**normal, "selection_exact": 0, "terminal_exact": 0, "trajectory_exact": 0}
    slices = {
        name: {"counts": {"rows": 50, "terminal_exact": 35}}
        for name in ("add", "subtract", "multiply")
    }
    depths = {
        str(depth): {"counts": {"rows": 50, "terminal_exact": 35}}
        for depth in (2, 3, 4, 5)
    }
    report = {
        "schema": "shohin-diverge-nta1-evaluation-v1",
        "rows": 279,
        "compiler": {"rates": {"operation_exact": 0.95, "role_exact": 0.9, "valid": 0.9}, "counts": {}},
        "arms": {
            "normal": {"counts": normal},
            "trust_source": {"counts": zero},
            "ignore_first_conflict": {"counts": zero},
            "initial_swap": {"counts": zero},
            "operation_shift": {"counts": zero},
        },
        "normal_per_error_operation": slices,
        "normal_per_depth": depths,
    }
    assert score(report)["status"] == "pass"
    report["arms"]["normal"]["counts"]["terminal_exact"] = 199
    assert score(report)["status"] == "fail"
    print("diverge NTA1 gate tests passed")


if __name__ == "__main__":
    main()
