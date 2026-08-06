#!/usr/bin/env python3
"""Tests for the NTA2 constrained transfer reducer."""

from score_diverge_nta2 import score


def main() -> None:
    normal = {"rows": 279, "selection_exact": 260, "terminal_exact": 260, "trajectory_exact": 260, "invalid": 0}
    zero = {**normal, "selection_exact": 0, "terminal_exact": 0, "trajectory_exact": 0}
    slices = {
        name: {"counts": {"rows": 10, "terminal_exact": 9}}
        for name in ("add", "subtract", "multiply")
    }
    depths = {
        str(depth): {"counts": {"rows": 10, "terminal_exact": 9}}
        for depth in (2, 3, 4, 5)
    }
    report = {
        "schema": "shohin-diverge-nta2-evaluation-v1",
        "rows": 279,
        "updates_after_fta1": 0,
        "compiler": {"rates": {"operation_exact": 1.0, "projected_role_exact": 1.0, "valid": 1.0}, "counts": {}},
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
    report["arms"]["normal"]["counts"]["terminal_exact"] = 249
    assert score(report)["status"] == "fail"
    print("diverge NTA2 gate tests passed")


if __name__ == "__main__":
    main()
