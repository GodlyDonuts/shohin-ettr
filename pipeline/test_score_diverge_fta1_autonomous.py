#!/usr/bin/env python3
"""Tests for the FTA1 autonomous gate reducer."""

from score_diverge_fta1_autonomous import score


def main() -> None:
    counts = {
        "rows": 480,
        "active_steps": 3854,
        "initial_exact": 480,
        "selection_exact": 480,
        "exact_steps": 3854,
        "trajectory_exact": 480,
        "terminal_exact": 480,
        "invalid": 0,
    }
    family = {
        name: {
            "counts": {
                **counts,
                "rows": 160,
                "selection_exact": 160,
                "trajectory_exact": 160,
                "terminal_exact": 160,
            }
        }
        for name in ("scalar", "register", "symbolic")
    }
    zero = {**counts, "selection_exact": 0, "trajectory_exact": 0, "terminal_exact": 0}
    report = {
        "schema": "shohin-diverge-fta1-autonomous-evaluation-v1",
        "rows": 480,
        "compiler": {
            "rates": {
                "role_exact": 1.0,
                "operation_exact": 1.0,
                "lhs_exact": 1.0,
                "rhs_exact": 1.0,
                "argument_exact": 1.0,
            }
        },
        "arms": {
            "normal": {"counts": counts, "per_family": family},
            "trust_source": {"counts": zero},
            "ignore_first_conflict": {"counts": zero},
            "initial_swap": {"counts": zero},
            "operation_shift": {"counts": zero},
        },
    }
    assert score(report)["status"] == "pass"
    report["arms"]["normal"]["counts"]["selection_exact"] = 431
    assert score(report)["status"] == "fail"
    print("diverge FTA1 autonomous gate tests passed")


if __name__ == "__main__":
    main()
