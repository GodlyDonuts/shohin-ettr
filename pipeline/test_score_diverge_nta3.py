#!/usr/bin/env python3
"""Tests for the NTA3 full-document reducer."""

from score_diverge_nta3 import score


def main() -> None:
    normal = {"selection_exact": 279, "terminal_exact": 279, "trajectory_exact": 279, "invalid": 0}
    zero = {**normal, "selection_exact": 0, "terminal_exact": 0, "trajectory_exact": 0}
    report = {
        "schema": "shohin-diverge-nta3-evaluation-v1",
        "rows": 279,
        "updates_after_fta1": 0,
        "scanner": {"exact_rows": 279, "target_rows": 279, "exact_transactions": 963, "target_transactions": 963},
        "compiler": {"rates": {"operation_exact": 1.0, "valid": 1.0}, "counts": {}},
        "arms": {
            "normal": {"counts": normal},
            "trust_source": {"counts": zero},
            "ignore_first_conflict": {"counts": zero},
            "initial_swap": {"counts": zero},
            "operation_shift": {"counts": zero},
        },
    }
    assert score(report)["status"] == "pass"
    report["scanner"]["exact_rows"] = 278
    assert score(report)["status"] == "fail"
    print("diverge NTA3 gate tests passed")


if __name__ == "__main__":
    main()
