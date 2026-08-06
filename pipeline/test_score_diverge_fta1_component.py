#!/usr/bin/env python3
"""Focused checks for the FTA1 gate reducer."""

from score_diverge_fta1_component import score


def _report(terminal: int = 440) -> dict[str, object]:
    family = {"counts": {"terminal_exact": 145, "trajectory_exact": 140}}
    return {
        "schema": "shohin-diverge-fta1-forced-evaluation-v1",
        "rows": 480,
        "compiler": {"rates": {"operation_exact": 0.995, "lhs_exact": 0.98, "argument_exact": 0.995}},
        "replay": {
            "normal": {"counts": {"terminal_exact": terminal, "invalid": 0}, "per_family": {name: family for name in ("scalar", "register", "symbolic")}},
            "initial_swap": {"counts": {"terminal_exact": 100}},
            "operation_shift": {"counts": {"terminal_exact": 100}},
        },
        "rhs_poison_invariant": True,
    }


def main() -> None:
    assert score(_report())["status"] == "pass"
    assert score(_report(200))["status"] == "fail"
    print("diverge FTA1 gate tests passed")


if __name__ == "__main__":
    main()
