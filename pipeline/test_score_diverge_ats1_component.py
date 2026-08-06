#!/usr/bin/env python3
"""Focused checks for the ATS1 component reducer."""

from score_diverge_ats1_component import score


def _report(terminal: int = 440) -> dict[str, object]:
    family = {
        "counts": {"terminal_exact": 145, "trajectory_exact": 140},
        "rates": {},
    }
    normal = {
        "counts": {"terminal_exact": terminal, "invalid": 0},
        "per_family": {name: family for name in ("scalar", "register", "symbolic")},
    }
    return {
        "schema": "shohin-diverge-ats1-forced-evaluation-v1",
        "rows": 480,
        "compiler": {"rates": {"operation_exact": 0.995, "lhs_exact": 0.98, "argument_exact": 0.995}},
        "replay": {
            "normal": normal,
            "initial_swap": {"counts": {"terminal_exact": 100}},
            "operation_shift": {"counts": {"terminal_exact": 100}},
        },
        "rhs_poison_invariant": True,
    }


def test_pass() -> None:
    assert score(_report())["status"] == "pass"


def test_fail() -> None:
    assert score(_report(terminal=200))["status"] == "fail"


def main() -> None:
    test_pass()
    test_fail()
    print("diverge ATS1 gate tests passed")


if __name__ == "__main__":
    main()
