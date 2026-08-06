#!/usr/bin/env python3
"""Frozen threshold checks for the DIVERGE-TOL1 reducer."""

from score_diverge_tol1_gate import score


def main() -> None:
    report = {
        "schema": "shohin-diverge-tol1-evaluation-v1",
        "rows": 1024,
        "clauses": 16888,
        "operation_exact": 16600,
        "structured_instruction_exact": 16200,
        "counts": {
            "program_exact": 900,
            "treatment_answer": 920,
            "raw_answer": 600,
            "operation_shift_answer": 100,
            "binding_derangement_answer": 120,
            "state_reset_answer": 0,
            "query_only_answer": 30,
        },
        "feature_counts": {name: 1024 for name in ("guard", "swap", "register_operand", "rational")},
        "feature_correct": {name: 920 for name in ("guard", "swap", "register_operand", "rational")},
        "malformed_packets_accepted": 0,
    }
    result = score(report)
    assert result["pass"]
    report["counts"]["treatment_answer"] = 700
    assert not score(report)["pass"]
    print("diverge TOL1 gate tests passed")


if __name__ == "__main__":
    main()
