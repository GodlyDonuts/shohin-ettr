#!/usr/bin/env python3
"""Fail-closed reducer for the frozen DIVERGE-TOL1 OOD gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


class TOL1GateError(RuntimeError):
    """The TOL1 evaluation cannot be reduced under the frozen contract."""


def score(report: dict[str, object]) -> dict[str, object]:
    if report.get("schema") != "shohin-diverge-tol1-evaluation-v1":
        raise TOL1GateError("TOL1 evaluation schema differs")
    rows = int(report["rows"])
    clauses = int(report["clauses"])
    if rows != 1_024 or clauses <= rows:
        raise TOL1GateError("TOL1 OOD board dimensions differ")
    counts = report["counts"]
    treatment = int(counts["treatment_answer"]) / rows
    raw = int(counts["raw_answer"]) / rows
    operation_shift = int(counts["operation_shift_answer"]) / rows
    binding = int(counts["binding_derangement_answer"]) / rows
    state_reset = int(counts["state_reset_answer"]) / rows
    query_only = int(counts["query_only_answer"]) / rows
    rates = {
        "operation_exact": int(report["operation_exact"]) / clauses,
        "structured_instruction_exact": int(report["structured_instruction_exact"]) / clauses,
        "program_exact": int(counts["program_exact"]) / rows,
        "treatment_answer": treatment,
        "raw_answer": raw,
        "operation_shift_answer": operation_shift,
        "binding_derangement_answer": binding,
        "state_reset_answer": state_reset,
        "query_only_answer": query_only,
    }
    feature_counts = report["feature_counts"]
    feature_correct = report["feature_correct"]
    feature_rates = {
        name: int(feature_correct[name]) / int(feature_counts[name])
        for name in ("guard", "swap", "register_operand", "rational")
    }
    checks = {
        "operation_at_least_97_percent": rates["operation_exact"] >= 0.97,
        "instruction_at_least_95_percent": rates["structured_instruction_exact"] >= 0.95,
        "program_at_least_80_percent": rates["program_exact"] >= 0.80,
        "answer_at_least_85_percent": treatment >= 0.85,
        "each_feature_at_least_80_percent": min(feature_rates.values()) >= 0.80,
        "structured_beats_raw_by_20_points": treatment - raw >= 0.20,
        "operation_shift_loses_50_points": treatment - operation_shift >= 0.50,
        "binding_derangement_loses_50_points": treatment - binding >= 0.50,
        "state_reset_loses_50_points": treatment - state_reset >= 0.50,
        "query_only_below_20_percent": query_only < 0.20,
        "zero_malformed_packets_accepted": int(report.get("malformed_packets_accepted", 0)) == 0,
    }
    return {
        "schema": "shohin-diverge-tol1-gate-v1",
        "pass": all(checks.values()),
        "checks": checks,
        "rates": rates,
        "feature_rates": feature_rates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing TOL1 gate: {args.output}")
    report = json.loads(args.evaluation.read_text())
    result = score(report)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
