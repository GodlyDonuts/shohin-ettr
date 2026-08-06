#!/usr/bin/env python3
"""Score the frozen zero-shot natural arithmetic transfer gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def score(report: dict[str, object]) -> dict[str, object]:
    if report.get("schema") != "shohin-diverge-nta1-evaluation-v1" or report.get("rows") != 279:
        raise ValueError("NTA1 evaluation contract differs")
    compiler = report["compiler"]
    arms = report["arms"]
    normal = arms["normal"]["counts"]
    conditions = {
        "compiler_operation_at_least_90_percent": compiler["rates"]["operation_exact"] >= 0.90,
        "compiler_role_at_least_80_percent": compiler["rates"]["role_exact"] >= 0.80,
        "compiler_valid_at_least_80_percent": compiler["rates"]["valid"] >= 0.80,
        "selection_exact_at_least_200": normal["selection_exact"] >= 200,
        "terminal_exact_at_least_200": normal["terminal_exact"] >= 200,
        "trajectory_exact_at_least_200": normal["trajectory_exact"] >= 200,
        "invalid_at_most_28": normal["invalid"] <= 28,
        "trust_source_drop_at_least_150": normal["terminal_exact"] - arms["trust_source"]["counts"]["terminal_exact"] >= 150,
        "ignore_conflict_drop_at_least_150": normal["terminal_exact"] - arms["ignore_first_conflict"]["counts"]["terminal_exact"] >= 150,
        "initial_swap_drop_at_least_120": normal["terminal_exact"] - arms["initial_swap"]["counts"]["terminal_exact"] >= 120,
        "operation_shift_drop_at_least_120": normal["terminal_exact"] - arms["operation_shift"]["counts"]["terminal_exact"] >= 120,
    }
    for operation, arm in report["normal_per_error_operation"].items():
        counts = arm["counts"]
        conditions[f"{operation}_terminal_at_least_60_percent"] = counts["terminal_exact"] / counts["rows"] >= 0.60
    for depth, arm in report["normal_per_depth"].items():
        counts = arm["counts"]
        conditions[f"depth_{depth}_terminal_at_least_60_percent"] = counts["terminal_exact"] / counts["rows"] >= 0.60
    passed = all(conditions.values())
    return {
        "schema": "shohin-diverge-nta1-gate-v1",
        "status": "pass" if passed else "fail",
        "conditions": conditions,
        "compiler": compiler["counts"],
        "normal": normal,
        "controls": {
            name: arms[name]["counts"]
            for name in (
                "trust_source",
                "ignore_first_conflict",
                "initial_swap",
                "operation_shift",
            )
        },
        "decision": (
            "authorize_one_source_disjoint_supervised_natural_compiler_gate"
            if passed
            else "localize_zero_shot_natural_interface_failure_before_training"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing output: {args.output}")
    result = score(json.loads(args.evaluation.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
