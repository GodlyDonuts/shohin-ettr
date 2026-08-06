#!/usr/bin/env python3
"""Score the frozen NTA2 constrained natural-transfer gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def score(report: dict[str, object]) -> dict[str, object]:
    if report.get("schema") != "shohin-diverge-nta2-evaluation-v1" or report.get("rows") != 279:
        raise ValueError("NTA2 evaluation contract differs")
    if report.get("updates_after_fta1") != 0:
        raise ValueError("NTA2 is a zero-update gate")
    compiler = report["compiler"]
    arms = report["arms"]
    normal = arms["normal"]["counts"]
    conditions = {
        "operation_exact_at_least_95_percent": compiler["rates"]["operation_exact"] >= 0.95,
        "projected_roles_at_least_95_percent": compiler["rates"]["projected_role_exact"] >= 0.95,
        "valid_packets_at_least_95_percent": compiler["rates"]["valid"] >= 0.95,
        "selection_exact_at_least_250": normal["selection_exact"] >= 250,
        "terminal_exact_at_least_250": normal["terminal_exact"] >= 250,
        "trajectory_exact_at_least_250": normal["trajectory_exact"] >= 250,
        "zero_invalid": normal["invalid"] == 0,
        "trust_source_drop_at_least_200": normal["terminal_exact"] - arms["trust_source"]["counts"]["terminal_exact"] >= 200,
        "ignore_conflict_drop_at_least_200": normal["terminal_exact"] - arms["ignore_first_conflict"]["counts"]["terminal_exact"] >= 200,
        "initial_swap_drop_at_least_150": normal["terminal_exact"] - arms["initial_swap"]["counts"]["terminal_exact"] >= 150,
        "operation_shift_drop_at_least_150": normal["terminal_exact"] - arms["operation_shift"]["counts"]["terminal_exact"] >= 150,
    }
    for operation, arm in report["normal_per_error_operation"].items():
        counts = arm["counts"]
        conditions[f"{operation}_terminal_at_least_80_percent"] = counts["terminal_exact"] / counts["rows"] >= 0.80
    for depth, arm in report["normal_per_depth"].items():
        counts = arm["counts"]
        conditions[f"depth_{depth}_terminal_at_least_80_percent"] = counts["terminal_exact"] / counts["rows"] >= 0.80
    passed = all(conditions.values())
    return {
        "schema": "shohin-diverge-nta2-gate-v1",
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
            "promote_finite_state_projection_to_context_rich_natural_trace_gate"
            if passed
            else "close_zero_update_natural_projection_before_supervision"
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
