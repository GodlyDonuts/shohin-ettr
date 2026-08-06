#!/usr/bin/env python3
"""Fail-closed reducer for the frozen FTA1 autonomous composition gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


class FTA1AutonomousGateError(RuntimeError):
    """The autonomous FTA1 report cannot be scored."""


def score(report: dict[str, object]) -> dict[str, object]:
    if report.get("schema") != "shohin-diverge-fta1-autonomous-evaluation-v1":
        raise FTA1AutonomousGateError("autonomous FTA1 schema differs")
    if int(report.get("rows", -1)) != 480:
        raise FTA1AutonomousGateError("autonomous gate requires 480 rows")
    compiler = report["compiler"]
    arms = report["arms"]
    normal = arms["normal"]
    conditions = {
        "compiler_role_at_least_99_percent": compiler["rates"]["role_exact"] >= 0.99,
        "compiler_operation_at_least_99_percent": compiler["rates"]["operation_exact"] >= 0.99,
        "compiler_lhs_at_least_99_percent": compiler["rates"]["lhs_exact"] >= 0.99,
        "compiler_rhs_at_least_99_percent": compiler["rates"]["rhs_exact"] >= 0.99,
        "compiler_arguments_at_least_99_percent": compiler["rates"]["argument_exact"] >= 0.99,
        "selection_exact_at_least_432": normal["counts"]["selection_exact"] >= 432,
        "terminal_exact_at_least_432": normal["counts"]["terminal_exact"] >= 432,
        "trajectory_exact_at_least_432": normal["counts"]["trajectory_exact"] >= 432,
        "zero_invalid": normal["counts"]["invalid"] == 0,
        "trust_source_drop_at_least_384": normal["counts"]["terminal_exact"] - arms["trust_source"]["counts"]["terminal_exact"] >= 384,
        "ignore_conflict_drop_at_least_384": normal["counts"]["terminal_exact"] - arms["ignore_first_conflict"]["counts"]["terminal_exact"] >= 384,
        "initial_swap_drop_at_least_240": normal["counts"]["terminal_exact"] - arms["initial_swap"]["counts"]["terminal_exact"] >= 240,
        "operation_shift_drop_at_least_240": normal["counts"]["terminal_exact"] - arms["operation_shift"]["counts"]["terminal_exact"] >= 240,
    }
    for family in ("scalar", "register", "symbolic"):
        counts = normal["per_family"][family]["counts"]
        conditions[f"{family}_selection_at_least_136"] = counts["selection_exact"] >= 136
        conditions[f"{family}_terminal_at_least_136"] = counts["terminal_exact"] >= 136
        conditions[f"{family}_trajectory_at_least_136"] = counts["trajectory_exact"] >= 136
    passed = all(conditions.values())
    return {
        "schema": "shohin-diverge-fta1-autonomous-gate-v1",
        "status": "pass" if passed else "fail",
        "conditions": conditions,
        "normal": normal["counts"],
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
            "promote_typed_contradiction_replay_to_one_natural_trace_transfer_gate"
            if passed
            else "close_fta1_autonomous_composition_without_nearby_variants"
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
