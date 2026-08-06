#!/usr/bin/env python3
"""Fail-closed reducer for the frozen ATS1 forced-boundary component gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


class ATS1GateError(RuntimeError):
    """The ATS1 report cannot be scored under the frozen contract."""


def score(report: dict[str, object]) -> dict[str, object]:
    if report.get("schema") != "shohin-diverge-ats1-forced-evaluation-v1":
        raise ATS1GateError("ATS1 evaluation schema differs")
    if int(report.get("rows", -1)) != 480:
        raise ATS1GateError("ATS1 gate requires exactly 480 OOD rows")
    compiler = report["compiler"]
    replay = report["replay"]
    normal = replay["normal"]
    initial_swap = replay["initial_swap"]
    operation_shift = replay["operation_shift"]
    conditions = {
        "operation_exact_at_least_99_percent": compiler["rates"]["operation_exact"] >= 0.99,
        "lhs_state_exact_at_least_95_percent": compiler["rates"]["lhs_exact"] >= 0.95,
        "arguments_exact_at_least_99_percent": compiler["rates"]["argument_exact"] >= 0.99,
        "terminal_exact_at_least_432": normal["counts"]["terminal_exact"] >= 432,
        "zero_invalid_packets": normal["counts"]["invalid"] == 0,
        "rhs_poison_invariant": bool(report.get("rhs_poison_invariant")),
        "initial_swap_drop_at_least_240": (
            normal["counts"]["terminal_exact"] - initial_swap["counts"]["terminal_exact"] >= 240
        ),
        "operation_shift_drop_at_least_240": (
            normal["counts"]["terminal_exact"] - operation_shift["counts"]["terminal_exact"] >= 240
        ),
    }
    for family in ("scalar", "register", "symbolic"):
        values = normal["per_family"][family]["counts"]
        conditions[f"{family}_terminal_at_least_136"] = values["terminal_exact"] >= 136
        conditions[f"{family}_trajectory_at_least_128"] = values["trajectory_exact"] >= 128
    passed = all(conditions.values())
    return {
        "schema": "shohin-diverge-ats1-component-gate-v1",
        "status": "pass" if passed else "fail",
        "conditions": conditions,
        "terminal_exact": normal["counts"]["terminal_exact"],
        "initial_swap_terminal_exact": initial_swap["counts"]["terminal_exact"],
        "operation_shift_terminal_exact": operation_shift["counts"]["terminal_exact"],
        "decision": (
            "authorize_one_autonomous_crp1_composition_gate"
            if passed
            else "close_ats1_without_autonomous_composition_or_nearby_variants"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing output: {args.output}")
    report = json.loads(args.evaluation.read_text())
    result = score(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
