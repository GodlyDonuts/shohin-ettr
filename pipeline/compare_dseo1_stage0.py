#!/usr/bin/env python3
"""Apply the frozen conjunctive DSEO1 Stage-0 action/repair gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class DSEO1CompareError(RuntimeError):
    """The DSEO1 Stage-0 comparison inputs differ."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load(path: Path, arm: str) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if (
        report.get("schema") != "shohin-dseo1-paired-evaluation-merged-v1"
        or report.get("status") != "complete"
        or report.get("arm") != arm
        or int(report.get("pair_count", 0)) != 1024
        or int(report.get("row_count", 0)) != 2048
    ):
        raise DSEO1CompareError(f"DSEO1 merged report differs: {arm}")
    return report


def compare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise DSEO1CompareError("DSEO1 comparison output exists")
    aligned = load(args.aligned, "aligned")
    swapped = load(args.swapped, "swapped")
    hidden = load(args.hidden, "hidden")
    final_only = load(args.final_only, "final_only")
    owner = load(args.owner, "owner")
    fixed = ("data_sha256", "data_report_sha256", "pair_count", "row_count")
    if any(report.get(key) != aligned.get(key) for report in (swapped, hidden, final_only, owner) for key in fixed):
        raise DSEO1CompareError("DSEO1 comparison population differs")
    per_family = all(
        float(metrics["action_accuracy"]) >= 0.95
        for metrics in aligned["family_metrics"].values()
    )
    aligned_clean = int(aligned["member_metrics"]["clean"]["answer_correct"])
    owner_clean = int(owner["member_metrics"]["clean"]["answer_correct"])
    aligned_fault_accuracy = float(aligned["member_metrics"]["fault"]["answer_accuracy"])
    gates = {
        "aligned_action_accuracy_ge_0_95": float(aligned["action_accuracy"]) >= 0.95,
        "aligned_each_family_action_accuracy_ge_0_95": per_family,
        "counterfactual_consistency_ge_0_90": float(aligned["counterfactual_consistency"]) >= 0.90,
        "swapped_action_accuracy_le_0_60": float(swapped["action_accuracy"]) <= 0.60,
        "hidden_action_accuracy_le_0_60": float(hidden["action_accuracy"]) <= 0.60,
        "clean_answer_nonregression": aligned_clean >= owner_clean,
        "fault_repair_accuracy_ge_0_90": aligned_fault_accuracy >= 0.90,
    }
    passed = all(gates.values())
    report = {
        "schema": "shohin-dseo1-stage0-comparison-v1",
        "status": "complete",
        "passed": passed,
        "holdout_used": False,
        "thresholds_frozen_before_output": True,
        "gates": gates,
        "metrics": {
            arm: {
                "action_accuracy": source["action_accuracy"],
                "answer_accuracy": source["answer_accuracy"],
                "counterfactual_consistency": source["counterfactual_consistency"],
                "member_metrics": source["member_metrics"],
                "family_metrics": source["family_metrics"],
            }
            for arm, source in (
                ("aligned", aligned),
                ("swapped", swapped),
                ("hidden", hidden),
                ("final_only", final_only),
                ("owner", owner),
            )
        },
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in (
                ("aligned", args.aligned),
                ("swapped", args.swapped),
                ("hidden", args.hidden),
                ("final_only", args.final_only),
                ("owner", args.owner),
            )
        },
        "action_intervention_pending": True,
        "capability_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--swapped", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--final-only", type=Path, required=True)
    parser.add_argument("--owner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args)
    print(json.dumps({"passed_without_intervention": report["passed"], "gates": report["gates"]}, sort_keys=True))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
