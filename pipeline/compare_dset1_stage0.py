#!/usr/bin/env python3
"""Apply the prospectively frozen DSET1 Stage-0 conjunctive gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from merge_dset1_evaluation_shards import MERGED_SCHEMA


SCHEMA = "shohin-dset1-stage0-comparison-v1"


class DSET1CompareError(RuntimeError):
    """DSET1 comparison inputs differ from the frozen gate."""


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


def accuracy(report: dict[str, Any], key: str) -> float:
    return int(report[key]) / int(report["row_count"])


def load(path: Path, arm: str) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if report.get("schema") != MERGED_SCHEMA or report.get("status") != "complete" or report.get("arm") != arm or report.get("holdout_used") is not False:
        raise DSET1CompareError(f"DSET1 {arm} input differs")
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise DSET1CompareError("DSET1 comparison output exists")
    arms = {name: load(path, name) for name, path in (("aligned", args.aligned), ("swapped", args.swapped), ("hidden", args.hidden))}
    aligned = arms["aligned"]
    fixed = ("row_count", "pair_count", "data_sha256", "data_report_sha256")
    if any(any(report.get(key) != aligned.get(key) for key in fixed) for report in arms.values()):
        raise DSET1CompareError("DSET1 matched-arm geometry differs")
    aligned_correct = int(aligned["execution_correct"])
    script_accuracy = accuracy(aligned, "script_exact")
    execution_accuracy = accuracy(aligned, "execution_correct")
    family_script = [float(value["script_exact_accuracy"]) for value in aligned["family_metrics"].values()]
    clean = aligned["member_metrics"]["clean"]
    fault = aligned["member_metrics"]["fault"]
    gates = {
        "aligned_script_accuracy_ge_0_90": script_accuracy >= 0.90,
        "aligned_each_family_script_accuracy_ge_0_90": min(family_script) >= 0.90,
        "aligned_consistency_ge_0_90": float(aligned["counterfactual_consistency"]) >= 0.90,
        "aligned_execution_accuracy_ge_0_95": execution_accuracy >= 0.95,
        "aligned_clean_copy_ge_0_99": float(clean["execution_correct_accuracy"]) >= 0.99,
        "aligned_fault_repair_ge_0_90": float(fault["execution_correct_accuracy"]) >= 0.90,
        "aligned_beats_swapped_by_13": aligned_correct - int(arms["swapped"]["execution_correct"]) >= 13,
        "aligned_beats_hidden_by_13": aligned_correct - int(arms["hidden"]["execution_correct"]) >= 13,
        "swapped_script_accuracy_le_0_60": accuracy(arms["swapped"], "script_exact") <= 0.60,
        "hidden_script_accuracy_le_0_60": accuracy(arms["hidden"], "script_exact") <= 0.60,
        "zero_execution_errors": not aligned.get("execution_errors"),
        "zero_exhaustion": int(aligned.get("max_token_exhausted", -1)) == 0,
    }
    passed = all(gates.values())
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "thresholds_frozen_before_output": True,
        "holdout_used": False,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in (("aligned", args.aligned), ("swapped", args.swapped), ("hidden", args.hidden))
        },
        "metrics": {
            name: {
                "script_accuracy": accuracy(value, "script_exact"),
                "execution_accuracy": accuracy(value, "execution_correct"),
                "counterfactual_consistency": value["counterfactual_consistency"],
                "execution_correct": value["execution_correct"],
                "family_metrics": value["family_metrics"],
                "member_metrics": value["member_metrics"],
                "execution_errors": value["execution_errors"],
            }
            for name, value in arms.items()
        },
        "margins": {
            "aligned_minus_swapped": aligned_correct - int(arms["swapped"]["execution_correct"]),
            "aligned_minus_hidden": aligned_correct - int(arms["hidden"]["execution_correct"]),
        },
        "gates": gates,
        "passed": passed,
        "capability_authorized": passed,
        "decision": "open_dset1_capability" if passed else "close_exact_dset1_v0",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--swapped", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps({"gates": report["gates"], "margins": report["margins"], "passed": report["passed"]}, sort_keys=True))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
