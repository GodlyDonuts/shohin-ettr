#!/usr/bin/env python3
"""Apply the prospectively frozen ISET1 development gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from merge_dset1_evaluation_shards import MERGED_SCHEMA


SCHEMA = "shohin-iset1-stage0-comparison-v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load(path: Path, arm: str) -> dict:
    report = json.loads(path.read_text())
    if (
        report.get("schema") != MERGED_SCHEMA
        or report.get("status") != "complete"
        or report.get("arm") != arm
        or report.get("holdout_used") is not False
    ):
        raise RuntimeError(f"ISET1 {arm} report differs")
    return report


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise RuntimeError("ISET1 comparison exists")
    arms = {
        "aligned": load(args.aligned, "aligned"),
        "swapped": load(args.swapped, "swapped"),
        "hidden": load(args.hidden, "hidden"),
    }
    aligned = arms["aligned"]
    for key in ("row_count", "pair_count", "data_sha256", "data_report_sha256"):
        if any(report.get(key) != aligned.get(key) for report in arms.values()):
            raise RuntimeError("ISET1 matched geometry differs")
    rows = int(aligned["row_count"])
    correct = int(aligned["execution_correct"])
    family = aligned["family_metrics"]
    clean = aligned["member_metrics"]["clean"]
    fault = aligned["member_metrics"]["fault"]
    gates = {
        "execution_accuracy_ge_0_98": correct / rows >= 0.98,
        "each_family_script_accuracy_ge_0_95": min(float(value["script_exact_accuracy"]) for value in family.values()) >= 0.95,
        "clean_identity_edit_ge_0_99": float(clean["execution_correct_accuracy"]) >= 0.99,
        "fault_repair_ge_0_97": float(fault["execution_correct_accuracy"]) >= 0.97,
        "pair_consistency_ge_0_95": float(aligned["counterfactual_consistency"]) >= 0.95,
        "aligned_beats_swapped_by_0_20": (correct - int(arms["swapped"]["execution_correct"])) / rows >= 0.20,
        "aligned_beats_hidden_by_0_20": (correct - int(arms["hidden"]["execution_correct"])) / rows >= 0.20,
        "zero_execution_errors": not aligned["execution_errors"],
        "zero_exhaustion": int(aligned["max_token_exhausted"]) == 0,
    }
    passed = all(gates.values())
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "thresholds_frozen_before_output": True,
        "holdout_used": False,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in (("aligned", args.aligned), ("swapped", args.swapped), ("hidden", args.hidden))
        },
        "row_count": rows,
        "metrics": {
            name: {
                "script_exact": report["script_exact"],
                "execution_correct": report["execution_correct"],
                "counterfactual_consistency": report["counterfactual_consistency"],
                "family_metrics": report["family_metrics"],
                "member_metrics": report["member_metrics"],
                "execution_errors": report["execution_errors"],
            }
            for name, report in arms.items()
        },
        "margins": {
            "aligned_minus_swapped": correct - int(arms["swapped"]["execution_correct"]),
            "aligned_minus_hidden": correct - int(arms["hidden"]["execution_correct"]),
        },
        "gates": gates,
        "passed": passed,
        "holdout_authorized": passed,
        "decision": "open_iset1_holdout" if passed else "close_exact_iset1_v0",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--swapped", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"gates": report["gates"], "passed": report["passed"]}, sort_keys=True))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
