#!/usr/bin/env python3
"""Apply the frozen PSET1 Stage-0 conjunctive gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from merge_pset1_evaluation_shards import MERGED_SCHEMA


SCHEMA = "shohin-pset1-stage0-comparison-v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load(path: Path, arm: str, intervention: str) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if report.get("schema") != MERGED_SCHEMA or report.get("status") != "complete" or report.get("arm") != arm or report.get("intervention") != intervention:
        raise RuntimeError("PSET1 comparison input differs")
    return report


def ratio(report: dict[str, Any], key: str) -> float:
    return int(report[key]) / int(report["row_count"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise RuntimeError("PSET1 comparison output exists")
    reports = {
        "aligned": load(args.aligned, "aligned", "normal"),
        "hidden": load(args.hidden, "aligned", "hidden"),
        "shuffled": load(args.shuffled, "aligned", "shuffled"),
        "permuted": load(args.permuted, "permuted", "normal"),
    }
    aligned = reports["aligned"]
    fixed = ("pair_count", "row_count", "data_sha256", "data_report_sha256", "host_checkpoint_sha256")
    if any(any(report.get(key) != aligned.get(key) for key in fixed) for report in reports.values()):
        raise RuntimeError("PSET1 matched geometry differs")
    family_floor = min(
        counts["program_exact"] / counts["rows"]
        for counts in aligned["family_counts"].values()
    )
    clean = aligned["member_counts"]["clean"]
    fault = aligned["member_counts"]["fault"]
    margins = {
        name: int(aligned["execution_correct"]) - int(report["execution_correct"])
        for name, report in reports.items() if name != "aligned"
    }
    gates = {
        "aligned_program_accuracy_ge_0_95": ratio(aligned, "program_exact") >= 0.95,
        "aligned_each_family_program_accuracy_ge_0_95": family_floor >= 0.95,
        "aligned_execution_accuracy_ge_0_95": ratio(aligned, "execution_correct") >= 0.95,
        "aligned_clean_copy_ge_0_99": clean["execution_correct"] / clean["rows"] >= 0.99,
        "aligned_fault_repair_ge_0_90": fault["execution_correct"] / fault["rows"] >= 0.90,
        "aligned_consistency_ge_0_95": aligned["counterfactual_consistent_pairs"] / aligned["pair_count"] >= 0.95,
        "aligned_beats_permuted_by_13": margins["permuted"] >= 13,
        "aligned_beats_hidden_by_13": margins["hidden"] >= 13,
        "aligned_beats_shuffled_by_13": margins["shuffled"] >= 13,
        "permuted_program_accuracy_le_0_60": ratio(reports["permuted"], "program_exact") <= 0.60,
        "hidden_program_accuracy_le_0_60": ratio(reports["hidden"], "program_exact") <= 0.60,
        "forced_keep_intervention_ge_0_95": aligned["force_keep_breaks_correct_repair"] / fault["rows"] >= 0.95,
        "zero_execution_errors": not aligned["execution_errors"],
        "zero_generation_exhaustion": int(aligned["replacement_finished"]) == int(aligned["row_count"]),
    }
    passed = all(gates.values())
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "thresholds_frozen_before_output": True,
        "holdout_used": False,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in (("aligned", args.aligned), ("hidden", args.hidden), ("shuffled", args.shuffled), ("permuted", args.permuted))
        },
        "metrics": reports,
        "margins": margins,
        "gates": gates,
        "passed": passed,
        "decision": "open_pset1_natural_draft_gate" if passed else "close_exact_pset1_v0",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--shuffled", type=Path, required=True)
    parser.add_argument("--permuted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"gates": report["gates"], "margins": report["margins"], "passed": report["passed"]}, sort_keys=True))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
