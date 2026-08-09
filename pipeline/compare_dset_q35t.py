#!/usr/bin/env python3
"""Apply the frozen two-arm trained Qwen DSET transfer gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from merge_dset1_evaluation_shards import MERGED_SCHEMA


SCHEMA = "shohin-dset-q35-trained-transfer-comparison-v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load(path: Path, arm: str) -> dict:
    report = json.loads(path.read_text())
    if (
        report.get("schema") != MERGED_SCHEMA
        or report.get("status") != "complete"
        or report.get("arm") != arm
        or report.get("holdout_used") is not False
    ):
        raise RuntimeError(f"DSET-Q35T {arm} report differs")
    return report


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise RuntimeError("DSET-Q35T comparison exists")
    aligned = load(args.aligned, "aligned")
    hidden = load(args.hidden, "hidden")
    for key in ("row_count", "pair_count", "data_sha256", "data_report_sha256"):
        if aligned.get(key) != hidden.get(key):
            raise RuntimeError("DSET-Q35T matched geometry differs")
    rows = int(aligned["row_count"])
    aligned_correct = int(aligned["execution_correct"])
    family = aligned["family_metrics"]
    member = aligned["member_metrics"]
    gates = {
        "execution_accuracy_ge_0_95": aligned_correct / rows >= 0.95,
        "numeric_script_accuracy_ge_0_90": float(family["numeric_final"]["script_exact_accuracy"]) >= 0.90,
        "choice_script_accuracy_ge_0_90": float(family["choice_final"]["script_exact_accuracy"]) >= 0.90,
        "clean_copy_ge_0_99": float(member["clean"]["execution_correct_accuracy"]) >= 0.99,
        "fault_repair_ge_0_90": float(member["fault"]["execution_correct_accuracy"]) >= 0.90,
        "pair_consistency_ge_0_90": float(aligned["counterfactual_consistency"]) >= 0.90,
        "aligned_beats_hidden_by_13": aligned_correct - int(hidden["execution_correct"]) >= 13,
        "zero_execution_errors": not aligned.get("execution_errors"),
        "zero_exhaustion": int(aligned.get("max_token_exhausted", -1)) == 0,
    }
    passed = all(gates.values())
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "thresholds_frozen_before_output": True,
        "holdout_used": False,
        "inputs": {
            "aligned": {"path": str(args.aligned.resolve()), "sha256": sha256_file(args.aligned)},
            "hidden": {"path": str(args.hidden.resolve()), "sha256": sha256_file(args.hidden)},
        },
        "row_count": rows,
        "aligned_execution_correct": aligned_correct,
        "hidden_execution_correct": int(hidden["execution_correct"]),
        "aligned_minus_hidden": aligned_correct - int(hidden["execution_correct"]),
        "aligned_family_metrics": family,
        "aligned_member_metrics": member,
        "aligned_counterfactual_consistency": aligned["counterfactual_consistency"],
        "aligned_execution_errors": aligned["execution_errors"],
        "gates": gates,
        "passed": passed,
        "confirmation_authorized": passed,
        "decision": "freeze_dset_q35t_confirmation" if passed else "close_exact_dset_q35t",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"gates": report["gates"], "passed": report["passed"]}, sort_keys=True))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
