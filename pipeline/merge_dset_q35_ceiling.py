#!/usr/bin/env python3
"""Merge and gate the frozen DSET-Q35 capacity ceiling."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path


SHARD_SCHEMA = "shohin-dset-q35-ceiling-shard-v1"
SCHEMA = "shohin-dset-q35-ceiling-v1"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(paths: list[Path], output: Path) -> dict:
    if output.exists() or len(paths) != 8:
        raise RuntimeError("DSET-Q35 merge geometry differs")
    reports = [json.loads(path.read_text()) for path in paths]
    first = reports[0]
    fixed = ("shard_count", "model_config_sha256", "dset_data_sha256", "pset_data_sha256")
    if any(report.get("schema") != SHARD_SCHEMA or report.get("status") != "complete" for report in reports):
        raise RuntimeError("DSET-Q35 shard differs")
    if any(any(report.get(key) != first.get(key) for key in fixed) for report in reports):
        raise RuntimeError("DSET-Q35 shard binding differs")
    if {report["shard_index"] for report in reports} != set(range(8)):
        raise RuntimeError("DSET-Q35 shard coverage differs")
    results = [row for report in reports for row in report["results"]]
    if len(results) != 512:
        raise RuntimeError("DSET-Q35 row coverage differs")
    row_keys = {
        (row["source_identity_sha256"], row["pair_member"])
        for row in results
    }
    if len(row_keys) != 512:
        raise RuntimeError("DSET-Q35 row identity coverage differs")
    family = defaultdict(Counter)
    member = defaultdict(Counter)
    errors = Counter()
    for row in results:
        for group, key in ((family, row["corruption_family"]), (member, row["pair_member"])):
            group[key]["rows"] += 1
            group[key]["script_exact"] += int(row["script_exact"])
            group[key]["execution_correct"] += int(row["execution_correct"])
        if row["execution_error"]:
            errors[row["execution_error"]] += 1
    script = sum(row["script_exact"] for row in results)
    execution = sum(row["execution_correct"] for row in results)
    clean = member["clean"]
    fault = member["fault"]
    gates = {
        "execution_accuracy_ge_0_95": execution / 512 >= 0.95,
        "numeric_script_accuracy_ge_0_90": family["numeric_final"]["script_exact"] / family["numeric_final"]["rows"] >= 0.90,
        "choice_script_accuracy_ge_0_90": family["choice_final"]["script_exact"] / family["choice_final"]["rows"] >= 0.90,
        "clean_copy_ge_0_99": clean["execution_correct"] / clean["rows"] >= 0.99,
        "fault_repair_ge_0_90": fault["execution_correct"] / fault["rows"] >= 0.90,
        "zero_execution_errors": not errors,
        "zero_exhaustion": sum(row["max_token_exhausted"] for row in results) == 0,
    }
    passed = all(gates.values())
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "thresholds_frozen_before_output": True,
        "holdout_used": False,
        "row_count": 512,
        "pair_count": 256,
        "script_exact": script,
        "execution_correct": execution,
        "family_counts": {key: dict(value) for key, value in family.items()},
        "member_counts": {key: dict(value) for key, value in member.items()},
        "execution_errors": dict(errors),
        "max_token_exhausted": sum(row["max_token_exhausted"] for row in results),
        "generated_tokens": sum(row["generated_tokens"] for row in results),
        "elapsed_seconds_sum": sum(float(report["elapsed_seconds"]) for report in reports),
        "peak_gpu_memory_bytes_max": max(int(report["peak_gpu_memory_bytes"]) for report in reports),
        "gates": gates,
        "passed": passed,
        "trained_transfer_authorized": passed,
        "decision": "freeze_trained_dset_q35_transfer" if passed else "close_untrained_dset_q35_ceiling",
        "results": sorted(results, key=lambda row: (row["source_identity_sha256"], row["pair_member"])),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.inputs, args.output)
    print(json.dumps({"gates": report["gates"], "passed": report["passed"]}, sort_keys=True))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
