#!/usr/bin/env python3
"""Merge exact PSET1 evaluation shards."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from typing import Any

from eval_pset1_pointer import REPORT_SCHEMA


MERGED_SCHEMA = "shohin-pset1-pointer-evaluation-merged-v1"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def merge(paths: list[Path], output: Path) -> dict[str, Any]:
    if output.exists() or not paths:
        raise RuntimeError("PSET1 merge output exists or inputs absent")
    reports = [json.loads(path.read_text()) for path in paths]
    first = reports[0]
    fixed = (
        "arm", "intervention", "shard_count", "data_sha256",
        "data_report_sha256", "pointer_checkpoint_sha256", "host_checkpoint_sha256",
    )
    if any(report.get("schema") != REPORT_SCHEMA or report.get("status") != "complete" for report in reports):
        raise RuntimeError("PSET1 shard differs")
    if any(any(report.get(key) != first.get(key) for key in fixed) for report in reports):
        raise RuntimeError("PSET1 shard geometry differs")
    if {int(report["shard_index"]) for report in reports} != set(range(int(first["shard_count"]))):
        raise RuntimeError("PSET1 shard coverage differs")
    results = [row for report in reports for row in report["results"]]
    if len(results) != 512 or len({(row["source_identity_sha256"], row["pair_member"]) for row in results}) != 512:
        raise RuntimeError("PSET1 merged row coverage differs")
    counts = Counter()
    family = defaultdict(Counter)
    member = defaultdict(Counter)
    errors = Counter()
    for result in results:
        for key in ("program_exact", "execution_correct", "replacement_finished", "force_keep_breaks_correct_repair"):
            value = int(result[key])
            counts[key] += value
            family[result["corruption_family"]][key] += value
            member[result["pair_member"]][key] += value
        family[result["corruption_family"]]["rows"] += 1
        member[result["pair_member"]]["rows"] += 1
        if result["execution_error"]:
            errors[result["execution_error"]] += 1
    by_pair = defaultdict(list)
    for result in results:
        by_pair[result["source_identity_sha256"]].append(result)
    consistency = sum(len(rows) == 2 and all(row["program_exact"] for row in rows) for rows in by_pair.values())
    payload = {
        "schema": MERGED_SCHEMA,
        "status": "complete",
        "arm": first["arm"],
        "intervention": first["intervention"],
        "holdout_used": False,
        "pair_count": 256,
        "row_count": 512,
        **counts,
        "counterfactual_consistent_pairs": consistency,
        "execution_errors": dict(errors),
        "family_counts": {key: dict(value) for key, value in family.items()},
        "member_counts": {key: dict(value) for key, value in member.items()},
        "data_sha256": first["data_sha256"],
        "data_report_sha256": first["data_report_sha256"],
        "pointer_checkpoint_sha256": first["pointer_checkpoint_sha256"],
        "host_checkpoint_sha256": first["host_checkpoint_sha256"],
        "generated_bytes": sum(int(report["generated_bytes"]) for report in reports),
        "elapsed_seconds_sum": sum(float(report["elapsed_seconds"]) for report in reports),
        "peak_gpu_memory_bytes_max": max(int(report["peak_gpu_memory_bytes"]) for report in reports),
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
    merge(args.inputs, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
