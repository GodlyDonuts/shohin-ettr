#!/usr/bin/env python3
"""Merge complete, disjoint DSET1 evaluation shards."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from typing import Any

from eval_dset1_span_edit import REPORT_SCHEMA


MERGED_SCHEMA = "shohin-dset1-span-edit-evaluation-merged-v1"


class DSET1MergeError(RuntimeError):
    """The DSET1 shard set is incomplete or inconsistent."""


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def rates(counts: Counter[str]) -> dict[str, Any]:
    rows = int(counts["rows"])
    result = dict(counts)
    for key in ("action_correct", "script_exact", "execution_correct", "trajectory_exact"):
        result[f"{key}_accuracy"] = counts[key] / rows
    return result


def merge(paths: list[Path], output: Path) -> dict[str, Any]:
    if output.exists() or not paths:
        raise DSET1MergeError("DSET1 merge output exists or input is empty")
    reports = [json.loads(path.read_text()) for path in paths]
    first = reports[0]
    fixed = (
        "status",
        "arm",
        "model_root",
        "model_revision",
        "adapter_checkpoint_sha256",
        "data_sha256",
        "data_report_sha256",
        "shard_count",
        "max_new_tokens",
    )
    if any(
        report.get("schema") != REPORT_SCHEMA
        or any(report.get(key) != first.get(key) for key in fixed)
        for report in reports
    ):
        raise DSET1MergeError("DSET1 shard contract differs")
    count = int(first["shard_count"])
    if len(reports) != count or {int(report["shard_index"]) for report in reports} != set(range(count)):
        raise DSET1MergeError("DSET1 shard coverage differs")
    results = [row for report in reports for row in report["results"]]
    if len({row["identity_sha256"] for row in results}) != len(results):
        raise DSET1MergeError("DSET1 identity duplicated")
    pairs = defaultdict(list)
    family = defaultdict(Counter)
    member = defaultdict(Counter)
    errors = Counter()
    for row in results:
        pairs[row["pair_identity_sha256"]].append(row)
        if row.get("execution_error"):
            errors[row["execution_error"]] += 1
        for groups, name in ((family, row["corruption_family"]), (member, row["pair_member"])):
            groups[name]["rows"] += 1
            for key in ("action_correct", "script_exact", "execution_correct", "trajectory_exact"):
                groups[name][key] += int(row[key])
    if any(len(pair) != 2 or {row["pair_member"] for row in pair} != {"clean", "fault"} for pair in pairs.values()):
        raise DSET1MergeError("DSET1 pair coverage differs")
    consistent = sum(all(row["script_exact"] for row in pair) for pair in pairs.values())
    elapsed = sum(float(report["elapsed_seconds"]) for report in reports)
    generated = sum(int(report["generated_tokens"]) for report in reports)
    merged = {
        "schema": MERGED_SCHEMA,
        "status": "complete",
        "arm": first["arm"],
        "model_root": first["model_root"],
        "model_revision": first["model_revision"],
        "adapter_checkpoint_sha256": first["adapter_checkpoint_sha256"],
        "data_sha256": first["data_sha256"],
        "data_report_sha256": first["data_report_sha256"],
        "holdout_used": False,
        "input_shards": [str(path.resolve()) for path in paths],
        "shard_count": count,
        "pair_count": len(pairs),
        "row_count": len(results),
        "action_correct": sum(row["action_correct"] for row in results),
        "script_exact": sum(row["script_exact"] for row in results),
        "execution_correct": sum(row["execution_correct"] for row in results),
        "trajectory_exact": sum(row["trajectory_exact"] for row in results),
        "counterfactual_consistent_pairs": consistent,
        "counterfactual_consistency": consistent / len(pairs),
        "family_metrics": {name: rates(value) for name, value in family.items()},
        "member_metrics": {name: rates(value) for name, value in member.items()},
        "execution_errors": dict(errors),
        "generated_tokens": generated,
        "max_token_exhausted": sum(int(report["max_token_exhausted"]) for report in reports),
        "aggregate_gpu_seconds": elapsed,
        "generated_tokens_per_gpu_second": generated / elapsed,
        "peak_gpu_memory_bytes": max(int(report["peak_gpu_memory_bytes"]) for report in reports),
        "results": sorted(results, key=lambda row: row["identity_sha256"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, merged)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = merge(args.inputs, args.output)
    print(json.dumps({key: report[key] for key in ("arm", "script_exact", "execution_correct", "counterfactual_consistency")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
