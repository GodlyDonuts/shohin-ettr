#!/usr/bin/env python3
"""Merge complete disjoint GSET1 evaluation shards."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path


SHARD_SCHEMA = "shohin-gset1-causal-gate-evaluation-v1"
MERGED_SCHEMA = "shohin-gset1-causal-gate-evaluation-merged-v1"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def rates(counts: Counter) -> dict:
    rows = int(counts["rows"])
    payload = dict(counts)
    for key in ("gate_action_correct", "script_exact", "execution_correct"):
        payload[f"{key}_accuracy"] = int(counts[key]) / rows
    return payload


def merge(paths: list[Path], output: Path) -> dict:
    if output.exists() or not paths:
        raise RuntimeError("GSET1 merge output exists or inputs are empty")
    reports = [json.loads(path.read_text()) for path in paths]
    first = reports[0]
    fixed = (
        "status", "arm", "intervention", "feature_control", "shard_count",
        "dset_checkpoint_sha256", "gate_checkpoint_sha256", "data_sha256", "model_loader",
    )
    if any(
        report.get("schema") != SHARD_SCHEMA
        or any(report.get(key) != first.get(key) for key in fixed)
        for report in reports
    ):
        raise RuntimeError("GSET1 shard contract differs")
    count = int(first["shard_count"])
    if len(reports) != count or {int(report["shard_index"]) for report in reports} != set(range(count)):
        raise RuntimeError("GSET1 shard coverage differs")
    results = [row for report in reports for row in report["results"]]
    if len({row["identity_sha256"] for row in results}) != len(results):
        raise RuntimeError("GSET1 identities are duplicated")
    pairs = defaultdict(list)
    family, member = defaultdict(Counter), defaultdict(Counter)
    errors = Counter()
    for row in results:
        pairs[row["pair_identity_sha256"]].append(row)
        if row.get("execution_error"):
            errors[row["execution_error"]] += 1
        for grouping, key in ((family, row["corruption_family"]), (member, row["pair_member"])):
            grouping[key]["rows"] += 1
            for metric in ("gate_action_correct", "script_exact", "execution_correct"):
                grouping[key][metric] += int(row[metric])
    if any(len(pair) != 2 or {item["pair_member"] for item in pair} != {"clean", "fault"} for pair in pairs.values()):
        raise RuntimeError("GSET1 pair coverage differs")
    consistent = sum(all(item["gate_action_correct"] for item in pair) for pair in pairs.values())
    payload = {
        "schema": MERGED_SCHEMA,
        "status": "complete",
        "arm": first["arm"],
        "intervention": first["intervention"],
        "feature_control": first["feature_control"],
        "holdout_used": False,
        "shard_count": count,
        "input_shards": [str(path.resolve()) for path in paths],
        "dset_checkpoint_sha256": first["dset_checkpoint_sha256"],
        "gate_checkpoint_sha256": first["gate_checkpoint_sha256"],
        "data_sha256": first["data_sha256"],
        "pair_count": len(pairs),
        "row_count": len(results),
        "gate_action_correct": sum(int(row["gate_action_correct"]) for row in results),
        "script_exact": sum(int(row["script_exact"]) for row in results),
        "execution_correct": sum(int(row["execution_correct"]) for row in results),
        "counterfactual_consistent_pairs": consistent,
        "counterfactual_consistency": consistent / len(pairs),
        "family_metrics": {key: rates(value) for key, value in family.items()},
        "member_metrics": {key: rates(value) for key, value in member.items()},
        "execution_errors": dict(errors),
        "generated_tokens": sum(int(report["generated_tokens"]) for report in reports),
        "max_token_exhausted": sum(int(report["max_token_exhausted"]) for report in reports),
        "aggregate_gpu_seconds": sum(float(report["elapsed_seconds"]) for report in reports),
        "peak_gpu_memory_bytes": max(int(report["peak_gpu_memory_bytes"]) for report in reports),
        "results": sorted(results, key=lambda row: row["identity_sha256"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = merge(args.inputs, args.output)
    print(json.dumps({key: report[key] for key in ("arm", "gate_action_correct", "execution_correct")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
