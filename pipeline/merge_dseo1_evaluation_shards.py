#!/usr/bin/env python3
"""Merge complete disjoint DSEO1 paired-evaluation shards."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from typing import Any

from eval_dseo1_paired_action import REPORT_SCHEMA


class DSEO1MergeError(RuntimeError):
    """The DSEO1 shard set is incomplete, duplicated, or inconsistent."""


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def merge(paths: list[Path], output: Path) -> dict[str, Any]:
    if output.exists() or not paths:
        raise DSEO1MergeError("DSEO1 merge output exists or inputs are empty")
    reports = [json.loads(path.read_text()) for path in paths]
    first = reports[0]
    fixed = (
        "schema",
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
    for report in reports:
        if any(report.get(key) != first.get(key) for key in fixed):
            raise DSEO1MergeError("DSEO1 shard contract differs")
    shard_count = int(first["shard_count"])
    if (
        first.get("schema") != REPORT_SCHEMA
        or first.get("status") != "complete"
        or len(reports) != shard_count
        or {int(report["shard_index"]) for report in reports} != set(range(shard_count))
    ):
        raise DSEO1MergeError("DSEO1 shard coverage differs")
    results = [row for report in reports for row in report["results"]]
    identities = [row["identity_sha256"] for row in results]
    if len(identities) != len(set(identities)):
        raise DSEO1MergeError("DSEO1 evaluation identity duplicated")
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family = defaultdict(Counter)
    member = defaultdict(Counter)
    for row in results:
        by_pair[row["pair_identity_sha256"]].append(row)
        for key in ("action_correct", "answer_correct"):
            family[row["corruption_family"]][key] += int(row[key])
            member[row["pair_member"]][key] += int(row[key])
        family[row["corruption_family"]]["rows"] += 1
        member[row["pair_member"]]["rows"] += 1
    if any(
        len(pair) != 2 or {row["pair_member"] for row in pair} != {"clean", "fault"}
        for pair in by_pair.values()
    ):
        raise DSEO1MergeError("DSEO1 merged pair coverage differs")
    consistent = sum(
        all(row["action_correct"] for row in pair)
        and len({row["predicted_action"] for row in pair}) == 2
        for pair in by_pair.values()
    )

    def rates(counts: Counter[str]) -> dict[str, Any]:
        rows = int(counts["rows"])
        return {
            **dict(counts),
            "action_accuracy": counts["action_correct"] / rows,
            "answer_accuracy": counts["answer_correct"] / rows,
        }

    elapsed = sum(float(report["elapsed_seconds"]) for report in reports)
    generated = sum(int(report["generated_tokens"]) for report in reports)
    merged = {
        **{key: first[key] for key in fixed if key not in {"shard_count"}},
        "schema": "shohin-dseo1-paired-evaluation-merged-v1",
        "status": "complete",
        "input_shards": [str(path.resolve()) for path in paths],
        "shard_count": shard_count,
        "pair_count": len(by_pair),
        "row_count": len(results),
        "action_correct": sum(row["action_correct"] for row in results),
        "action_accuracy": sum(row["action_correct"] for row in results) / len(results),
        "answer_correct": sum(row["answer_correct"] for row in results),
        "answer_accuracy": sum(row["answer_correct"] for row in results) / len(results),
        "counterfactual_consistent_pairs": consistent,
        "counterfactual_consistency": consistent / len(by_pair),
        "family_metrics": {name: rates(counts) for name, counts in family.items()},
        "member_metrics": {name: rates(counts) for name, counts in member.items()},
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
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "arm",
                    "action_accuracy",
                    "answer_accuracy",
                    "counterfactual_consistency",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
