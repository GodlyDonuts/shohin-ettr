#!/usr/bin/env python3
"""Merge complete batch-aligned TTR1 control shards and score once."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from hf_idr1_evaluate_reviser import load_rows, shard_bounds
from hf_vcr1_evaluate_reviser import _atomic_json, _atomic_lines, sha256_file, summarize
from ttr1_control_evaluate import REPORT_SCHEMA, TTR1ControlError


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TTR1ControlError(f"JSON object required: {path}")
    return value


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TTR1ControlError("TTR1 candidate must be an object")
                rows.append(value)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.report.exists() or args.candidates_output.exists():
        raise TTR1ControlError("TTR1 merged output already exists")
    rows = load_rows(args.data, args.split)
    reports = [_load_json(path) for path in args.shard_report]
    shard_count = len(reports)
    if shard_count < 2:
        raise TTR1ControlError("at least two TTR1 shard reports are required")

    shared_keys = (
        "control",
        "split",
        "model_root",
        "model_revision",
        "adapter_checkpoint_sha256",
        "data_sha256",
        "data_report_sha256",
        "model_loader",
        "max_new_tokens_per_attempt",
        "attempts_per_identity",
        "batch_size",
        "seed",
    )
    reference = reports[0]
    results_by_identity: dict[str, dict[str, Any]] = {}
    counters: Counter[str] = Counter()
    for report_path, report in zip(args.shard_report, reports, strict=True):
        if report.get("schema") != REPORT_SCHEMA or report.get("status") != "complete":
            raise TTR1ControlError("TTR1 shard report is incomplete")
        if any(report.get(key) != reference.get(key) for key in shared_keys):
            raise TTR1ControlError("TTR1 shard settings differ")
        if report.get("shard_count") != shard_count:
            raise TTR1ControlError("TTR1 shard count differs")
        index = report.get("shard_index")
        if not isinstance(index, int) or not 0 <= index < shard_count:
            raise TTR1ControlError("TTR1 shard index is invalid")
        start, end = shard_bounds(
            len(rows), index, shard_count, int(report["batch_size"])
        )
        if report.get("row_start") != start or report.get("row_end") != end:
            raise TTR1ControlError("TTR1 shard bounds differ")
        candidate_path = Path(report.get("candidates_output", ""))
        if report.get("candidates_sha256") != sha256_file(candidate_path):
            raise TTR1ControlError("TTR1 shard candidate hash differs")
        candidates = _load_candidates(candidate_path)
        expected_ids = [row["identity_sha256"] for row in rows[start:end]]
        if [row.get("identity_sha256") for row in candidates] != expected_ids:
            raise TTR1ControlError("TTR1 shard identity coverage differs")
        for candidate in candidates:
            if (
                candidate.get("schema") != "shohin-ttr1-control-candidate-v1"
                or candidate.get("control") != reference["control"]
            ):
                raise TTR1ControlError("TTR1 shard candidate contract differs")
            identity = candidate["identity_sha256"]
            if identity in results_by_identity:
                raise TTR1ControlError("TTR1 shard identity is duplicated")
            results_by_identity[identity] = candidate
        counters.update(report.get("counters", {}))

    if {report["shard_index"] for report in reports} != set(range(shard_count)):
        raise TTR1ControlError("TTR1 shard index coverage differs")
    results = [results_by_identity[row["identity_sha256"]] for row in rows]
    candidates_sha256 = _atomic_lines(args.candidates_output, results)
    report = {
        **{key: reference[key] for key in shared_keys},
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "merged_from_shards": True,
        "shard_count": shard_count,
        "shard_reports": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.shard_report
        ],
        "full_row_count": len(rows),
        "counters": dict(sorted(counters.items())),
        "aggregate_gpu_seconds": sum(
            float(item["elapsed_seconds"]) for item in reports
        ),
        "critical_path_seconds": max(
            float(item["elapsed_seconds"]) for item in reports
        ),
        "generated_tokens_per_second": counters["generated_tokens"]
        / sum(float(item["elapsed_seconds"]) for item in reports),
        "peak_gpu_memory_bytes": max(
            int(item["peak_gpu_memory_bytes"]) for item in reports
        ),
        "metrics": summarize(rows, results)["metrics"],
        "candidates_output": str(args.candidates_output.resolve()),
        "candidates_sha256": candidates_sha256,
    }
    _atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "holdout"), required=True)
    parser.add_argument("--shard-report", type=Path, action="append", required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"control": report["control"], "metrics": report["metrics"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
