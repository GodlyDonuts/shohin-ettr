#!/usr/bin/env python3
"""Merge complete, batch-aligned IDR1 evaluation shards and score once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hf_idr1_evaluate_reviser import (
    REPORT_SCHEMA,
    IDR1EvaluationError,
    load_rows,
    shard_bounds,
    summarize,
)
from hf_vcr1_evaluate_reviser import _atomic_json, _atomic_lines, sha256_file


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IDR1EvaluationError(f"JSON object required: {path}")
    return value


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.report.exists() or args.candidates_output.exists():
        raise IDR1EvaluationError("IDR1 merged output already exists")
    rows = load_rows(args.data, args.split)
    reports = [_load_json(path) for path in args.shard_report]
    shard_count = len(reports)
    if shard_count < 2:
        raise IDR1EvaluationError("at least two IDR1 shard reports are required")

    results_by_identity: dict[str, dict[str, Any]] = {}
    shared_keys = (
        "split",
        "model_root",
        "model_revision",
        "adapter_checkpoint_sha256",
        "data_sha256",
        "data_report_sha256",
        "generation_mode",
        "max_new_tokens",
        "batch_size",
        "seed",
        "ecr_code_intervention",
    )
    reference = reports[0]
    for report_path, report in zip(args.shard_report, reports, strict=True):
        if report.get("schema") != REPORT_SCHEMA or report.get("status") != "complete":
            raise IDR1EvaluationError("IDR1 shard report is incomplete")
        if any(report.get(key) != reference.get(key) for key in shared_keys):
            raise IDR1EvaluationError("IDR1 shard settings differ")
        if report.get("shard_count") != shard_count:
            raise IDR1EvaluationError("IDR1 shard count differs")
        index = report.get("shard_index")
        if not isinstance(index, int) or not 0 <= index < shard_count:
            raise IDR1EvaluationError("IDR1 shard index is invalid")
        start, end = shard_bounds(len(rows), index, shard_count, report["batch_size"])
        if report.get("row_start") != start or report.get("row_end") != end:
            raise IDR1EvaluationError("IDR1 shard bounds differ")
        candidate_path = Path(report.get("candidates_output", ""))
        if report.get("candidates_sha256") != sha256_file(candidate_path):
            raise IDR1EvaluationError("IDR1 shard candidate hash differs")
        candidates = _load_candidates(candidate_path)
        expected_ids = [row["identity_sha256"] for row in rows[start:end]]
        if [row.get("identity_sha256") for row in candidates] != expected_ids:
            raise IDR1EvaluationError("IDR1 shard identity coverage differs")
        for candidate in candidates:
            identity = candidate["identity_sha256"]
            if identity in results_by_identity:
                raise IDR1EvaluationError("IDR1 shard identity is duplicated")
            results_by_identity[identity] = candidate

    if {report["shard_index"] for report in reports} != set(range(shard_count)):
        raise IDR1EvaluationError("IDR1 shard index coverage differs")
    results = [results_by_identity[row["identity_sha256"]] for row in rows]
    candidate_sha256 = _atomic_lines(args.candidates_output, results)
    summary = summarize(rows, results, args.split)
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
        "generated_tokens": sum(report["generated_tokens"] for report in reports),
        "max_token_exhausted": sum(
            report["max_token_exhausted"] for report in reports
        ),
        "aggregate_gpu_seconds": sum(report["elapsed_seconds"] for report in reports),
        "critical_path_seconds": max(report["elapsed_seconds"] for report in reports),
        "peak_gpu_memory_bytes": max(
            report["peak_gpu_memory_bytes"] for report in reports
        ),
        "routing_receipts": [report.get("routing_receipt") for report in reports],
        "candidates_output": str(args.candidates_output.resolve()),
        "candidates_sha256": candidate_sha256,
        **summary,
    }
    _atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--shard-report", type=Path, action="append", required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"gate": report["gate"], "metrics": report["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
