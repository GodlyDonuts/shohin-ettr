#!/usr/bin/env python3
"""Aggregate the three prospective dense screens and decide full-run promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCORE_SCHEMA = "shohin-dense-public-benchmark-score-v1"
SCHEMA = "shohin-dense-public-screen-analysis-v1"
BENCHMARKS = ("mmlu_pro", "ifeval", "musr")


class DenseScreenAnalysisError(RuntimeError):
    """A screen score, host binding, or prospective gate differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(paths: list[Path], output: Path) -> dict[str, Any]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if len(reports) != 3 or {report.get("benchmark") for report in reports} != set(
        BENCHMARKS
    ):
        raise DenseScreenAnalysisError("exactly three benchmark reports are required")
    first = reports[0]
    bindings = (
        "host",
        "model_revision",
        "draft_checkpoint_sha256",
        "revision_checkpoint_sha256",
    )
    for report in reports:
        if (
            report.get("schema") != SCORE_SCHEMA
            or report.get("status") != "complete"
            or report.get("label") != "prospective_256_row_screen_not_full_benchmark"
            or report.get("rows") != 256
            or any(report.get(key) != first.get(key) for key in bindings)
        ):
            raise DenseScreenAnalysisError("screen report binding differs")
    unchanged = sum(report["metrics"]["unchanged_correct"] for report in reports)
    revision = sum(report["metrics"]["trained_revision_correct"] for report in reports)
    wins = sum(report["metrics"]["wins"] for report in reports)
    losses = sum(report["metrics"]["losses"] for report in reports)
    retained = unchanged - losses
    per_benchmark = {
        report["benchmark"]: {
            "unchanged_correct": report["metrics"]["unchanged_correct"],
            "trained_revision_correct": report["metrics"]["trained_revision_correct"],
            "delta_count": report["metrics"]["paired_delta_count"],
            "delta_points": report["metrics"]["paired_delta_points"],
            "retention": report["metrics"]["baseline_correct_retention"],
        }
        for report in reports
    }
    checks = {
        "combined_gain_at_least_2_points": revision - unchanged >= 16,
        "baseline_correct_retention_at_least_95_percent": unchanged == 0
        or retained / unchanged >= 0.95,
        "no_benchmark_loses_more_than_2_items": all(
            value["delta_count"] >= -2 for value in per_benchmark.values()
        ),
        "at_least_two_benchmarks_nonnegative": sum(
            value["delta_count"] >= 0 for value in per_benchmark.values()
        )
        >= 2,
    }
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "label": "prospective_768_row_broad_screen_not_full_benchmarks",
        "host": first["host"],
        "model_revision": first["model_revision"],
        "draft_checkpoint_sha256": first["draft_checkpoint_sha256"],
        "revision_checkpoint_sha256": first["revision_checkpoint_sha256"],
        "screen_reports": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in paths
        ],
        "rows": 768,
        "combined": {
            "unchanged_correct": unchanged,
            "trained_revision_correct": revision,
            "delta_count": revision - unchanged,
            "delta_points": 100 * (revision - unchanged) / 768,
            "wins": wins,
            "losses": losses,
            "retained": retained,
            "retention": retained / unchanged if unchanged else 1.0,
        },
        "benchmarks": per_benchmark,
        "promotion_checks": checks,
        "promote_to_full_confirmations": all(checks.values()),
        "stop_this_host_after_screen": not all(checks.values()),
    }
    if output.exists():
        raise DenseScreenAnalysisError("analysis output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run(args.score, args.output)
    print(
        json.dumps(
            {
                "host": payload["host"],
                "promote": payload["promote_to_full_confirmations"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
