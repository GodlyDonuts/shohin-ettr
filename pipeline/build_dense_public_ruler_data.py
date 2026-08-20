#!/usr/bin/env python3
"""Materialize an official RULER screen without duplicating other benchmark data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_dense_public_site_benchmark_data import (
    REPORT_SCHEMA,
    SiteBenchmarkDataError,
    atomic_json,
    load_ruler,
    sha256_file,
    write_benchmark,
)


def run(args: argparse.Namespace) -> dict:
    if args.output_root.exists():
        raise SiteBenchmarkDataError("RULER output root already exists")
    rows = load_ruler(args.ruler_jsonl)
    output = write_benchmark(args.output_root, "ruler", rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "benchmarks": {"ruler": output},
        "source_files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.ruler_jsonl
        ],
        "ruler_scope": (
            "official_generator_screen_50_rows_per_task_at_4k_8k_16k_32k; "
            "not_a_full_RULER_leaderboard_placement"
        ),
        "ruler_source_commit": args.ruler_commit,
        "assessors_visible_to_model": False,
    }
    atomic_json(args.output_root / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruler-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--ruler-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps({"ruler": report["benchmarks"]["ruler"]["rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
