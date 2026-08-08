#!/usr/bin/env python3
"""Reconstruct the source-only SDR1 product board."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_vcr1_product_data import (
    PAIR_SCHEMA,
    VCR1ProductDataError,
    _atomic_json,
    _atomic_lines,
    assessor_rows,
    load_pairs,
)
from build_vcr1_revision_data import sha256_file
from hf_product_reasoning_eval import _task_prompt

OUTPUT_SCHEMA = "shohin-sdr1-product-eval-v1"
REPORT_SCHEMA = "shohin-sdr1-product-data-report-v1"


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.report.exists():
        raise VCR1ProductDataError("SDR1 product output already exists")
    merge = json.loads(args.pair_report.read_text(encoding="utf-8"))
    if merge.get("schema") != PAIR_SCHEMA or merge.get("status") != "complete":
        raise VCR1ProductDataError("SDR1 product pair report is incomplete")
    if Path(merge.get("pairs", "")).resolve() != args.pairs.resolve():
        raise VCR1ProductDataError("SDR1 product pair path differs")
    if merge.get("pairs_sha256") != sha256_file(args.pairs):
        raise VCR1ProductDataError("SDR1 product pair hash differs")
    pairs = load_pairs(args.pairs)
    assessors = assessor_rows(merge)
    if set(assessors) != {row["identity_sha256"] for row in pairs}:
        raise VCR1ProductDataError("SDR1 product assessor coverage differs")
    output_rows: list[dict[str, Any]] = []
    for row in pairs:
        identity = row["identity_sha256"]
        source = assessors[identity]
        if row.get("task") != source.get("task"):
            raise VCR1ProductDataError("SDR1 product task binding differs")
        candidates = {item["lineage"]: item for item in row["candidates"]}
        if set(candidates) != {"base", "expert"}:
            raise VCR1ProductDataError("SDR1 product lineages differ")
        output_rows.append(
            {
                "schema": OUTPUT_SCHEMA,
                "identity_sha256": identity,
                "task": row["task"],
                "question": _task_prompt(str(row["task"]), source),
                "candidates": [candidates["base"], candidates["expert"]],
                "assessor": source,
                "runtime_fields": ["question"],
                "candidate_text_visible": False,
            }
        )
    output_sha256 = _atomic_lines(args.output, output_rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "pairs": str(args.pairs.resolve()),
        "pairs_sha256": sha256_file(args.pairs),
        "pair_report": str(args.pair_report.resolve()),
        "pair_report_sha256": sha256_file(args.pair_report),
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
        "rows": len(output_rows),
        "runtime_fields": ["question"],
        "candidate_text_visible": False,
        "assessor_fields_visible_to_model": False,
        "tasks": sorted({row["task"] for row in output_rows}),
    }
    _atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--pair-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    report = build(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
