#!/usr/bin/env python3
"""Reconstruct the preserved product board for conditional VCR1 scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_vcr1_revision_data import revision_prompt, sha256_file, source_task_prompt
from hf_product_reasoning_eval import _question, select_rows

PAIR_SCHEMA = "shohin-cvg1-evaluation-pairs-v1"
OUTPUT_SCHEMA = "shohin-vcr1-product-eval-v1"
REPORT_SCHEMA = "shohin-vcr1-product-data-report-v1"


class VCR1ProductDataError(RuntimeError):
    """The preserved product reports or reconstructed assessor rows differ."""


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise VCR1ProductDataError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise VCR1ProductDataError(f"refusing existing report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_pairs(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows or any(row.get("schema") != PAIR_SCHEMA for row in rows):
        raise VCR1ProductDataError("VCR1 product pair schema differs")
    if len({row["identity_sha256"] for row in rows}) != len(rows):
        raise VCR1ProductDataError("VCR1 product pair identity is duplicated")
    return rows


def assessor_rows(merge_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    inputs = merge_report.get("inputs", {}).get("base")
    if not isinstance(inputs, dict):
        raise VCR1ProductDataError("VCR1 product input report map is missing")
    for task, receipt in sorted(inputs.items()):
        report_path = Path(receipt["path"])
        if sha256_file(report_path) != receipt["sha256"]:
            raise VCR1ProductDataError("VCR1 source report hash differs")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        data_path = Path(report["data"])
        if sha256_file(data_path) != report["data_sha256"]:
            raise VCR1ProductDataError("VCR1 source data hash differs")
        data_rows = [
            json.loads(line)
            for line in data_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        selected = select_rows(
            task,
            data_rows,
            int(report["total"]),
            int(report["subset_seed"]),
        )
        results = report["results"]
        if len(selected) != len(results):
            raise VCR1ProductDataError("VCR1 source selection cardinality differs")
        for source, result in zip(selected, results, strict=True):
            if _question(source) != result["question"]:
                raise VCR1ProductDataError("VCR1 source question binding differs")
            identity = result["identity_sha256"]
            if identity in mapped:
                raise VCR1ProductDataError("VCR1 source identity is duplicated")
            mapped[identity] = {**source, "task": task}
    return mapped


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.report.exists():
        raise VCR1ProductDataError("VCR1 product output already exists")
    merge = json.loads(args.pair_report.read_text(encoding="utf-8"))
    if merge.get("schema") != PAIR_SCHEMA or merge.get("status") != "complete":
        raise VCR1ProductDataError("VCR1 product pair report is incomplete")
    if Path(merge.get("pairs", "")).resolve() != args.pairs.resolve():
        raise VCR1ProductDataError("VCR1 product pair path differs")
    if merge.get("pairs_sha256") != sha256_file(args.pairs):
        raise VCR1ProductDataError("VCR1 product pair hash differs")
    pairs = load_pairs(args.pairs)
    assessors = assessor_rows(merge)
    if set(assessors) != {row["identity_sha256"] for row in pairs}:
        raise VCR1ProductDataError("VCR1 product assessor coverage differs")
    output_rows: list[dict[str, Any]] = []
    for row in pairs:
        identity = row["identity_sha256"]
        source = assessors[identity]
        if row.get("task") != source.get("task"):
            raise VCR1ProductDataError("VCR1 product task binding differs")
        candidates = {item["lineage"]: item for item in row["candidates"]}
        if set(candidates) != {"base", "expert"}:
            raise VCR1ProductDataError("VCR1 product lineages differ")
        output_rows.append(
            {
                "schema": OUTPUT_SCHEMA,
                "identity_sha256": identity,
                "task": row["task"],
                "question": revision_prompt(
                    source_task_prompt(source),
                    str(candidates["base"]["completion"]),
                    str(candidates["expert"]["completion"]),
                ),
                "candidates": [candidates["base"], candidates["expert"]],
                "assessor": source,
                "runtime_fields": ["question"],
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
