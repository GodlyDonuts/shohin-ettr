#!/usr/bin/env python3
"""Aggregate one-round visible-test code repairs with complete coverage checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-visible-code-repair-aggregate-v1"


class VisibleCodeRepairAggregateError(RuntimeError):
    """Repair shards do not form one complete, disjoint evaluation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VisibleCodeRepairAggregateError(f"malformed JSON: {path}") from exc


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_bytes().splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VisibleCodeRepairAggregateError(f"malformed JSONL: {path}") from exc


def _eval_identity(row: dict[str, Any]) -> str:
    task = str(row.get("task") or "")
    question = next(
        (
            str(row[key])
            for key in ("question", "problem", "prompt", "text", "input")
            if row.get(key)
        ),
        "",
    )
    if not task or not question:
        raise VisibleCodeRepairAggregateError("repair row identity cannot be derived")
    return hashlib.sha256(f"{task}\0{question}".encode()).hexdigest()


def aggregate(
    repair_rows: list[dict[str, Any]],
    build_report: dict[str, Any],
    eval_reports: list[dict[str, Any]],
    *,
    eval_paths: list[Path],
) -> dict[str, Any]:
    if build_report.get("schema") != "shohin-visible-code-repair-bank-v1":
        raise VisibleCodeRepairAggregateError("repair build schema differs")
    if build_report.get("status") != "complete":
        raise VisibleCodeRepairAggregateError("repair build is incomplete")
    if int(build_report.get("repair_rows", -1)) != len(repair_rows):
        raise VisibleCodeRepairAggregateError("repair row count differs")

    by_eval_identity: dict[str, dict[str, Any]] = {}
    original_identities: set[str] = set()
    for row in repair_rows:
        if row.get("repair_schema") != "shohin-visible-code-repair-bank-v1":
            raise VisibleCodeRepairAggregateError("repair row schema differs")
        original = str(row.get("original_identity_sha256") or "")
        if not original or original in original_identities:
            raise VisibleCodeRepairAggregateError("repair source identities repeat")
        original_identities.add(original)
        identity = _eval_identity(row)
        if identity in by_eval_identity:
            raise VisibleCodeRepairAggregateError("repair prompt identities repeat")
        by_eval_identity[identity] = row

    observed: dict[str, dict[str, Any]] = {}
    report_receipts = []
    for path, report in zip(eval_paths, eval_reports, strict=True):
        if report.get("status") != "complete" or report.get("task") != "mbpp":
            raise VisibleCodeRepairAggregateError("evaluation report differs")
        results = report.get("results") or []
        if int(report.get("total", -1)) != len(results):
            raise VisibleCodeRepairAggregateError("evaluation result count differs")
        if int(report.get("correct", -1)) != sum(
            bool(row.get("correct")) for row in results
        ):
            raise VisibleCodeRepairAggregateError("evaluation correct count differs")
        for result in results:
            identity = str(result.get("identity_sha256") or "")
            if identity not in by_eval_identity:
                raise VisibleCodeRepairAggregateError(
                    "evaluation identity is outside repair bank"
                )
            if identity in observed:
                raise VisibleCodeRepairAggregateError("evaluation identities overlap")
            observed[identity] = result
        report_receipts.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "total": len(results),
                "correct": int(report["correct"]),
                "data_sha256": str(report.get("data_sha256") or ""),
            }
        )
    if set(observed) != set(by_eval_identity):
        raise VisibleCodeRepairAggregateError("repair evaluation coverage is incomplete")

    repaired_correct = sum(bool(row.get("correct")) for row in observed.values())
    source_total = int(build_report["source_total"])
    source_correct = int(build_report["source_selected_correct"])
    if source_total - source_correct != len(repair_rows):
        raise VisibleCodeRepairAggregateError("source failures differ from repair bank")
    final_correct = source_correct + repaired_correct
    return {
        "schema": SCHEMA,
        "status": "complete",
        "source_total": source_total,
        "source_selected_correct": source_correct,
        "repair_rows": len(repair_rows),
        "repair_correct": repaired_correct,
        "repair_accuracy": repaired_correct / len(repair_rows),
        "final_correct": final_correct,
        "final_accuracy": final_correct / source_total,
        "gain": repaired_correct,
        "evaluation_reports": report_receipts,
        "results": [
            {
                "original_identity_sha256": str(
                    by_eval_identity[identity]["original_identity_sha256"]
                ),
                "repair_identity_sha256": identity,
                "correct": bool(result.get("correct")),
                "completion": str(result.get("completion") or ""),
                "execution": result.get("execution"),
            }
            for identity, result in sorted(observed.items())
        ],
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise VisibleCodeRepairAggregateError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-bank", type=Path, required=True)
    parser.add_argument("--build-report", type=Path, required=True)
    parser.add_argument("--eval", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_report = _json(args.build_report)
    if build_report.get("output_sha256") != _sha256(args.repair_bank):
        raise VisibleCodeRepairAggregateError("repair bank hash differs")
    report = aggregate(
        _rows(args.repair_bank),
        build_report,
        [_json(path) for path in args.eval],
        eval_paths=args.eval,
    )
    report.update(
        {
            "repair_bank": str(args.repair_bank.resolve()),
            "repair_bank_sha256": _sha256(args.repair_bank),
            "build_report": str(args.build_report.resolve()),
            "build_report_sha256": _sha256(args.build_report),
        }
    )
    _atomic_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
