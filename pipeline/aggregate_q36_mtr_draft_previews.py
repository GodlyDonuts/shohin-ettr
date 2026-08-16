#!/usr/bin/env python3
"""Aggregate disjoint Q36 owner-draft preview reports for engineering feedback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

INPUT_SCHEMA = "shohin-q36-mtr-draft-preview-v1"
OUTPUT_SCHEMA = "shohin-q36-mtr-draft-preview-aggregate-v1"
TASKS = ("math500", "bbh_logic", "mbpp")


class Q36MTRDraftPreviewAggregateError(RuntimeError):
    """Raised when preview shards cannot form one exact aggregate."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Q36MTRDraftPreviewAggregateError(
            f"unreadable preview report: {path}"
        ) from error
    if not isinstance(value, dict):
        raise Q36MTRDraftPreviewAggregateError("preview report is not an object")
    return value


def aggregate(paths: list[Path], *, label: str) -> dict[str, Any]:
    if not label or not paths:
        raise Q36MTRDraftPreviewAggregateError("aggregate label or reports are absent")
    resolved = [path.resolve(strict=True) for path in paths]
    if len(resolved) != len(set(resolved)):
        raise Q36MTRDraftPreviewAggregateError("duplicate preview report path")

    outcomes: list[dict[str, Any]] = []
    identities: set[str] = set()
    receipts = []
    for path in resolved:
        report = _load(path)
        rows = report.get("outcomes")
        if (
            report.get("schema") != INPUT_SCHEMA
            or report.get("status") != "complete"
            or report.get("split") != "development"
            or report.get("interpretation")
            != "exploratory_model_owned_draft_only_not_matched_gate"
            or not isinstance(rows, list)
            or report.get("rows") != len(rows)
            or not isinstance(report.get("candidates_sha256"), str)
            or len(report["candidates_sha256"]) != 64
        ):
            raise Q36MTRDraftPreviewAggregateError("preview report contract differs")
        for row in rows:
            if not isinstance(row, dict):
                raise Q36MTRDraftPreviewAggregateError("preview outcome differs")
            identity = row.get("identity_sha256")
            if (
                not isinstance(identity, str)
                or len(identity) != 64
                or identity in identities
                or row.get("task") not in TASKS
                or not isinstance(row.get("correct"), bool)
                or not isinstance(row.get("explicit_final_answer"), bool)
                or not isinstance(row.get("max_token_exhausted"), bool)
            ):
                raise Q36MTRDraftPreviewAggregateError(
                    "preview identity or outcome differs"
                )
            identities.add(identity)
            outcomes.append(row)
        receipts.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "candidates_sha256": report["candidates_sha256"],
                "rows": len(rows),
            }
        )

    metrics: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        selected = [row for row in outcomes if row["task"] == task]
        correct = sum(row["correct"] for row in selected)
        metrics[task] = {
            "rows": len(selected),
            "correct": correct,
            "accuracy": correct / len(selected) if selected else None,
            "max_token_exhausted": sum(row["max_token_exhausted"] for row in selected),
            "explicit_final_answers": sum(
                row["explicit_final_answer"] for row in selected
            ),
        }

    total = len(outcomes)
    correct = sum(row["correct"] for row in outcomes)
    exhausted = sum(row["max_token_exhausted"] for row in outcomes)
    nonexhausted = total - exhausted
    status_counts = Counter(
        (
            "exhausted" if row["max_token_exhausted"] else "nonexhausted",
            "correct" if row["correct"] else "incorrect",
        )
        for row in outcomes
    )
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "complete",
        "interpretation": "engineering_owner_draft_preview_not_matched_gate",
        "label": label,
        "report_count": len(receipts),
        "rows": total,
        "unique_identities": len(identities),
        "correct": correct,
        "accuracy": correct / total,
        "explicit_final_answers": sum(row["explicit_final_answer"] for row in outcomes),
        "max_token_exhausted": exhausted,
        "domains": metrics,
        "completion_status": {
            "nonexhausted_rows": nonexhausted,
            "nonexhausted_correct": status_counts[("nonexhausted", "correct")],
            "nonexhausted_accuracy": (
                status_counts[("nonexhausted", "correct")] / nonexhausted
                if nonexhausted
                else None
            ),
            "exhausted_rows": exhausted,
            "exhausted_correct": status_counts[("exhausted", "correct")],
            "exhausted_accuracy": (
                status_counts[("exhausted", "correct")] / exhausted
                if exhausted
                else None
            ),
        },
        "inputs": receipts,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRDraftPreviewAggregateError("aggregate output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = aggregate(args.report, label=args.label)
    _atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
