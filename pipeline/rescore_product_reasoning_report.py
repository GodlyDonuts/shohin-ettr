#!/usr/bin/env python3
"""Rescore saved product completions without regenerating model output."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from typing import Any

TRAIN_DIRECTORY = Path(__file__).resolve().parents[1] / "train"
if str(TRAIN_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIRECTORY))

from hf_product_reasoning_eval import TASKS  # noqa: E402


SCHEMA = "shohin-hf-product-reasoning-eval-v3-rescore"


class ProductRescoreError(RuntimeError):
    """Raised when a saved report cannot be rescored exactly."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rescore_report(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProductRescoreError(f"missing report: {path}")
    with path.open("r", encoding="utf-8") as handle:
        original = json.load(handle)
    if original.get("status") != "complete":
        raise ProductRescoreError("only complete reports may be rescored")
    task_name = original.get("task")
    if task_name not in TASKS:
        raise ProductRescoreError(f"unknown task: {task_name!r}")
    rows = original.get("results")
    if not isinstance(rows, list) or len(rows) != original.get("total"):
        raise ProductRescoreError("result rows do not match report total")

    task = TASKS[task_name]
    rescored_rows: list[dict[str, Any]] = []
    false_to_true = 0
    true_to_false = 0
    for row in rows:
        rescored = dict(row)
        old_prediction = row.get("prediction")
        old_correct = bool(row.get("correct"))
        if task.get("kind") == "code":
            new_prediction = old_prediction
            new_correct = old_correct
        else:
            completion = row.get("completion")
            if not isinstance(completion, str):
                raise ProductRescoreError("answer row is missing its completion")
            new_prediction = task["extract"](completion)
            new_correct = bool(task["match"](new_prediction, row.get("gold")))
        rescored["original_prediction"] = old_prediction
        rescored["original_correct"] = old_correct
        rescored["prediction"] = new_prediction
        rescored["correct"] = new_correct
        false_to_true += int(not old_correct and new_correct)
        true_to_false += int(old_correct and not new_correct)
        rescored_rows.append(rescored)

    correct = sum(bool(row["correct"]) for row in rescored_rows)
    if task_name == "math500":
        try:
            score_backend = f"math-verify-{importlib.metadata.version('math-verify')}"
        except importlib.metadata.PackageNotFoundError:
            score_backend = "normalized-exact-fallback"
    else:
        score_backend = "shohin-answer-v3"
    report = dict(original)
    report.update(
        {
            "accuracy": correct / len(rescored_rows),
            "correct": correct,
            "original_accuracy": original.get("accuracy"),
            "original_correct": original.get("correct"),
            "rescore_change_count": false_to_true + true_to_false,
            "rescore_changes": {
                "false_to_true": false_to_true,
                "true_to_false": true_to_false,
            },
            "rescore_backend": score_backend,
            "rescored_from": str(path.resolve()),
            "rescored_from_sha256": _sha256(path),
            "results": rescored_rows,
            "schema": SCHEMA,
        }
    )
    return report


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = rescore_report(args.input)
    _write_atomic(args.output, report)
    print(
        json.dumps(
            {
                "accuracy": report["accuracy"],
                "correct": report["correct"],
                "rescore_changes": report["rescore_changes"],
                "total": report["total"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
