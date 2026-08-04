#!/usr/bin/env python3
"""Non-destructively rescore a completed product evaluation report."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from hf_product_reasoning_eval import TASKS


SCHEMA = "shohin-product-eval-rescore-v1"


class ProductEvalRescoreError(RuntimeError):
    """An evaluation report cannot be rescored exactly."""


def rescore_report(payload: dict[str, Any]) -> dict[str, Any]:
    task = str(payload.get("task") or "")
    if task not in TASKS or TASKS[task].get("kind") != "answer":
        raise ProductEvalRescoreError("evaluation task is not answer-scored")
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ProductEvalRescoreError("evaluation results are missing")
    matcher = TASKS[task]["match"]
    transitions: Counter[str] = Counter()
    rescored_results: list[dict[str, Any]] = []
    correct = 0
    for row in results:
        if "gold" not in row or "prediction" not in row:
            raise ProductEvalRescoreError("result scoring fields are incomplete")
        previous = bool(row.get("correct"))
        updated = bool(matcher(row.get("prediction"), row.get("gold")))
        rescored = dict(row)
        rescored["correct"] = updated
        rescored_results.append(rescored)
        correct += int(updated)
        transitions[f"{int(previous)}->{int(updated)}"] += 1
    total = len(rescored_results)
    if int(payload.get("total", total)) != total:
        raise ProductEvalRescoreError("evaluation total differs from results")
    report = dict(payload)
    report.update(
        {
            "rescore_schema": SCHEMA,
            "original_correct": int(payload.get("correct", 0)),
            "original_accuracy": float(payload.get("accuracy", 0.0)),
            "label_transitions": dict(sorted(transitions.items())),
            "correct": correct,
            "accuracy": correct / total,
            "results": rescored_results,
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ProductEvalRescoreError(f"refusing existing output: {args.output}")
    source_bytes = args.report.read_bytes()
    payload = json.loads(source_bytes)
    report = rescore_report(payload)
    report["source_report"] = str(args.report.resolve())
    report["source_report_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "original_correct": report["original_correct"],
                "correct": report["correct"],
                "total": report["total"],
                "accuracy": report["accuracy"],
                "label_transitions": report["label_transitions"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
