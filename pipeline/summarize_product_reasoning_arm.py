#!/usr/bin/env python3
"""Create one hash-bound product-reasoning benchmark receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-product-reasoning-arm-summary-v1"
TASKS_BY_DOMAIN = {
    "grade_school_math": ("gsm8k",),
    "competition_math": ("math500",),
    "code": ("humaneval", "mbpp"),
    "science": ("gpqa",),
    "logic": ("bbh_logic",),
}
MAIN_TASKS = tuple(task for tasks in TASKS_BY_DOMAIN.values() for task in tasks)
TASKS = (*MAIN_TASKS, "aime")


class ProductArmSummaryError(RuntimeError):
    """The product arm is incomplete or internally inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProductArmSummaryError(f"missing artifact: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ProductArmSummaryError(f"artifact is not an object: {path}")
    return payload


def summarize_arm(
    *,
    name: str,
    eval_prefix: Path,
    training_report: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    training = _load_json(training_report)
    if training.get("status") != "complete":
        raise ProductArmSummaryError("training report is not complete")
    if not checkpoint.is_file():
        raise ProductArmSummaryError(f"missing checkpoint: {checkpoint}")

    reports: dict[str, dict[str, Any]] = {}
    files: dict[str, dict[str, str]] = {}
    for task in TASKS:
        path = Path(f"{eval_prefix}_{task}.json")
        report = _load_json(path)
        if report.get("status") != "complete" or report.get("task") != task:
            raise ProductArmSummaryError(f"invalid evaluation identity: {path}")
        correct, total = report.get("correct"), report.get("total")
        if (
            not isinstance(correct, int)
            or not isinstance(total, int)
            or total <= 0
            or not 0 <= correct <= total
        ):
            raise ProductArmSummaryError(f"invalid score in {path}")
        reports[task] = report
        files[task] = {"path": str(path.resolve()), "sha256": _sha256(path)}

    domains: dict[str, dict[str, Any]] = {}
    for domain, tasks in TASKS_BY_DOMAIN.items():
        correct = sum(reports[task]["correct"] for task in tasks)
        total = sum(reports[task]["total"] for task in tasks)
        domains[domain] = {
            "accuracy": correct / total,
            "correct": correct,
            "tasks": list(tasks),
            "total": total,
        }
    return {
        "arm": name,
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": _sha256(checkpoint),
        },
        "evaluation_files": files,
        "scores": {
            "aime": {
                "accuracy": reports["aime"]["correct"] / reports["aime"]["total"],
                "correct": reports["aime"]["correct"],
                "total": reports["aime"]["total"],
            },
            "domains": domains,
            "macro_accuracy": sum(row["accuracy"] for row in domains.values())
            / len(domains),
            "solved": sum(reports[task]["correct"] for task in MAIN_TASKS),
            "total": sum(reports[task]["total"] for task in MAIN_TASKS),
        },
        "schema": SCHEMA,
        "status": "complete",
        "training": training,
        "training_report": {
            "path": str(training_report.resolve()),
            "sha256": _sha256(training_report),
        },
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--eval-prefix", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = summarize_arm(
        name=args.name,
        eval_prefix=args.eval_prefix,
        training_report=args.training_report,
        checkpoint=args.checkpoint,
    )
    _atomic_write(args.output, report)
    print(json.dumps(report["scores"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
