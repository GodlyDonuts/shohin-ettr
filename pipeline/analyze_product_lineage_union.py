#!/usr/bin/env python3
"""Measure the static whole-lineage oracle ceiling of two product arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-product-lineage-union-v1"
TASKS_BY_DOMAIN = {
    "grade_school_math": ("gsm8k",),
    "competition_math": ("math500",),
    "code": ("humaneval", "mbpp"),
    "science": ("gpqa",),
    "logic": ("bbh_logic",),
}
MAIN_TASKS = tuple(task for tasks in TASKS_BY_DOMAIN.values() for task in tasks)
TASKS = (*MAIN_TASKS, "aime")
COMPARABILITY_FIELDS = (
    "task",
    "data_sha256",
    "selection_sha256",
    "generation_mode",
    "generation_seed",
    "max_new_tokens",
    "generation_stop_token_ids",
    "subset_seed",
    "effective_enable_thinking",
    "total",
)


class LineageUnionError(RuntimeError):
    """The two arms are incomplete or not instance-matched."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, task: str) -> dict[str, Any]:
    if not path.is_file():
        raise LineageUnionError(f"missing report: {path}")
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("status") != "complete" or report.get("task") != task:
        raise LineageUnionError(f"invalid report identity: {path}")
    if not isinstance(report.get("results"), list):
        raise LineageUnionError(f"report has no per-instance results: {path}")
    return report


def analyze_union(
    *, left_name: str, left_prefix: Path, right_name: str, right_prefix: Path
) -> dict[str, Any]:
    task_receipts: dict[str, dict[str, Any]] = {}
    union_scores: dict[str, tuple[int, int]] = {}
    for task in TASKS:
        left_path = Path(f"{left_prefix}_{task}.json")
        right_path = Path(f"{right_prefix}_{task}.json")
        left = _load(left_path, task)
        right = _load(right_path, task)
        for field in COMPARABILITY_FIELDS:
            if left.get(field) != right.get(field):
                raise LineageUnionError(f"unmatched {task} field: {field}")
        left_rows = {row.get("identity_sha256"): row for row in left["results"]}
        right_rows = {row.get("identity_sha256"): row for row in right["results"]}
        if None in left_rows or None in right_rows or left_rows.keys() != right_rows.keys():
            raise LineageUnionError(f"unmatched result identities: {task}")
        both = left_only = right_only = neither = 0
        for identity in left_rows:
            left_correct = bool(left_rows[identity].get("correct"))
            right_correct = bool(right_rows[identity].get("correct"))
            both += left_correct and right_correct
            left_only += left_correct and not right_correct
            right_only += right_correct and not left_correct
            neither += not left_correct and not right_correct
        total = len(left_rows)
        union_correct = both + left_only + right_only
        union_scores[task] = (union_correct, total)
        task_receipts[task] = {
            "both_correct": both,
            "left_correct": int(left.get("correct", 0)),
            "left_only": left_only,
            "neither_correct": neither,
            "right_correct": int(right.get("correct", 0)),
            "right_only": right_only,
            "total": total,
            "union_accuracy": union_correct / total,
            "union_correct": union_correct,
        }

    domains: dict[str, dict[str, Any]] = {}
    for domain, tasks in TASKS_BY_DOMAIN.items():
        correct = sum(union_scores[task][0] for task in tasks)
        total = sum(union_scores[task][1] for task in tasks)
        domains[domain] = {
            "accuracy": correct / total,
            "correct": correct,
            "tasks": list(tasks),
            "total": total,
        }
    return {
        "arms": {"left": left_name, "right": right_name},
        "files": {
            task: {
                "left": {
                    "path": str(Path(f"{left_prefix}_{task}.json").resolve()),
                    "sha256": _sha256(Path(f"{left_prefix}_{task}.json")),
                },
                "right": {
                    "path": str(Path(f"{right_prefix}_{task}.json").resolve()),
                    "sha256": _sha256(Path(f"{right_prefix}_{task}.json")),
                },
            }
            for task in TASKS
        },
        "schema": SCHEMA,
        "status": "complete",
        "tasks": task_receipts,
        "union": {
            "aime": {
                "accuracy": union_scores["aime"][0] / union_scores["aime"][1],
                "correct": union_scores["aime"][0],
                "total": union_scores["aime"][1],
            },
            "domains": domains,
            "macro_accuracy": sum(row["accuracy"] for row in domains.values())
            / len(domains),
            "solved": sum(union_scores[task][0] for task in MAIN_TASKS),
            "total": sum(union_scores[task][1] for task in MAIN_TASKS),
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
    parser.add_argument("--left-name", required=True)
    parser.add_argument("--left-prefix", required=True, type=Path)
    parser.add_argument("--right-name", required=True)
    parser.add_argument("--right-prefix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze_union(
        left_name=args.left_name,
        left_prefix=args.left_prefix,
        right_name=args.right_name,
        right_prefix=args.right_prefix,
    )
    _atomic_write(args.output, report)
    print(json.dumps(report["union"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
