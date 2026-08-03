#!/usr/bin/env python3
"""Measure whether frozen product experts have routable outcome complementarity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-product-outcome-pair-analysis-v1"


class OutcomePairError(RuntimeError):
    """Paired expert reports violate the comparison contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_report(path: Path) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise OutcomePairError(f"cannot read report {path}: {error}") from error
    if not isinstance(report, dict) or report.get("status") != "complete":
        raise OutcomePairError(f"report is not complete: {path}")
    results = report.get("results")
    if not isinstance(results, list) or not results:
        raise OutcomePairError(f"report has no results: {path}")
    return report


def analyze_pair(
    baseline: dict[str, Any], dense: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for field in ("task", "data_sha256", "model_revision", "selection_sha256"):
        if baseline.get(field) != dense.get(field):
            raise OutcomePairError(f"paired report field differs: {field}")

    baseline_results = baseline["results"]
    dense_results = dense["results"]
    dense_by_id = {row.get("identity_sha256"): row for row in dense_results}
    if None in dense_by_id or len(dense_by_id) != len(dense_results):
        raise OutcomePairError("dense report identities are missing or duplicated")
    baseline_ids = [row.get("identity_sha256") for row in baseline_results]
    if None in baseline_ids or len(set(baseline_ids)) != len(baseline_ids):
        raise OutcomePairError("baseline report identities are missing or duplicated")
    if set(baseline_ids) != set(dense_by_id):
        raise OutcomePairError("paired report identity sets differ")

    counts = {"both_correct": 0, "baseline_only": 0, "dense_only": 0, "both_wrong": 0}
    labels: list[dict[str, Any]] = []
    for baseline_row in baseline_results:
        identity = baseline_row["identity_sha256"]
        dense_row = dense_by_id[identity]
        for field in ("question", "gold"):
            if baseline_row.get(field) != dense_row.get(field):
                raise OutcomePairError(f"paired result field differs for {identity}: {field}")
        baseline_correct = bool(baseline_row.get("correct"))
        dense_correct = bool(dense_row.get("correct"))
        if baseline_correct and dense_correct:
            outcome = "both_correct"
        elif baseline_correct:
            outcome = "baseline_only"
        elif dense_correct:
            outcome = "dense_only"
        else:
            outcome = "both_wrong"
        counts[outcome] += 1
        labels.append(
            {
                "identity_sha256": identity,
                "question": baseline_row["question"],
                "gold": baseline_row.get("gold"),
                "outcome": outcome,
            }
        )

    total = len(labels)
    baseline_correct = counts["both_correct"] + counts["baseline_only"]
    dense_correct = counts["both_correct"] + counts["dense_only"]
    oracle_correct = total - counts["both_wrong"]
    better_correct = max(baseline_correct, dense_correct)
    summary = {
        "task": baseline["task"],
        "total": total,
        "counts": counts,
        "baseline_accuracy": baseline_correct / total,
        "dense_accuracy": dense_correct / total,
        "paired_oracle_accuracy": oracle_correct / total,
        "oracle_lift_over_best": (oracle_correct - better_correct) / total,
        "baseline_only_rate": counts["baseline_only"] / total,
        "dense_only_rate": counts["dense_only"] / total,
    }
    return summary, labels


def analyze_reports(
    baseline_paths: list[Path],
    dense_paths: list[Path],
    *,
    minimum_oracle_lift: float,
    minimum_exclusive_rate: float,
) -> dict[str, Any]:
    if len(baseline_paths) != len(dense_paths) or not baseline_paths:
        raise OutcomePairError("baseline and dense report counts must match and be nonzero")
    tasks: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    inputs: list[dict[str, str]] = []
    seen_tasks: set[str] = set()
    for baseline_path, dense_path in zip(baseline_paths, dense_paths, strict=True):
        baseline = _load_report(baseline_path)
        dense = _load_report(dense_path)
        summary, task_labels = analyze_pair(baseline, dense)
        if summary["task"] in seen_tasks:
            raise OutcomePairError(f"duplicate task: {summary['task']}")
        seen_tasks.add(summary["task"])
        tasks.append(summary)
        labels.extend({"task": summary["task"], **row} for row in task_labels)
        inputs.append(
            {
                "baseline": str(baseline_path.resolve()),
                "baseline_sha256": _sha256(baseline_path),
                "dense": str(dense_path.resolve()),
                "dense_sha256": _sha256(dense_path),
            }
        )

    total = sum(task["total"] for task in tasks)
    counts = {
        outcome: sum(task["counts"][outcome] for task in tasks)
        for outcome in ("both_correct", "baseline_only", "dense_only", "both_wrong")
    }
    baseline_correct = counts["both_correct"] + counts["baseline_only"]
    dense_correct = counts["both_correct"] + counts["dense_only"]
    oracle_correct = total - counts["both_wrong"]
    aggregate = {
        "total": total,
        "counts": counts,
        "baseline_accuracy": baseline_correct / total,
        "dense_accuracy": dense_correct / total,
        "paired_oracle_accuracy": oracle_correct / total,
        "oracle_lift_over_best": (oracle_correct - max(baseline_correct, dense_correct)) / total,
        "baseline_only_rate": counts["baseline_only"] / total,
        "dense_only_rate": counts["dense_only"] / total,
    }
    proceed = (
        aggregate["oracle_lift_over_best"] >= minimum_oracle_lift
        and aggregate["baseline_only_rate"] >= minimum_exclusive_rate
        and aggregate["dense_only_rate"] >= minimum_exclusive_rate
    )
    return {
        "schema": SCHEMA,
        "status": "complete",
        "inputs": inputs,
        "thresholds": {
            "minimum_oracle_lift": minimum_oracle_lift,
            "minimum_exclusive_rate_each_arm": minimum_exclusive_rate,
        },
        "tasks": tasks,
        "aggregate": aggregate,
        "decision": "train-outcome-gate" if proceed else "close-outcome-routing",
        "labels": labels,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise OutcomePairError(f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--dense", type=Path, action="append", required=True)
    parser.add_argument("--minimum-oracle-lift", type=float, default=0.05)
    parser.add_argument("--minimum-exclusive-rate", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.minimum_oracle_lift <= 1:
        parser.error("minimum oracle lift must be in [0, 1]")
    if not 0 <= args.minimum_exclusive_rate <= 1:
        parser.error("minimum exclusive rate must be in [0, 1]")
    report = analyze_reports(
        args.baseline,
        args.dense,
        minimum_oracle_lift=args.minimum_oracle_lift,
        minimum_exclusive_rate=args.minimum_exclusive_rate,
    )
    _atomic_json(args.output, report)
    print(json.dumps({"aggregate": report["aggregate"], "decision": report["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
