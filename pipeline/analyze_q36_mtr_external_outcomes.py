#!/usr/bin/env python3
"""Turn a Q36 external outcome matrix into architecture-selection evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from score_q36_mtr_external import ARMS, REPORT_SCHEMA, TASKS, _mcnemar_exact

SCHEMA = "shohin-q36-mtr-external-outcome-analysis-v1"
FOREST_SCHEMA = "shohin-q36-mtr-external-forest-score-v1"
CONSENSUS_SCHEMA = "shohin-q36-mtr-external-consensus-score-v1"


class Q36MTRExternalOutcomeAnalysisError(RuntimeError):
    """The detailed external result or a companion result differs."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Q36MTRExternalOutcomeAnalysisError(
            "external result unreadable"
        ) from error
    if not isinstance(value, dict):
        raise Q36MTRExternalOutcomeAnalysisError("external result differs")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRExternalOutcomeAnalysisError("external analysis output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_detailed(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("rows")
    outcomes = report.get("outcomes")
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("status") != "complete"
        or not isinstance(rows, int)
        or isinstance(rows, bool)
        or rows <= 0
        or not isinstance(outcomes, list)
        or len(outcomes) != rows
        or not isinstance(report.get("arms"), dict)
    ):
        raise Q36MTRExternalOutcomeAnalysisError("detailed external result differs")
    identities: list[str] = []
    task_rows = {task: 0 for task in TASKS}
    correct = {arm: 0 for arm in ARMS}
    oracle = 0
    for row in outcomes:
        identity = row.get("identity_sha256") if isinstance(row, dict) else None
        task = row.get("task") if isinstance(row, dict) else None
        values = row.get("correct") if isinstance(row, dict) else None
        if (
            not isinstance(identity, str)
            or len(identity) != 64
            or task not in TASKS
            or not isinstance(values, dict)
            or set(values) != set(ARMS)
            or any(not isinstance(values[arm], bool) for arm in ARMS)
        ):
            raise Q36MTRExternalOutcomeAnalysisError("external outcome row differs")
        identities.append(identity)
        task_rows[task] += 1
        oracle += int(any(values.values()))
        for arm in ARMS:
            correct[arm] += int(values[arm])
    if identities != sorted(set(identities)):
        raise Q36MTRExternalOutcomeAnalysisError("external outcome order differs")
    for arm in ARMS:
        arm_report = report["arms"].get(arm)
        if (
            not isinstance(arm_report, dict)
            or arm_report.get("correct") != correct[arm]
            or arm_report.get("total") != rows
            or not isinstance(arm_report.get("domains"), dict)
        ):
            raise Q36MTRExternalOutcomeAnalysisError("external arm aggregate differs")
        for task in TASKS:
            domain = arm_report["domains"].get(task)
            observed = sum(
                int(row["correct"][arm]) for row in outcomes if row["task"] == task
            )
            if not isinstance(domain, dict) or domain != {
                "correct": observed,
                "total": task_rows[task],
            }:
                raise Q36MTRExternalOutcomeAnalysisError(
                    "external domain aggregate differs"
                )
    if report.get("all_arm_oracle_correct") != oracle:
        raise Q36MTRExternalOutcomeAnalysisError("external oracle aggregate differs")
    return outcomes


def _pair(left: str, right: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    left_only = sum(row["correct"][left] and not row["correct"][right] for row in rows)
    right_only = sum(row["correct"][right] and not row["correct"][left] for row in rows)
    both = sum(row["correct"][left] and row["correct"][right] for row in rows)
    neither = len(rows) - both - left_only - right_only
    return {
        "both_correct": both,
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "both_wrong": neither,
        "net_left_minus_right": left_only - right_only,
        "oracle_correct": both + left_only + right_only,
        "mcnemar_exact_two_sided_p": _mcnemar_exact(left_only, right_only),
    }


def _candidate_results(
    detailed: dict[str, Any],
    forest: dict[str, Any] | None,
    consensus: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    candidates = [
        {
            "architecture": f"arm:{arm}",
            "correct": detailed["arms"][arm]["correct"],
            "total": detailed["rows"],
        }
        for arm in ARMS
    ]
    if forest is not None:
        target = forest.get("target")
        if (
            forest.get("schema") != FOREST_SCHEMA
            or forest.get("status") != "complete"
            or forest.get("split") != detailed.get("split")
            or forest.get("rows") != detailed.get("rows")
            or not isinstance(target, dict)
            or not isinstance(target.get("correct"), int)
        ):
            raise Q36MTRExternalOutcomeAnalysisError("external forest result differs")
        candidates.append(
            {
                "architecture": "learned_forest",
                "correct": target["correct"],
                "total": detailed["rows"],
            }
        )
    if consensus is not None:
        rules = consensus.get("rules")
        if (
            consensus.get("schema") != CONSENSUS_SCHEMA
            or consensus.get("status") != "complete"
            or consensus.get("split") != detailed.get("split")
            or consensus.get("rows") != detailed.get("rows")
            or not isinstance(rules, dict)
        ):
            raise Q36MTRExternalOutcomeAnalysisError(
                "external consensus result differs"
            )
        for rule, result in sorted(rules.items()):
            if not isinstance(result, dict) or not isinstance(
                result.get("correct"), int
            ):
                raise Q36MTRExternalOutcomeAnalysisError(
                    "external consensus rule differs"
                )
            candidates.append(
                {
                    "architecture": f"consensus:{rule}",
                    "correct": result["correct"],
                    "total": detailed["rows"],
                }
            )
    for candidate in candidates:
        candidate["accuracy"] = candidate["correct"] / candidate["total"]
    return sorted(candidates, key=lambda row: (-row["correct"], row["architecture"]))


def _failure_geometry(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    patterns: dict[str, dict[str, Any]] = {}
    exclusive = {
        arm: {"rows": 0, "by_task": {task: 0 for task in TASKS}} for arm in ARMS
    }
    all_wrong = {"rows": 0, "by_task": {task: 0 for task in TASKS}}
    unanimous_correct = {"rows": 0, "by_task": {task: 0 for task in TASKS}}
    disagreement = {"rows": 0, "by_task": {task: 0 for task in TASKS}}
    for row in outcomes:
        task = row["task"]
        values = row["correct"]
        bits = "".join("1" if values[arm] else "0" for arm in ARMS)
        pattern = patterns.setdefault(
            bits,
            {"rows": 0, "by_task": {candidate: 0 for candidate in TASKS}},
        )
        pattern["rows"] += 1
        pattern["by_task"][task] += 1
        correct_arms = [arm for arm in ARMS if values[arm]]
        if not correct_arms:
            all_wrong["rows"] += 1
            all_wrong["by_task"][task] += 1
        elif len(correct_arms) == len(ARMS):
            unanimous_correct["rows"] += 1
            unanimous_correct["by_task"][task] += 1
        else:
            disagreement["rows"] += 1
            disagreement["by_task"][task] += 1
        if len(correct_arms) == 1:
            arm = correct_arms[0]
            exclusive[arm]["rows"] += 1
            exclusive[arm]["by_task"][task] += 1

    unchanged_correct = sum(row["correct"]["unchanged"] for row in outcomes)
    retention = {}
    for arm in ARMS:
        retained = sum(
            row["correct"]["unchanged"] and row["correct"][arm] for row in outcomes
        )
        lost = unchanged_correct - retained
        repaired = sum(
            not row["correct"]["unchanged"] and row["correct"][arm] for row in outcomes
        )
        retention[arm] = {
            "unchanged_correct_retained": retained,
            "unchanged_correct_lost": lost,
            "unchanged_wrong_repaired": repaired,
            "net_gain_over_unchanged": repaired - lost,
            "unchanged_correct_retention_rate": (
                retained / unchanged_correct if unchanged_correct else None
            ),
        }
    return {
        "arm_order": list(ARMS),
        "correctness_patterns": dict(sorted(patterns.items())),
        "all_fixed_arms_wrong": all_wrong,
        "all_fixed_arms_correct": unanimous_correct,
        "fixed_arm_disagreement": disagreement,
        "exclusive_correct_by_arm": exclusive,
        "retention_vs_unchanged": retention,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    detailed = _load(args.detailed)
    outcomes = _validate_detailed(detailed)
    pairwise = {}
    per_task = {}
    for left_index, left in enumerate(ARMS):
        for right in ARMS[left_index + 1 :]:
            name = f"{left}__vs__{right}"
            pairwise[name] = _pair(left, right, outcomes)
            per_task[name] = {
                task: _pair(
                    left, right, [row for row in outcomes if row["task"] == task]
                )
                for task in TASKS
            }
    forest = _load(args.forest) if args.forest else None
    consensus = _load(args.consensus) if args.consensus else None
    candidates = _candidate_results(detailed, forest, consensus)
    best_fixed = max(
        ARMS,
        key=lambda arm: (detailed["arms"][arm]["correct"], -ARMS.index(arm)),
    )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "split": detailed["split"],
        "rows": detailed["rows"],
        "best_fixed_arm": best_fixed,
        "best_fixed_correct": detailed["arms"][best_fixed]["correct"],
        "selected_architecture": candidates[0]["architecture"],
        "selected_correct": candidates[0]["correct"],
        "candidate_ranking": candidates,
        "pairwise": pairwise,
        "pairwise_by_task": per_task,
        "all_arm_oracle_correct": detailed["all_arm_oracle_correct"],
        "oracle_gap_over_best_fixed": detailed["all_arm_oracle_correct"]
        - detailed["arms"][best_fixed]["correct"],
        "failure_geometry": _failure_geometry(outcomes),
    }
    _atomic_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detailed", type=Path, required=True)
    parser.add_argument("--forest", type=Path)
    parser.add_argument("--consensus", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "selected_architecture": report["selected_architecture"],
                "correct": report["selected_correct"],
                "oracle_gap": report["oracle_gap_over_best_fixed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
