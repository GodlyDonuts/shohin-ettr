#!/usr/bin/env python3
"""Apply the frozen KR2 development gate to stage-specific owner outputs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from build_idr1_revision_data import _atomic_json
from build_vcr1_revision_data import _load_jsonl, sha256_file


TASKS = ("math500", "bbh_logic", "mbpp")
DEPTH_ONE = {"overall": 589, "math500": 223, "bbh_logic": 349, "mbpp": 17}
ABSOLUTE_FLOOR = 615
CONTROL_MARGIN = 26
RETENTION_FLOOR = 0.98
KEEP_PRECISION_FLOOR = 0.95
KEEP_COUNT_FLOOR = 64


class KR2ComparisonError(RuntimeError):
    """KR2 selected outputs or frozen thresholds differ."""


def _report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "complete":
        raise KR2ComparisonError(f"incomplete report: {path}")
    return value


def _bound_candidates(path: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    if report.get("selected_candidates_sha256") != sha256_file(path):
        raise KR2ComparisonError("selected candidates are not report-bound")
    rows = _load_jsonl(path)
    if len(rows) != 1289:
        raise KR2ComparisonError("selected development geometry differs")
    return rows


def compare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise KR2ComparisonError("refusing existing KR2 comparison")
    treatment_report = _report(args.treatment_report)
    direct_report = _report(args.direct_report)
    treatment = _bound_candidates(args.treatment_candidates, treatment_report)
    direct = _bound_candidates(args.direct_candidates, direct_report)
    depth_one = _load_jsonl(args.depth_one_candidates)
    if len(depth_one) != 1289:
        raise KR2ComparisonError("depth-one development geometry differs")
    treatment_by_id = {str(row["identity_sha256"]): row for row in treatment}
    direct_by_id = {str(row["identity_sha256"]): row for row in direct}
    depth_one_by_id = {str(row["identity_sha256"]): row for row in depth_one}
    if not (
        len(treatment_by_id)
        == len(direct_by_id)
        == len(depth_one_by_id)
        == 1289
        and set(treatment_by_id) == set(direct_by_id) == set(depth_one_by_id)
    ):
        raise KR2ComparisonError("KR2 identity coverage differs")

    transitions: dict[str, Counter[str]] = {
        "overall": Counter(),
        **{task: Counter() for task in TASKS},
    }
    for identity, before in depth_one_by_id.items():
        treatment_row = treatment_by_id[identity]
        direct_row = direct_by_id[identity]
        task = str(before.get("task"))
        if task not in TASKS or {
            treatment_row.get("task"),
            direct_row.get("task"),
        } != {task}:
            raise KR2ComparisonError("KR2 task binding differs")
        for group in ("overall", task):
            transitions[group]["depth_one_correct"] += int(bool(before.get("correct")))
            transitions[group]["treatment_correct"] += int(bool(treatment_row.get("correct")))
            transitions[group]["direct_correct"] += int(bool(direct_row.get("correct")))
            transitions[group]["preserved_correct"] += int(
                bool(before.get("correct")) and bool(treatment_row.get("correct"))
            )
            transitions[group]["broken_correct"] += int(
                bool(before.get("correct")) and not bool(treatment_row.get("correct"))
            )
            transitions[group]["repaired_error"] += int(
                not bool(before.get("correct")) and bool(treatment_row.get("correct"))
            )

    metrics: dict[str, dict[str, Any]] = {}
    for group, values in transitions.items():
        before = values["depth_one_correct"]
        metrics[group] = {
            **dict(values),
            "treatment_delta_from_depth_one": values["treatment_correct"] - before,
            "direct_delta_from_depth_one": values["direct_correct"] - before,
            "treatment_margin_over_direct": values["treatment_correct"]
            - values["direct_correct"],
            "depth_one_correct_retention": (
                values["preserved_correct"] / before if before else 0.0
            ),
        }

    keep_count = int(treatment_report.get("actions", {}).get("keep", 0))
    keep_precision = float(treatment_report.get("keep_precision", 0.0))
    gates = {
        "depth_one_receipt_matches": all(
            metrics[group]["depth_one_correct"] == expected
            for group, expected in DEPTH_ONE.items()
        ),
        "absolute_capability": metrics["overall"]["treatment_correct"]
        >= ABSOLUTE_FLOOR,
        "matched_control_margin": metrics["overall"]["treatment_margin_over_direct"]
        >= CONTROL_MARGIN,
        "all_domains_nonnegative": all(
            metrics[task]["treatment_delta_from_depth_one"] >= 0 for task in TASKS
        ),
        "conservative_retention": metrics["overall"]["depth_one_correct_retention"]
        >= RETENTION_FLOOR,
        "keep_precision": keep_precision >= KEEP_PRECISION_FLOOR,
        "keep_action_nontrivial": keep_count >= KEEP_COUNT_FLOOR,
    }
    report = {
        "schema": "shohin-kr2-stage-owner-comparison-v1",
        "status": "complete",
        "frozen_thresholds": {
            "absolute_correct": ABSOLUTE_FLOOR,
            "matched_control_margin": CONTROL_MARGIN,
            "domain_delta_floor": 0,
            "retention": RETENTION_FLOOR,
            "keep_precision": KEEP_PRECISION_FLOOR,
            "keep_count": KEEP_COUNT_FLOOR,
        },
        "treatment_report": {
            "path": str(args.treatment_report.resolve()),
            "sha256": sha256_file(args.treatment_report),
        },
        "direct_report": {
            "path": str(args.direct_report.resolve()),
            "sha256": sha256_file(args.direct_report),
        },
        "depth_one_candidates": {
            "path": str(args.depth_one_candidates.resolve()),
            "sha256": sha256_file(args.depth_one_candidates),
        },
        "keep_count": keep_count,
        "keep_precision": keep_precision,
        "metrics": metrics,
        "gates": gates,
        "gate_pass": all(gates.values()),
        "failure_policy": "close KR2 without prompt, rank, update, seed, or threshold rescue",
        "success_policy": "unlock exactly one source-disjoint holdout evaluation",
    }
    _atomic_json(args.output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatment-candidates", type=Path, required=True)
    parser.add_argument("--treatment-report", type=Path, required=True)
    parser.add_argument("--direct-candidates", type=Path, required=True)
    parser.add_argument("--direct-report", type=Path, required=True)
    parser.add_argument("--depth-one-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    report = compare(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
