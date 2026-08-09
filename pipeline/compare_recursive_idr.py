#!/usr/bin/env python3
"""Compare frozen depth-one and depth-two IDR1 development completions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from build_idr1_revision_data import _atomic_json
from build_vcr1_revision_data import _load_jsonl, sha256_file


TASKS = ("math500", "bbh_logic", "mbpp")
DEPTH_ONE_FLOORS = {"overall": 589, "math500": 223, "bbh_logic": 349, "mbpp": 17}
DEPTH_TWO_FLOOR = 615
RETENTION_FLOOR = 0.98


class RecursiveIDRComparisonError(RuntimeError):
    """Candidate coverage or the frozen comparison contract differs."""


def _report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "complete":
        raise RecursiveIDRComparisonError(f"incomplete report: {path}")
    return value


def compare(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise RecursiveIDRComparisonError("refusing existing comparison output")
    first_report = _report(args.depth_one_report)
    second_report = _report(args.depth_two_report)
    first = _load_jsonl(args.depth_one_candidates)
    second = _load_jsonl(args.depth_two_candidates)
    if (
        first_report.get("candidates_sha256") != sha256_file(args.depth_one_candidates)
        or second_report.get("candidates_sha256")
        != sha256_file(args.depth_two_candidates)
        or first_report.get("split") != "development"
        or second_report.get("split") != "development"
    ):
        raise RecursiveIDRComparisonError("candidate/report binding differs")
    first_by_id = {str(row.get("identity_sha256")): row for row in first}
    second_by_id = {str(row.get("identity_sha256")): row for row in second}
    if (
        len(first) != 1289
        or len(second) != 1289
        or len(first_by_id) != len(first)
        or set(first_by_id) != set(second_by_id)
    ):
        raise RecursiveIDRComparisonError("development identity coverage differs")

    transitions: dict[str, Counter[str]] = {
        "overall": Counter(),
        **{task: Counter() for task in TASKS},
    }
    for identity, before in first_by_id.items():
        after = second_by_id[identity]
        task = str(before.get("task"))
        if task not in TASKS or after.get("task") != task:
            raise RecursiveIDRComparisonError("candidate task binding differs")
        before_correct = bool(before.get("correct"))
        after_correct = bool(after.get("correct"))
        key = (
            "preserved_correct"
            if before_correct and after_correct
            else "broken_correct"
            if before_correct
            else "repaired_error"
            if after_correct
            else "persistent_error"
        )
        transitions["overall"][key] += 1
        transitions[task][key] += 1

    metrics: dict[str, dict[str, Any]] = {}
    for group, values in transitions.items():
        depth_one_correct = values["preserved_correct"] + values["broken_correct"]
        depth_two_correct = values["preserved_correct"] + values["repaired_error"]
        retention = (
            values["preserved_correct"] / depth_one_correct
            if depth_one_correct
            else 0.0
        )
        metrics[group] = {
            **dict(values),
            "depth_one_correct": depth_one_correct,
            "depth_two_correct": depth_two_correct,
            "net_delta": depth_two_correct - depth_one_correct,
            "depth_one_correct_retention": retention,
        }

    gates = {
        "depth_one_receipt_matches": all(
            metrics[group]["depth_one_correct"] == expected
            for group, expected in DEPTH_ONE_FLOORS.items()
        ),
        "material_overall_gain": metrics["overall"]["depth_two_correct"]
        >= DEPTH_TWO_FLOOR,
        "all_domains_nonnegative": all(
            metrics[task]["net_delta"] >= 0 for task in TASKS
        ),
        "depth_one_correct_retention": metrics["overall"]
        ["depth_one_correct_retention"]
        >= RETENTION_FLOOR,
    }
    output = {
        "schema": "shohin-ridr1-recurrent-depth-comparison-v1",
        "status": "complete",
        "frozen_thresholds": {
            "depth_one_correct": DEPTH_ONE_FLOORS,
            "depth_two_overall_correct": DEPTH_TWO_FLOOR,
            "minimum_depth_one_correct_retention": RETENTION_FLOOR,
            "domain_delta_floor": 0,
        },
        "depth_one": {
            "candidates": str(args.depth_one_candidates.resolve()),
            "candidates_sha256": sha256_file(args.depth_one_candidates),
            "report": str(args.depth_one_report.resolve()),
            "report_sha256": sha256_file(args.depth_one_report),
        },
        "depth_two": {
            "candidates": str(args.depth_two_candidates.resolve()),
            "candidates_sha256": sha256_file(args.depth_two_candidates),
            "report": str(args.depth_two_report.resolve()),
            "report_sha256": sha256_file(args.depth_two_report),
        },
        "metrics": metrics,
        "gates": gates,
        "gate_pass": all(gates.values()),
        "failure_policy": (
            "close recurrent depth for this reviser; no depth-three, prompt, seed, "
            "or threshold rescue"
        ),
        "success_policy": "unlock exactly one sealed holdout depth-two evaluation",
    }
    _atomic_json(args.output, output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-one-candidates", type=Path, required=True)
    parser.add_argument("--depth-one-report", type=Path, required=True)
    parser.add_argument("--depth-two-candidates", type=Path, required=True)
    parser.add_argument("--depth-two-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    output = compare(parser.parse_args())
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
