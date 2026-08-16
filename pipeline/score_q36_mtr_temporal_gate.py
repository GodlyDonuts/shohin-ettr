#!/usr/bin/env python3
"""Score a trained Q36 temporal gate against the matched unchanged arm."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any

from pcf1_code_sandbox import (
    atomic_json as sandbox_atomic_json,
    qualify_allocation,
    qualify_mbpp_assessor_setups,
    score_completion,
)
from score_q36_mtr_external import (
    CANDIDATE_SCHEMA,
    REPORT_SCHEMA as BASELINE_SCHEMA,
    _mcnemar_exact,
    load_assessors,
    sha256_file,
)

ARM = "temporal_gate"
ARMS = (ARM, "multi_trajectory_gate")
REPORT_SCHEMA = "shohin-q36-mtr-temporal-gate-score-v1"
TASKS = ("math500", "bbh_logic", "mbpp")


class Q36MTRTemporalGateScoreError(RuntimeError):
    """The temporal candidates, baseline, or score result differs."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Q36MTRTemporalGateScoreError(f"unreadable JSON: {path}") from error
    if not isinstance(payload, dict):
        raise Q36MTRTemporalGateScoreError("temporal JSON differs")
    return payload


def load_temporal_candidates(
    paths: list[Path], identities: set[str], expected_shards: int, arm: str = ARM
) -> dict[str, dict[str, Any]]:
    if len(paths) != expected_shards:
        raise Q36MTRTemporalGateScoreError("temporal candidate geometry differs")
    candidates: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise Q36MTRTemporalGateScoreError(
                    "temporal candidate JSON differs"
                ) from error
            identity = row.get("identity_sha256") if isinstance(row, dict) else None
            if (
                not isinstance(row, dict)
                or row.get("schema") != CANDIDATE_SCHEMA
                or row.get("arm") != arm
                or row.get("task") not in TASKS
                or not isinstance(identity, str)
                or identity in candidates
                or not isinstance(row.get("completion"), str)
                or not isinstance(row.get("generated_tokens"), int)
                or isinstance(row.get("generated_tokens"), bool)
                or not isinstance(row.get("max_token_exhausted"), bool)
            ):
                raise Q36MTRTemporalGateScoreError("temporal candidate row differs")
            candidates[identity] = row
    if set(candidates) != identities:
        raise Q36MTRTemporalGateScoreError("temporal candidate coverage differs")
    return candidates


def load_baseline(
    path: Path, assessors: dict[str, dict[str, Any]], expected_rows: int
) -> dict[str, bool]:
    report = _load_json(path)
    outcomes = report.get("outcomes")
    if (
        report.get("schema") != BASELINE_SCHEMA
        or report.get("status") != "complete"
        or report.get("rows") != expected_rows
        or not isinstance(outcomes, list)
        or len(outcomes) != expected_rows
    ):
        raise Q36MTRTemporalGateScoreError("temporal baseline report differs")
    baseline: dict[str, bool] = {}
    for row in outcomes:
        identity = row.get("identity_sha256") if isinstance(row, dict) else None
        correct = row.get("correct") if isinstance(row, dict) else None
        value = correct.get("unchanged") if isinstance(correct, dict) else None
        if (
            not isinstance(identity, str)
            or identity in baseline
            or identity not in assessors
            or row.get("task") != assessors[identity]["task"]
            or not isinstance(value, bool)
        ):
            raise Q36MTRTemporalGateScoreError("temporal baseline outcome differs")
        baseline[identity] = value
    if set(baseline) != set(assessors):
        raise Q36MTRTemporalGateScoreError("temporal baseline coverage differs")
    return baseline


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRTemporalGateScoreError("temporal score output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.sandbox_receipt.exists():
        raise Q36MTRTemporalGateScoreError("temporal score output exists")
    assessors = load_assessors(args.assessors, args.expected_rows)
    identities = set(assessors)
    candidates = load_temporal_candidates(
        args.temporal_candidates, identities, args.shard_count, args.arm
    )
    baseline = load_baseline(args.baseline_score, assessors, args.expected_rows)
    sandbox = qualify_allocation()
    sandbox_sha256 = sandbox_atomic_json(args.sandbox_receipt, sandbox)
    setups = qualify_mbpp_assessor_setups(
        [row["assessor"] for row in assessors.values() if row["task"] == "mbpp"]
    )
    temporal: dict[str, bool] = {}
    task_correct: Counter[str] = Counter()
    task_rows = Counter(row["task"] for row in assessors.values())
    empty = exhausted = 0
    for identity in sorted(identities):
        candidate = candidates[identity]
        assessor = assessors[identity]
        if candidate["task"] != assessor["task"]:
            raise Q36MTRTemporalGateScoreError("temporal task binding differs")
        result = score_completion(assessor["assessor"], candidate["completion"])
        correct = result.get("correct")
        if not isinstance(correct, bool):
            raise Q36MTRTemporalGateScoreError("temporal score result differs")
        temporal[identity] = correct
        task_correct[candidate["task"]] += int(correct)
        empty += int(not candidate["completion"].strip())
        exhausted += int(candidate["max_token_exhausted"])
    temporal_correct = sum(temporal.values())
    baseline_correct = sum(baseline.values())
    temporal_only = sum(temporal[i] and not baseline[i] for i in identities)
    baseline_only = sum(baseline[i] and not temporal[i] for i in identities)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "rows": args.expected_rows,
        "shard_count": args.shard_count,
        "arm": args.arm,
        "assessors_sha256": sha256_file(args.assessors),
        "baseline_score_sha256": sha256_file(args.baseline_score),
        "temporal_candidate_sha256s": [
            sha256_file(path) for path in args.temporal_candidates
        ],
        "sandbox_receipt_sha256": sandbox_sha256,
        "sandbox_probe_sha256": sandbox.get("probe_sha256"),
        "mbpp_setup_qualification_count": len(setups),
        args.arm: {
            "correct": temporal_correct,
            "total": args.expected_rows,
            "accuracy": temporal_correct / args.expected_rows,
            "domains": {
                task: {"correct": task_correct[task], "total": task_rows[task]}
                for task in TASKS
            },
            "empty_completions": empty,
            "max_token_exhausted": exhausted,
        },
        "unchanged": {
            "correct": baseline_correct,
            "total": args.expected_rows,
            "accuracy": baseline_correct / args.expected_rows,
        },
        "gain_over_unchanged_count": temporal_correct - baseline_correct,
        "paired_vs_unchanged": {
            "temporal_only_correct": temporal_only,
            "unchanged_only_correct": baseline_only,
            "mcnemar_exact_two_sided_p": _mcnemar_exact(temporal_only, baseline_only),
        },
        "outcomes": [
            {
                "identity_sha256": identity,
                "task": assessors[identity]["task"],
                f"{args.arm}_correct": temporal[identity],
                "unchanged_correct": baseline[identity],
            }
            for identity in sorted(identities)
        ],
    }
    _atomic_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessors", type=Path, required=True)
    parser.add_argument("--baseline-score", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, default=ARM)
    parser.add_argument(
        "--temporal-candidates", type=Path, action="append", required=True
    )
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--sandbox-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                f"{args.arm}_correct": report[args.arm]["correct"],
                "unchanged_correct": report["unchanged"]["correct"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
