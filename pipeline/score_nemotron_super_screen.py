#!/usr/bin/env python3
"""Score the matched fixed-draft Nemotron Super screen once."""

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
    TASKS,
    _load_jsonl,
    _mcnemar_exact,
    load_assessors,
    sha256_file,
)

CANDIDATE_SCHEMA = "shohin-nemotron-super-fixed-draft-candidate-v1"
REPORT_SCHEMA = "shohin-nemotron-super-fixed-draft-screen-score-v1"
ARMS = ("unchanged", "self_refinement", "revision")
ROWS = 256
SHARDS = 4


class NemotronSuperScoreError(RuntimeError):
    """The matched 120B-A12B score contract differed."""


def load_candidates(
    arm: str, paths: list[Path], identities: set[str]
) -> dict[str, dict[str, Any]]:
    if arm not in ARMS or len(paths) != SHARDS:
        raise NemotronSuperScoreError("candidate shard geometry differs")
    candidates: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise NemotronSuperScoreError("candidate shard is absent")
        for row in _load_jsonl(path):
            identity = row.get("identity_sha256")
            if (
                row.get("schema") != CANDIDATE_SCHEMA
                or row.get("arm") != arm
                or row.get("task") not in TASKS
                or not isinstance(identity, str)
                or identity in candidates
                or not isinstance(row.get("completion"), str)
                or not isinstance(row.get("generated_tokens"), int)
                or isinstance(row.get("generated_tokens"), bool)
                or row["generated_tokens"] < 0
                or not isinstance(row.get("max_token_exhausted"), bool)
            ):
                raise NemotronSuperScoreError("candidate row differs")
            candidates[identity] = row
    if set(candidates) != identities:
        raise NemotronSuperScoreError("candidate identity coverage differs")
    return candidates


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise NemotronSuperScoreError("score output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def paired_report(
    left: dict[str, bool], right: dict[str, bool]
) -> dict[str, int | float]:
    if set(left) != set(right) or not left:
        raise NemotronSuperScoreError("paired outcome coverage differs")
    left_only = sum(left[identity] and not right[identity] for identity in left)
    right_only = sum(right[identity] and not left[identity] for identity in left)
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "net_correct": left_only - right_only,
        "mcnemar_exact_two_sided_p": _mcnemar_exact(left_only, right_only),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise NemotronSuperScoreError("score output exists")
    assessors = load_assessors(args.assessors, ROWS)
    identities = set(assessors)
    paths = {arm: getattr(args, f"{arm}_candidates") for arm in ARMS}
    candidates = {arm: load_candidates(arm, paths[arm], identities) for arm in ARMS}
    sandbox = qualify_allocation()
    sandbox_sha256 = sandbox_atomic_json(args.sandbox_receipt, sandbox)
    setup_receipts = qualify_mbpp_assessor_setups(
        [row["assessor"] for row in assessors.values() if row["task"] == "mbpp"]
    )

    outcomes: dict[str, dict[str, bool]] = {arm: {} for arm in ARMS}
    domain_correct: dict[str, Counter[str]] = {arm: Counter() for arm in ARMS}
    domain_rows = Counter(row["task"] for row in assessors.values())
    empty = Counter()
    exhausted = Counter()
    for identity in sorted(identities):
        assessor = assessors[identity]
        for arm in ARMS:
            candidate = candidates[arm][identity]
            if candidate["task"] != assessor["task"]:
                raise NemotronSuperScoreError("candidate task differs")
            result = score_completion(assessor["assessor"], candidate["completion"])
            correct = result.get("correct")
            if not isinstance(correct, bool):
                raise NemotronSuperScoreError("score result differs")
            outcomes[arm][identity] = correct
            domain_correct[arm][candidate["task"]] += int(correct)
            empty[arm] += int(not candidate["completion"].strip())
            exhausted[arm] += int(candidate["max_token_exhausted"])

    unchanged_correct = sum(outcomes["unchanged"].values())
    self_correct = sum(outcomes["self_refinement"].values())
    arm_reports: dict[str, Any] = {}
    for arm in ARMS:
        correct = sum(outcomes[arm].values())
        retained = sum(
            outcomes["unchanged"][identity] and outcomes[arm][identity]
            for identity in identities
        )
        arm_reports[arm] = {
            "correct": correct,
            "total": ROWS,
            "accuracy": correct / ROWS,
            "gain_over_unchanged_count": correct - unchanged_correct,
            "gain_over_unchanged_percentage_points": 100.0
            * (correct - unchanged_correct)
            / ROWS,
            "gain_over_self_refinement_count": correct - self_correct,
            "unchanged_correct_retained": retained,
            "unchanged_correct_retention": (
                retained / unchanged_correct if unchanged_correct else None
            ),
            "domains": {
                task: {
                    "correct": domain_correct[arm][task],
                    "total": domain_rows[task],
                }
                for task in TASKS
            },
            "empty_completions": empty[arm],
            "max_token_exhausted": exhausted[arm],
            "candidate_sha256s": [sha256_file(path) for path in paths[arm]],
        }
    revision_vs_unchanged = paired_report(outcomes["revision"], outcomes["unchanged"])
    revision_vs_self = paired_report(outcomes["revision"], outcomes["self_refinement"])
    outcome_rows = [
        {
            "identity_sha256": identity,
            "task": assessors[identity]["task"],
            "correct": {arm: outcomes[arm][identity] for arm in ARMS},
        }
        for identity in sorted(identities)
    ]
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "host": "NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
        "total_parameters": 120_000_000_000,
        "active_parameters": 12_000_000_000,
        "rows": ROWS,
        "shards_per_arm": SHARDS,
        "assessors_sha256": sha256_file(args.assessors),
        "sandbox_receipt_sha256": sandbox_sha256,
        "sandbox_probe_sha256": sandbox.get("probe_sha256"),
        "mbpp_setup_qualification_count": len(setup_receipts),
        "arms": arm_reports,
        "revision_vs_unchanged": revision_vs_unchanged,
        "revision_vs_self_refinement": revision_vs_self,
        "outcomes": outcome_rows,
        "decision": {
            "revision_improves_unchanged": revision_vs_unchanged["net_correct"] > 0,
            "revision_improves_self_refinement": revision_vs_self["net_correct"] > 0,
            "retains_at_least_95_percent_unchanged_correct": (
                arm_reports["revision"]["unchanged_correct_retention"] is not None
                and arm_reports["revision"]["unchanged_correct_retention"] >= 0.95
            ),
            "all_domains_nonnegative_vs_unchanged": all(
                domain_correct["revision"][task] >= domain_correct["unchanged"][task]
                for task in TASKS
            ),
        },
    }
    _atomic_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessors", type=Path, required=True)
    for arm in ARMS:
        parser.add_argument(
            f"--{arm.replace('_', '-')}-candidates",
            type=Path,
            action="append",
            required=True,
        )
    parser.add_argument("--sandbox-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {arm: result["arms"][arm]["correct"] for arm in ARMS}, sort_keys=True
        )
    )
