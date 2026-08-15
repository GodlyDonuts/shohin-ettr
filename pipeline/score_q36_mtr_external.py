#!/usr/bin/env python3
"""Score matched Q36 arms on a source-disjoint external partition."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from pcf1_code_sandbox import (
    atomic_json as sandbox_atomic_json,
    qualify_allocation,
    qualify_mbpp_assessor_setups,
    score_completion,
)

ASSESSOR_SCHEMA = "shohin-q36-mtr-external-validation-assessor-v1"
CANDIDATE_SCHEMA = "shohin-q36-mtr-candidate-v1"
REPORT_SCHEMA = "shohin-q36-mtr-external-score-v1"
ARMS = ("unchanged", "self_refinement", "revision", "draft_hidden", "interpolation")
TASKS = ("math500", "bbh_logic", "mbpp")
PARTITIONS = {
    "external_validation_screen": (256, 4),
    "external_validation": (1_023, 16),
    "external_validation_full": (1_279, 16),
}


class Q36MTRExternalScoreError(RuntimeError):
    """An external assessor, candidate, or score result differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise Q36MTRExternalScoreError(f"unreadable JSONL: {path}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise Q36MTRExternalScoreError(f"non-object JSONL row: {path}")
    return rows


def load_assessors(path: Path, expected_rows: int) -> dict[str, dict[str, Any]]:
    assessors: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        identity = row.get("identity_sha256")
        if (
            row.get("schema") != ASSESSOR_SCHEMA
            or row.get("split") != "external_validation"
            or row.get("task") not in TASKS
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in assessors
            or not isinstance(row.get("assessor"), dict)
            or row["assessor"].get("identity_sha256") != identity
            or row["assessor"].get("task") != row["task"]
        ):
            raise Q36MTRExternalScoreError("external assessor row differs")
        assessors[identity] = row
    if len(assessors) != expected_rows or {
        row["task"] for row in assessors.values()
    } != set(TASKS):
        raise Q36MTRExternalScoreError("external assessor coverage differs")
    return assessors


def load_candidates(
    arm: str, paths: list[Path], identities: set[str], expected_shards: int
) -> dict[str, dict[str, Any]]:
    if arm not in ARMS or len(paths) != expected_shards:
        raise Q36MTRExternalScoreError("external candidate geometry differs")
    candidates: dict[str, dict[str, Any]] = {}
    for path in paths:
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
                or not isinstance(row.get("max_token_exhausted"), bool)
            ):
                raise Q36MTRExternalScoreError("external candidate row differs")
            candidates[identity] = row
    if set(candidates) != identities:
        raise Q36MTRExternalScoreError("external candidate coverage differs")
    return candidates


def _mcnemar_exact(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(left_only, right_only) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRExternalScoreError("external score output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise Q36MTRExternalScoreError("external score output exists")
    expected = PARTITIONS.get(args.split)
    if expected != (args.expected_rows, args.shard_count):
        raise Q36MTRExternalScoreError("external partition geometry differs")
    assessors = load_assessors(args.assessors, args.expected_rows)
    identities = set(assessors)
    path_groups = {arm: getattr(args, f"{arm}_candidates") for arm in ARMS}
    candidates = {
        arm: load_candidates(arm, paths, identities, args.shard_count)
        for arm, paths in path_groups.items()
    }
    sandbox = qualify_allocation()
    sandbox_sha256 = sandbox_atomic_json(args.sandbox_receipt, sandbox)
    mbpp_assessors = [
        row["assessor"] for row in assessors.values() if row["task"] == "mbpp"
    ]
    setup_receipts = qualify_mbpp_assessor_setups(mbpp_assessors)

    outcomes: dict[str, dict[str, bool]] = {arm: {} for arm in ARMS}
    task_correct: dict[str, Counter[str]] = {arm: Counter() for arm in ARMS}
    task_rows = Counter(row["task"] for row in assessors.values())
    empty = Counter()
    exhausted = Counter()
    for identity in sorted(identities):
        assessor_row = assessors[identity]
        for arm in ARMS:
            candidate = candidates[arm][identity]
            if candidate["task"] != assessor_row["task"]:
                raise Q36MTRExternalScoreError("external task binding differs")
            score = score_completion(assessor_row["assessor"], candidate["completion"])
            correct = score.get("correct")
            if not isinstance(correct, bool):
                raise Q36MTRExternalScoreError("external score result differs")
            outcomes[arm][identity] = correct
            task_correct[arm][candidate["task"]] += int(correct)
            empty[arm] += int(not candidate["completion"].strip())
            exhausted[arm] += int(candidate["max_token_exhausted"])

    arm_reports: dict[str, Any] = {}
    unchanged = outcomes["unchanged"]
    for arm in ARMS:
        correct = sum(outcomes[arm].values())
        left_only = sum(
            outcomes[arm][identity] and not unchanged[identity]
            for identity in identities
        )
        right_only = sum(
            unchanged[identity] and not outcomes[arm][identity]
            for identity in identities
        )
        arm_reports[arm] = {
            "correct": correct,
            "total": args.expected_rows,
            "accuracy": correct / args.expected_rows,
            "domains": {
                task: {
                    "correct": task_correct[arm][task],
                    "total": task_rows[task],
                }
                for task in TASKS
            },
            "gain_over_unchanged_count": correct - sum(unchanged.values()),
            "paired_vs_unchanged": {
                "arm_only_correct": left_only,
                "unchanged_only_correct": right_only,
                "mcnemar_exact_two_sided_p": _mcnemar_exact(left_only, right_only),
            },
            "empty_completions": empty[arm],
            "max_token_exhausted": exhausted[arm],
            "candidate_sha256s": [sha256_file(path) for path in path_groups[arm]],
        }
    oracle = sum(
        any(outcomes[arm][identity] for arm in ARMS) for identity in identities
    )
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "split": args.split,
        "rows": args.expected_rows,
        "shard_count": args.shard_count,
        "assessors_sha256": sha256_file(args.assessors),
        "sandbox_receipt_sha256": sandbox_sha256,
        "sandbox_probe_sha256": sandbox.get("probe_sha256"),
        "mbpp_setup_qualification_count": len(setup_receipts),
        "arms": arm_reports,
        "all_arm_oracle_correct": oracle,
        "all_arm_oracle_accuracy": oracle / args.expected_rows,
    }
    _atomic_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assessors", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(PARTITIONS), required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
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


def main() -> None:
    report = run(parse_args())
    print(json.dumps({arm: row["correct"] for arm, row in report["arms"].items()}))


if __name__ == "__main__":
    main()
