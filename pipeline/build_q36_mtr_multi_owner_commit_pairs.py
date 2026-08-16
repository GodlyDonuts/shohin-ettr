#!/usr/bin/env python3
"""Build diversified pairwise training rows from three Q36 owner trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import build_q36_mtr_owner_commit_pairs as base
from build_q36_mtr_commit_pairs import (
    CALIBRATION_SEED,
    OUTCOMES,
    PAIR_SCHEMA,
    REPORT_SCHEMA,
    calibration_split,
    expected_outcome,
)
from q36_mtr_roles import MODEL_REVISION

OWNER_NAMES = ("current", "owner_71", "owner_8")
PAIR_CHOICES = ((0, 1), (0, 2), (1, 2))


class Q36MTRMultiOwnerPairError(RuntimeError):
    """The diversified owner pair geometry or source binding differs."""


def owner_pair_index(identity: str, seed: int) -> int:
    """Assign one source-disjoint owner pair without exposing owner identity."""

    if not isinstance(identity, str) or len(identity) != 64:
        raise Q36MTRMultiOwnerPairError("multi-owner identity differs")
    digest = hashlib.sha256(f"q36-multi-owner:{seed}:{identity}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % len(PAIR_CHOICES)


def _candidate(
    row: dict[str, Any], *, lineage: str, correct: bool | None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "lineage": lineage,
        "completion": row["completion"],
    }
    if correct is not None:
        result.update(
            {
                "correct": correct,
                "generated_tokens": row["generated_tokens"],
                "max_token_exhausted": row["max_token_exhausted"],
            }
        )
    return result


def build(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.seed != CALIBRATION_SEED:
        raise Q36MTRMultiOwnerPairError("multi-owner seed differs")
    train_source = base._source(args.train_source, "train")
    development_source = base._source(args.development_source, "development")
    candidate_paths = (
        args.current_candidates,
        args.owner71_candidates,
        args.owner8_candidates,
    )
    train_score_paths = (
        args.current_train_score,
        args.owner71_train_score,
        args.owner8_train_score,
    )
    development_score_paths = (
        args.current_development_score,
        args.owner71_development_score,
        args.owner8_development_score,
    )
    candidates = [base._candidate_rows(paths) for paths in candidate_paths]
    train_scores = [
        base._scores(paths, owner, owner_paths, "train")
        for paths, owner, owner_paths in zip(
            train_score_paths, candidates, candidate_paths, strict=True
        )
    ]
    development_scores = [
        base._scores(paths, owner, owner_paths, "development")
        for paths, owner, owner_paths in zip(
            development_score_paths, candidates, candidate_paths, strict=True
        )
    ]
    expected = set(train_source) | set(development_source)
    if any(set(owner) != expected for owner in candidates):
        raise Q36MTRMultiOwnerPairError("multi-owner candidate coverage differs")

    training_rows: list[dict[str, Any]] = []
    outcomes: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    for identity in sorted(train_source):
        source = train_source[identity]
        left_index, right_index = PAIR_CHOICES[owner_pair_index(identity, args.seed)]
        left = candidates[left_index][identity]
        right = candidates[right_index][identity]
        if left["task"] != source["task"] or right["task"] != source["task"]:
            raise Q36MTRMultiOwnerPairError("multi-owner train task binding differs")
        left_correct = bool(train_scores[left_index][identity]["correct"])
        right_correct = bool(train_scores[right_index][identity]["correct"])
        outcome = expected_outcome(left_correct, right_correct)
        local_split = calibration_split(identity, args.seed)
        pair_name = f"{OWNER_NAMES[left_index]}:{OWNER_NAMES[right_index]}"
        outcomes[outcome] += 1
        split_counts[local_split] += 1
        pair_counts[pair_name] += 1
        training_rows.append(
            {
                "schema": PAIR_SCHEMA,
                "identity_sha256": identity,
                "split": local_split,
                "task": source["task"],
                "question": source["source_prompt"],
                "outcome_class": outcome,
                "candidates": [
                    _candidate(left, lineage="revision", correct=left_correct),
                    _candidate(right, lineage="unchanged", correct=right_correct),
                ],
            }
        )
    if set(outcomes) != set(OUTCOMES) or set(pair_counts) != {
        f"{OWNER_NAMES[left]}:{OWNER_NAMES[right]}" for left, right in PAIR_CHOICES
    }:
        raise Q36MTRMultiOwnerPairError("multi-owner training coverage differs")

    development_rows: list[dict[str, Any]] = []
    development_pair_counts: Counter[str] = Counter()
    for identity in sorted(development_source):
        source = development_source[identity]
        left_index, right_index = PAIR_CHOICES[owner_pair_index(identity, args.seed)]
        left = candidates[left_index][identity]
        right = candidates[right_index][identity]
        if (
            left["task"] != source["task"]
            or right["task"] != source["task"]
            or any(identity not in scores for scores in development_scores)
        ):
            raise Q36MTRMultiOwnerPairError(
                "multi-owner development task binding differs"
            )
        development_pair_counts[
            f"{OWNER_NAMES[left_index]}:{OWNER_NAMES[right_index]}"
        ] += 1
        development_rows.append(
            {
                "schema": PAIR_SCHEMA,
                "identity_sha256": identity,
                "split": "development",
                "task": source["task"],
                "question": source["source_prompt"],
                "candidates": [
                    _candidate(left, lineage="revision", correct=None),
                    _candidate(right, lineage="unchanged", correct=None),
                ],
            }
        )

    training_sha = base._atomic_lines(args.training_output, training_rows)
    development_sha = base._atomic_lines(args.development_output, development_rows)
    training_report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "model_revision": MODEL_REVISION,
        "source_split": "calibration",
        "rows": len(training_rows),
        "output": str(args.training_output.resolve()),
        "output_sha256": training_sha,
        "outcomes": {name: outcomes[name] for name in OUTCOMES},
        "calibration_splits": dict(sorted(split_counts.items())),
        "owner_trajectory_pair": True,
        "multi_owner_diversified_pairwise": True,
        "owner_pair_counts": dict(sorted(pair_counts.items())),
        "owner_candidate_files": {
            name: [base.sha256_file(path) for path in paths]
            for name, paths in zip(OWNER_NAMES, candidate_paths, strict=True)
        },
    }
    development_report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "model_revision": MODEL_REVISION,
        "source_split": "development",
        "rows": len(development_rows),
        "output": str(args.development_output.resolve()),
        "output_sha256": development_sha,
        "labels_or_correctness_fields": 0,
        "source_disjoint_from_calibration": True,
        "owner_trajectory_pair": True,
        "multi_owner_diversified_pairwise": True,
        "owner_pair_counts": dict(sorted(development_pair_counts.items())),
        "owner_candidate_files": {
            name: [base.sha256_file(path) for path in paths]
            for name, paths in zip(OWNER_NAMES, candidate_paths, strict=True)
        },
    }
    base._atomic_json(args.training_report, training_report)
    base._atomic_json(args.development_report, development_report)
    return training_report, development_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--development-source", type=Path, required=True)
    for owner in ("current", "owner71", "owner8"):
        parser.add_argument(
            f"--{owner}-candidates", type=Path, action="append", required=True
        )
        parser.add_argument(
            f"--{owner}-train-score", type=Path, action="append", required=True
        )
        parser.add_argument(
            f"--{owner}-development-score",
            type=Path,
            action="append",
            required=True,
        )
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--development-output", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=CALIBRATION_SEED)
    return parser.parse_args()


def main() -> int:
    training, development = build(parse_args())
    print(
        json.dumps({"training": training, "development": development}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
