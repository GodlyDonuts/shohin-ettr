#!/usr/bin/env python3
"""Build three-owner setwise Q36 commit rows on exact shared identities."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import build_q36_mtr_owner_commit_pairs as base
from build_q36_mtr_commit_pairs import CALIBRATION_SEED, calibration_split
from q36_mtr_roles import MODEL_REVISION

ROW_SCHEMA = "shohin-q36-mtr-setwise-commit-row-v1"
REPORT_SCHEMA = "shohin-q36-mtr-setwise-commit-data-report-v1"
OWNER_NAMES = ("current", "owner_71", "owner_8")
PATTERNS = tuple(f"{index:03b}" for index in range(8))


class Q36MTRSetwiseDataError(RuntimeError):
    """The three-owner setwise data geometry or lineage differs."""


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
        raise Q36MTRSetwiseDataError("setwise calibration seed differs")
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
        raise Q36MTRSetwiseDataError("setwise owner candidate coverage differs")

    training_rows: list[dict[str, Any]] = []
    patterns: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    for identity in sorted(train_source):
        source = train_source[identity]
        owner_rows = [owner[identity] for owner in candidates]
        if any(row["task"] != source["task"] for row in owner_rows):
            raise Q36MTRSetwiseDataError("setwise train task binding differs")
        correctness = [bool(scores[identity]["correct"]) for scores in train_scores]
        pattern = "".join("1" if value else "0" for value in correctness)
        split = calibration_split(identity, args.seed)
        patterns[pattern] += 1
        splits[split] += 1
        training_rows.append(
            {
                "schema": ROW_SCHEMA,
                "identity_sha256": identity,
                "split": split,
                "task": source["task"],
                "question": source["source_prompt"],
                "correctness_pattern": pattern,
                "candidates": [
                    _candidate(row, lineage=lineage, correct=correct)
                    for row, lineage, correct in zip(
                        owner_rows, OWNER_NAMES, correctness, strict=True
                    )
                ],
            }
        )
    if set(patterns) != set(PATTERNS):
        raise Q36MTRSetwiseDataError("setwise training lacks a correctness pattern")

    development_rows: list[dict[str, Any]] = []
    for identity in sorted(development_source):
        source = development_source[identity]
        owner_rows = [owner[identity] for owner in candidates]
        if any(row["task"] != source["task"] for row in owner_rows) or any(
            identity not in scores for scores in development_scores
        ):
            raise Q36MTRSetwiseDataError("setwise development binding differs")
        development_rows.append(
            {
                "schema": ROW_SCHEMA,
                "identity_sha256": identity,
                "split": "development",
                "task": source["task"],
                "question": source["source_prompt"],
                "candidates": [
                    _candidate(row, lineage=lineage, correct=None)
                    for row, lineage in zip(owner_rows, OWNER_NAMES, strict=True)
                ],
            }
        )

    training_sha = base._atomic_lines(args.training_output, training_rows)
    development_sha = base._atomic_lines(args.development_output, development_rows)
    candidate_hashes = {
        name: [base.sha256_file(path) for path in paths]
        for name, paths in zip(OWNER_NAMES, candidate_paths, strict=True)
    }
    training_report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "model_revision": MODEL_REVISION,
        "source_split": "calibration",
        "rows": len(training_rows),
        "output": str(args.training_output.resolve()),
        "output_sha256": training_sha,
        "owner_lineages": list(OWNER_NAMES),
        "correctness_patterns": {name: patterns[name] for name in PATTERNS},
        "calibration_splits": dict(sorted(splits.items())),
        "owner_candidate_files": candidate_hashes,
        "permutation_equivariant_training_target": True,
    }
    development_report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "model_revision": MODEL_REVISION,
        "source_split": "development",
        "rows": len(development_rows),
        "output": str(args.development_output.resolve()),
        "output_sha256": development_sha,
        "owner_lineages": list(OWNER_NAMES),
        "labels_or_correctness_fields": 0,
        "source_disjoint_from_calibration": True,
        "owner_candidate_files": candidate_hashes,
        "permutation_equivariant_training_target": True,
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
