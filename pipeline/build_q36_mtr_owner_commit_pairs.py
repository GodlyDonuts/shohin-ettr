#!/usr/bin/env python3
"""Build semantic-commit pairs from two independent Q36 owner trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_q36_mtr_commit_pairs import (
    CALIBRATION_SEED,
    OUTCOMES,
    PAIR_SCHEMA,
    REPORT_SCHEMA,
    calibration_split,
    expected_outcome,
)
from q36_mtr_roles import MODEL_REVISION

SOURCE_SCHEMAS = {
    "train": "shohin-pcf1-train-source-v1",
    "development": "shohin-pcf1-development-source-v1",
}
CANDIDATE_SCHEMA = "shohin-q36-mtr-model-draft-v1"
SCORE_SCHEMA = "shohin-q36-mtr-draft-preview-v1"
COUNTS = {"train": 5_824, "development": 1_289}
TASKS = ("math500", "bbh_logic", "mbpp")


class Q36MTROwnerCommitPairError(RuntimeError):
    """Owner trajectories, scores, or pair geometry differ."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise Q36MTROwnerCommitPairError(f"missing or linked input: {path}")
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise Q36MTROwnerCommitPairError(f"unreadable input: {path}") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise Q36MTROwnerCommitPairError(f"empty or malformed input: {path}")
    return rows


def _source(path: Path, split: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _jsonl(path):
        identity = row.get("identity_sha256")
        if (
            row.get("schema") != SOURCE_SCHEMAS[split]
            or row.get("split") != split
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in result
            or row.get("task") not in TASKS
            or not isinstance(row.get("source_prompt"), str)
            or not row["source_prompt"].strip()
        ):
            raise Q36MTROwnerCommitPairError("owner-commit source differs")
        if split == "development" and any(
            field in row for field in ("assessor", "answer", "gold", "response")
        ):
            raise Q36MTROwnerCommitPairError("development source exposes labels")
        result[identity] = row
    if len(result) != COUNTS[split]:
        raise Q36MTROwnerCommitPairError("owner-commit source count differs")
    return result


def _candidate_rows(paths: list[Path]) -> dict[str, dict[str, Any]]:
    if len(paths) != 16:
        raise Q36MTROwnerCommitPairError("owner candidate shard count differs")
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _jsonl(path):
            identity = row.get("identity_sha256")
            if (
                row.get("schema") != CANDIDATE_SCHEMA
                or not isinstance(identity, str)
                or len(identity) != 64
                or identity in result
                or row.get("split") not in COUNTS
                or row.get("task") not in TASKS
                or not isinstance(row.get("completion"), str)
                or not row["completion"].strip()
                or isinstance(row.get("generated_tokens"), bool)
                or not isinstance(row.get("generated_tokens"), int)
                or row["generated_tokens"] < 0
                or not isinstance(row.get("max_token_exhausted"), bool)
            ):
                raise Q36MTROwnerCommitPairError("owner candidate differs")
            result[identity] = row
    if len(result) != sum(COUNTS.values()):
        raise Q36MTROwnerCommitPairError("owner candidate coverage differs")
    return result


def _scores(
    paths: list[Path],
    candidates: dict[str, dict[str, Any]],
    candidate_paths: list[Path],
    split: str,
) -> dict[str, dict[str, Any]]:
    if len(paths) != 16:
        raise Q36MTROwnerCommitPairError("owner score shard count differs")
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise Q36MTROwnerCommitPairError("owner score report is missing or linked")
        report = json.loads(path.read_text(encoding="utf-8"))
        outcomes = report.get("outcomes")
        if (
            report.get("schema") != SCORE_SCHEMA
            or report.get("status") != "complete"
            or report.get("split") != split
            or not isinstance(outcomes, list)
            or report.get("rows") != len(outcomes)
            or not isinstance(report.get("candidates_sha256"), str)
        ):
            raise Q36MTROwnerCommitPairError("owner score report differs")
        candidate_hash = report["candidates_sha256"]
        matching_files = [
            candidate_path
            for candidate_path in candidate_paths
            if sha256_file(candidate_path) == candidate_hash
        ]
        if len(matching_files) != 1:
            raise Q36MTROwnerCommitPairError("owner score/candidate binding differs")
        for outcome in outcomes:
            identity = outcome.get("identity_sha256")
            candidate = candidates.get(identity)
            if (
                not isinstance(identity, str)
                or identity in result
                or candidate is None
                or candidate.get("split") != split
                or outcome.get("task") != candidate.get("task")
                or not isinstance(outcome.get("correct"), bool)
            ):
                raise Q36MTROwnerCommitPairError("owner score outcome differs")
            result[identity] = outcome
    if len(result) != COUNTS[split]:
        raise Q36MTROwnerCommitPairError("owner score coverage differs")
    return result


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise Q36MTROwnerCommitPairError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTROwnerCommitPairError(f"refusing existing report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    if args.seed != CALIBRATION_SEED:
        raise Q36MTROwnerCommitPairError("owner-commit seed differs")
    train_source = _source(args.train_source, "train")
    development_source = _source(args.development_source, "development")
    first = _candidate_rows(args.first_candidates)
    second = _candidate_rows(args.second_candidates)
    first_train = _scores(args.first_train_score, first, args.first_candidates, "train")
    second_train = _scores(
        args.second_train_score, second, args.second_candidates, "train"
    )
    first_development = _scores(
        args.first_development_score, first, args.first_candidates, "development"
    )
    second_development = _scores(
        args.second_development_score,
        second,
        args.second_candidates,
        "development",
    )
    expected = set(train_source) | set(development_source)
    if set(first) != expected or set(second) != expected:
        raise Q36MTROwnerCommitPairError("owner pair identity coverage differs")

    training_rows: list[dict[str, Any]] = []
    outcomes: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for identity in sorted(train_source):
        source = train_source[identity]
        left, right = first[identity], second[identity]
        if left["task"] != source["task"] or right["task"] != source["task"]:
            raise Q36MTROwnerCommitPairError("owner pair task binding differs")
        left_correct = bool(first_train[identity]["correct"])
        right_correct = bool(second_train[identity]["correct"])
        outcome = expected_outcome(left_correct, right_correct)
        local_split = calibration_split(identity, args.seed)
        outcomes[outcome] += 1
        split_counts[local_split] += 1
        training_rows.append(
            {
                "schema": PAIR_SCHEMA,
                "identity_sha256": identity,
                "split": local_split,
                "task": source["task"],
                "question": source["source_prompt"],
                "outcome_class": outcome,
                "candidates": [
                    {
                        "lineage": "revision",
                        "completion": left["completion"],
                        "correct": left_correct,
                        "generated_tokens": left["generated_tokens"],
                        "max_token_exhausted": left["max_token_exhausted"],
                    },
                    {
                        "lineage": "unchanged",
                        "completion": right["completion"],
                        "correct": right_correct,
                        "generated_tokens": right["generated_tokens"],
                        "max_token_exhausted": right["max_token_exhausted"],
                    },
                ],
            }
        )

    development_rows: list[dict[str, Any]] = []
    for identity in sorted(development_source):
        source = development_source[identity]
        left, right = first[identity], second[identity]
        if (
            left["task"] != source["task"]
            or right["task"] != source["task"]
            or identity not in first_development
            or identity not in second_development
        ):
            raise Q36MTROwnerCommitPairError("development owner pair binding differs")
        development_rows.append(
            {
                "schema": PAIR_SCHEMA,
                "identity_sha256": identity,
                "split": "development",
                "task": source["task"],
                "question": source["source_prompt"],
                "candidates": [
                    {"lineage": "revision", "completion": left["completion"]},
                    {"lineage": "unchanged", "completion": right["completion"]},
                ],
            }
        )

    training_sha = _atomic_lines(args.training_output, training_rows)
    development_sha = _atomic_lines(args.development_output, development_rows)
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
        "first_candidate_files": [sha256_file(path) for path in args.first_candidates],
        "second_candidate_files": [
            sha256_file(path) for path in args.second_candidates
        ],
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
        "first_candidate_files": [sha256_file(path) for path in args.first_candidates],
        "second_candidate_files": [
            sha256_file(path) for path in args.second_candidates
        ],
    }
    _atomic_json(args.training_report, training_report)
    _atomic_json(args.development_report, development_report)
    return training_report, development_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--development-source", type=Path, required=True)
    parser.add_argument("--first-candidates", type=Path, action="append", required=True)
    parser.add_argument(
        "--second-candidates", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--first-train-score", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--second-train-score", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--first-development-score", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--second-development-score", type=Path, action="append", required=True
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
