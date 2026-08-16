#!/usr/bin/env python3
"""Cross-validate the Q36 sparse router over exact held-out identity shards."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from typing import Any

import train_apply_q36_mtr_sparse_router as router

REPORT_SCHEMA = "shohin-q36-mtr-sparse-router-cross-validation-v1"
SCORE_SCHEMA = "shohin-q36-mtr-draft-preview-v1"
SOURCE_SCHEMA = "shohin-pcf1-development-source-v1"
FOLDS = 16
EPOCHS = 8
LEARNING_RATE = 0.07
BALANCED_PATTERNS = True


class Q36MTRSparseCrossValidationError(RuntimeError):
    """Cross-validation inputs, identity folds, or scores differ."""


def _source(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in router._jsonl(path):
        identity = row.get("identity_sha256")
        if (
            row.get("schema") != SOURCE_SCHEMA
            or row.get("split") != "development"
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in result
            or row.get("task") not in router.TASKS
            or not isinstance(row.get("source_prompt"), str)
            or not row["source_prompt"].strip()
            or any(field in row for field in ("assessor", "answer", "gold"))
        ):
            raise Q36MTRSparseCrossValidationError(
                "sparse cross-validation source differs"
            )
        result[identity] = row
    if len(result) != router.DEVELOPMENT_ROWS:
        raise Q36MTRSparseCrossValidationError(
            "sparse cross-validation source coverage differs"
        )
    return result


def _candidates(
    paths: list[Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    rows = router.load_development_candidates(paths, expected_shards=FOLDS)
    folds: dict[str, int] = {}
    for fold, path in enumerate(paths):
        for row in router._jsonl(path):
            if row.get("split") != "development":
                continue
            identity = row.get("identity_sha256")
            if identity in folds:
                raise Q36MTRSparseCrossValidationError(
                    "sparse cross-validation fold overlaps"
                )
            folds[identity] = fold
    if set(folds) != set(rows):
        raise Q36MTRSparseCrossValidationError(
            "sparse cross-validation fold coverage differs"
        )
    return rows, folds


def _scores(
    paths: list[Path],
    candidates: dict[str, dict[str, Any]],
    candidate_paths: list[Path],
) -> dict[str, dict[str, Any]]:
    if len(paths) != FOLDS:
        raise Q36MTRSparseCrossValidationError(
            "sparse cross-validation score shard count differs"
        )
    candidate_hashes = {
        router.sha256_file(path): path.resolve(strict=True) for path in candidate_paths
    }
    if len(candidate_hashes) != FOLDS:
        raise Q36MTRSparseCrossValidationError(
            "sparse cross-validation candidate hashes differ"
        )
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise Q36MTRSparseCrossValidationError(
                "sparse cross-validation score is missing or linked"
            )
        report = json.loads(path.read_text(encoding="utf-8"))
        outcomes = report.get("outcomes") if isinstance(report, dict) else None
        if (
            report.get("schema") != SCORE_SCHEMA
            or report.get("status") != "complete"
            or report.get("split") != "development"
            or report.get("candidates_sha256") not in candidate_hashes
            or not isinstance(outcomes, list)
            or report.get("rows") != len(outcomes)
            or report.get("correct")
            != sum(int(row.get("correct") is True) for row in outcomes)
        ):
            raise Q36MTRSparseCrossValidationError(
                "sparse cross-validation score report differs"
            )
        for outcome in outcomes:
            identity = outcome.get("identity_sha256")
            candidate = candidates.get(identity)
            if (
                not isinstance(identity, str)
                or identity in result
                or candidate is None
                or outcome.get("task") != candidate.get("task")
                or not isinstance(outcome.get("correct"), bool)
            ):
                raise Q36MTRSparseCrossValidationError(
                    "sparse cross-validation outcome differs"
                )
            result[identity] = outcome
    if set(result) != set(candidates):
        raise Q36MTRSparseCrossValidationError(
            "sparse cross-validation score coverage differs"
        )
    return result


def _explicit(candidate: dict[str, Any]) -> bool:
    return candidate["task"] == "mbpp" or bool(
        router.EXPLICIT_RE.search(candidate["completion"])
    )


def _heuristic(candidates: list[dict[str, Any]]) -> int:
    selected = 0
    for challenger in range(1, len(candidates)):
        current, proposed = candidates[selected], candidates[challenger]
        current_explicit, proposed_explicit = _explicit(current), _explicit(proposed)
        if proposed_explicit and not current_explicit:
            selected = challenger
        elif (
            proposed_explicit == current_explicit
            and current["max_token_exhausted"]
            and not proposed["max_token_exhausted"]
        ):
            selected = challenger
    return selected


def _mcnemar(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if not discordant:
        return 1.0
    lower = min(first_only, second_only)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRSparseCrossValidationError("sparse cross-validation output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def cross_validate(args: argparse.Namespace) -> dict[str, Any]:
    source = _source(args.development_source)
    candidate_groups: list[dict[str, dict[str, Any]]] = []
    reference_folds: dict[str, int] | None = None
    candidate_paths = (
        args.current_candidates,
        args.owner71_candidates,
        args.owner8_candidates,
    )
    score_paths = (
        args.current_scores,
        args.owner71_scores,
        args.owner8_scores,
    )
    for paths in candidate_paths:
        candidates, folds = _candidates(paths)
        if reference_folds is None:
            reference_folds = folds
        elif folds != reference_folds:
            raise Q36MTRSparseCrossValidationError(
                "sparse cross-validation owner folds differ"
            )
        candidate_groups.append(candidates)
    assert reference_folds is not None
    scores = [
        _scores(paths, candidates, owners)
        for paths, candidates, owners in zip(
            score_paths, candidate_groups, candidate_paths, strict=True
        )
    ]
    if any(set(group) != set(source) for group in candidate_groups) or any(
        set(group) != set(source) for group in scores
    ):
        raise Q36MTRSparseCrossValidationError(
            "sparse cross-validation identity coverage differs"
        )

    rows: list[dict[str, Any]] = []
    by_identity: dict[str, dict[str, Any]] = {}
    patterns: Counter[str] = Counter()
    for identity in sorted(source):
        source_row = source[identity]
        candidates = [group[identity] for group in candidate_groups]
        correctness = [group[identity]["correct"] for group in scores]
        if any(candidate["task"] != source_row["task"] for candidate in candidates):
            raise Q36MTRSparseCrossValidationError(
                "sparse cross-validation task binding differs"
            )
        pattern = "".join("1" if value else "0" for value in correctness)
        row = {
            "identity_sha256": identity,
            "task": source_row["task"],
            "question": source_row["source_prompt"],
            "correctness_pattern": pattern,
            "candidates": [
                {
                    "lineage": lineage,
                    "completion": candidate["completion"],
                    "correct": correct,
                    "generated_tokens": candidate["generated_tokens"],
                    "max_token_exhausted": candidate["max_token_exhausted"],
                }
                for lineage, candidate, correct in zip(
                    router.LINEAGES, candidates, correctness, strict=True
                )
            ],
            "_features": [
                router.candidate_features(
                    source_row["source_prompt"],
                    source_row["task"],
                    lineage,
                    candidate,
                )
                for lineage, candidate in zip(router.LINEAGES, candidates, strict=True)
            ],
        }
        rows.append(row)
        by_identity[identity] = row
        patterns[pattern] += 1

    outcomes: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for fold in range(FOLDS):
        training = [
            row for row in rows if reference_folds[row["identity_sha256"]] != fold
        ]
        held_out = [
            row for row in rows if reference_folds[row["identity_sha256"]] == fold
        ]
        weights, _ = router._fit(
            training,
            learning_rate=LEARNING_RATE,
            balanced=BALANCED_PATTERNS,
            epochs=EPOCHS,
        )
        fold_correct = 0
        for row in held_out:
            values = [router._score(weights, features) for features in row["_features"]]
            selected = max(range(3), key=lambda index: (values[index], -index))
            heuristic = _heuristic(
                [group[row["identity_sha256"]] for group in candidate_groups]
            )
            selected_correct = bool(row["candidates"][selected]["correct"])
            heuristic_correct = bool(row["candidates"][heuristic]["correct"])
            fold_correct += int(selected_correct)
            outcomes.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "task": row["task"],
                    "fold": fold,
                    "correctness_pattern": row["correctness_pattern"],
                    "selected_lineage": router.LINEAGES[selected],
                    "scores": values,
                    "correct": selected_correct,
                    "heuristic_lineage": router.LINEAGES[heuristic],
                    "heuristic_correct": heuristic_correct,
                }
            )
        folds.append(
            {
                "fold": fold,
                "training_rows": len(training),
                "held_out_rows": len(held_out),
                "held_out_correct": fold_correct,
            }
        )
    if (
        len(outcomes) != router.DEVELOPMENT_ROWS
        or len({row["identity_sha256"] for row in outcomes}) != router.DEVELOPMENT_ROWS
    ):
        raise Q36MTRSparseCrossValidationError(
            "sparse cross-validation output coverage differs"
        )
    outcomes.sort(key=lambda row: row["identity_sha256"])
    sparse_correct = sum(row["correct"] for row in outcomes)
    heuristic_correct = sum(row["heuristic_correct"] for row in outcomes)
    sparse_only = sum(
        row["correct"] and not row["heuristic_correct"] for row in outcomes
    )
    heuristic_only = sum(
        row["heuristic_correct"] and not row["correct"] for row in outcomes
    )
    result = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "interpretation": "engineering_16_fold_out_of_shard_cross_validation",
        "rows": len(outcomes),
        "training_labels_exclude_held_out_fold": True,
        "folds": folds,
        "training": {
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "balanced_patterns": BALANCED_PATTERNS,
            "dimension": router.DIMENSION,
            "feature_contract": "hashed_question_trajectory_owner_interactions_v1",
        },
        "correctness_patterns": dict(sorted(patterns.items())),
        "sparse_router": {
            "correct": sparse_correct,
            "accuracy": sparse_correct / len(outcomes),
            "domains": {
                task: {
                    "rows": sum(row["task"] == task for row in outcomes),
                    "correct": sum(
                        row["task"] == task and row["correct"] for row in outcomes
                    ),
                }
                for task in router.TASKS
            },
        },
        "candidate_only_heuristic": {
            "correct": heuristic_correct,
            "accuracy": heuristic_correct / len(outcomes),
        },
        "paired_vs_candidate_only": {
            "both_correct": sum(
                row["correct"] and row["heuristic_correct"] for row in outcomes
            ),
            "sparse_only_correct": sparse_only,
            "heuristic_only_correct": heuristic_only,
            "both_wrong": sum(
                not row["correct"] and not row["heuristic_correct"] for row in outcomes
            ),
            "mcnemar_exact_two_sided_p": _mcnemar(sparse_only, heuristic_only),
        },
        "outcomes": outcomes,
        "input_sha256": {
            "development_source": router.sha256_file(args.development_source),
            "candidate_files": {
                lineage: [router.sha256_file(path) for path in paths]
                for lineage, paths in zip(router.LINEAGES, candidate_paths, strict=True)
            },
            "score_files": {
                lineage: [router.sha256_file(path) for path in paths]
                for lineage, paths in zip(router.LINEAGES, score_paths, strict=True)
            },
        },
    }
    _atomic_json(args.output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-source", type=Path, required=True)
    for owner in ("current", "owner71", "owner8"):
        parser.add_argument(
            f"--{owner}-candidates", type=Path, action="append", required=True
        )
        parser.add_argument(
            f"--{owner}-scores", type=Path, action="append", required=True
        )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    payload = cross_validate(parse_args())
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
