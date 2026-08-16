#!/usr/bin/env python3
"""Train a cross-fitted calibration-only three-owner Q36 commit head."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

import train_apply_q36_mtr_calibration_stack as stack
import train_apply_q36_mtr_sparse_router as sparse

MODEL_SCHEMA = "shohin-q36-mtr-calibration-direct-commit-model-v1"
REPORT_SCHEMA = "shohin-q36-mtr-calibration-direct-commit-report-v1"
SELECTION_SCHEMA = "shohin-q36-mtr-calibration-direct-commit-selection-v1"
REGULARIZATION = 0.2
LEARNING_RATE = 0.15
STEPS = 1_000
THRESHOLDS = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 1.1)


class Q36MTRCalibrationDirectCommitError(RuntimeError):
    """Direct-commit inputs, cross-fitting, or application differs."""


def _categorical(row: dict[str, Any]) -> set[str]:
    scores = row["scores"]
    ordered = sorted(scores, reverse=True)
    margin = ordered[0] - ordered[1]
    margin_bin = 0 if margin < 0.1 else 1 if margin < 0.2 else 2 if margin < 0.5 else 3
    return {
        f"task={row['task']}",
        f"sparse={row['selected_lineage']}",
        f"production={row['production_commit_lineage']}",
        f"pair={row['task']}:{row['selected_lineage']}:{row['production_commit_lineage']}",
        f"margin={margin_bin}",
        f"task_margin={row['task']}:{margin_bin}",
    }


def _vocabulary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        name: index
        for index, name in enumerate(
            sorted({name for row in rows for name in _categorical(row)})
        )
    }


def _vector(row: dict[str, Any], vocabulary: dict[str, int]) -> np.ndarray:
    vector = np.zeros(len(vocabulary) + 11, dtype=np.float64)
    for name in _categorical(row):
        if name in vocabulary:
            vector[vocabulary[name]] = 1.0
    scores = np.asarray(row["scores"], dtype=np.float64)
    centered = scores - scores.mean()
    ordered = sorted(scores, reverse=True)
    vector[len(vocabulary) :] = (
        1.0,
        *scores,
        *centered,
        ordered[0] - ordered[1],
        float(scores.std()),
        float(sparse.LINEAGES.index(row["selected_lineage"])),
        float(sparse.LINEAGES.index(row["production_commit_lineage"])),
    )
    return vector


def _softmax(matrix: np.ndarray) -> np.ndarray:
    shifted = matrix - matrix.max(axis=1, keepdims=True)
    values = np.exp(np.clip(shifted, -40.0, 0.0))
    return values / values.sum(axis=1, keepdims=True)


def _fit(
    rows: list[dict[str, Any]], vocabulary: dict[str, int]
) -> tuple[np.ndarray, dict[str, Any]]:
    mixed = [
        row
        for row in rows
        if 0 < sum(row["candidate_correctness"]) < len(sparse.LINEAGES)
    ]
    if len(mixed) < 100:
        raise Q36MTRCalibrationDirectCommitError(
            "direct-commit target geometry differs"
        )
    features = np.stack([_vector(row, vocabulary) for row in mixed])
    targets = np.asarray(
        [
            np.asarray(row["candidate_correctness"], dtype=np.float64)
            / sum(row["candidate_correctness"])
            for row in mixed
        ],
        dtype=np.float64,
    )
    weights = np.zeros((features.shape[1], len(sparse.LINEAGES)), dtype=np.float64)
    for _ in range(STEPS):
        probabilities = _softmax(features @ weights)
        gradient = features.T @ (probabilities - targets) / len(mixed)
        gradient += REGULARIZATION * weights / len(mixed)
        weights -= LEARNING_RATE * gradient
    if not np.isfinite(weights).all():
        raise Q36MTRCalibrationDirectCommitError("direct-commit weights differ")
    probabilities = _softmax(features @ weights)
    selected = probabilities.argmax(axis=1)
    correct = sum(
        bool(row["candidate_correctness"][int(index)])
        for row, index in zip(mixed, selected, strict=True)
    )
    return weights, {
        "rows": len(mixed),
        "selection_correct": correct,
        "regularization": REGULARIZATION,
        "learning_rate": LEARNING_RATE,
        "steps": STEPS,
        "weight_l2": float(np.linalg.norm(weights)),
    }


def _probabilities(
    row: dict[str, Any], vocabulary: dict[str, int], weights: np.ndarray
) -> list[float]:
    values = _softmax((_vector(row, vocabulary) @ weights)[None, :])[0]
    return [float(value) for value in values]


def _cross_fit(
    outcomes: list[dict[str, Any]], vocabulary: dict[str, int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    for fold in range(stack.FOLDS):
        training = [row for row in outcomes if row["fold"] != fold]
        held_out = [row for row in outcomes if row["fold"] == fold]
        weights, fit_report = _fit(training, vocabulary)
        for row in held_out:
            probabilities = _probabilities(row, vocabulary, weights)
            direct_index = max(
                range(len(probabilities)),
                key=lambda index: (probabilities[index], -index),
            )
            production_index = sparse.LINEAGES.index(row["production_commit_lineage"])
            ordered = sorted(probabilities, reverse=True)
            predictions.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "fold": fold,
                    "task": row["task"],
                    "probabilities": probabilities,
                    "confidence_margin": ordered[0] - ordered[1],
                    "direct_index": direct_index,
                    "production_index": production_index,
                    "candidate_correctness": row["candidate_correctness"],
                }
            )
        folds.append(
            {
                "fold": fold,
                "training_rows": len(training),
                "held_out_rows": len(held_out),
                "fit": fit_report,
            }
        )
    if len(predictions) != len(outcomes) or len(
        {row["identity_sha256"] for row in predictions}
    ) != len(outcomes):
        raise Q36MTRCalibrationDirectCommitError(
            "direct-commit cross-fit coverage differs"
        )
    return sorted(predictions, key=lambda row: row["identity_sha256"]), folds


def _threshold_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    correct = 0
    interventions = 0
    regressions = 0
    retained = 0
    production_correct = 0
    for row in rows:
        production_index = row["production_index"]
        use_direct = (
            row["direct_index"] != production_index
            and row["confidence_margin"] >= threshold
        )
        selected = row["direct_index"] if use_direct else production_index
        candidate_correctness = row["candidate_correctness"]
        is_production_correct = bool(candidate_correctness[production_index])
        is_selected_correct = bool(candidate_correctness[selected])
        correct += int(is_selected_correct)
        production_correct += int(is_production_correct)
        retained += int(is_production_correct and is_selected_correct)
        regressions += int(is_production_correct and not is_selected_correct)
        interventions += int(use_direct)
    return {
        "threshold": threshold,
        "rows": len(rows),
        "correct": correct,
        "production_correct": production_correct,
        "interventions": interventions,
        "regressions": regressions,
        "production_correct_retained": retained,
        "retention_rate": retained / production_correct if production_correct else 1.0,
    }


def _choose_threshold(rows: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    trials = [_threshold_metrics(rows, threshold) for threshold in THRESHOLDS]
    best = max(
        trials,
        key=lambda trial: (
            trial["correct"],
            -trial["regressions"],
            -trial["interventions"],
            trial["threshold"],
        ),
    )
    return float(best["threshold"]), trials


def run(args: argparse.Namespace) -> dict[str, Any]:
    training_rows = sparse.load_training_rows(args.training_rows)
    development_rows = sparse.load_development_rows(args.development_rows)
    owners = [
        sparse.load_development_candidates(paths)
        for paths in (
            args.current_candidates,
            args.owner71_candidates,
            args.owner8_candidates,
        )
    ]
    if any(set(owner) != set(development_rows) for owner in owners):
        raise Q36MTRCalibrationDirectCommitError("direct-commit owner coverage differs")

    oof_outcomes, sparse_oof_report = stack._oof_sparse(training_rows)
    vocabulary = _vocabulary(oof_outcomes)
    cross_fitted, fold_reports = _cross_fit(oof_outcomes, vocabulary)
    thresholds: dict[str, float] = {}
    threshold_trials: dict[str, list[dict[str, Any]]] = {}
    for task in sparse.TASKS:
        task_rows = [row for row in cross_fitted if row["task"] == task]
        threshold, trials = _choose_threshold(task_rows)
        thresholds[task] = threshold
        threshold_trials[task] = trials

    final_meta_weights, final_meta_report = _fit(oof_outcomes, vocabulary)
    final_sparse_weights, sparse_history = sparse._fit(
        training_rows,
        learning_rate=stack.SPARSE_LEARNING_RATE,
        balanced=stack.SPARSE_BALANCED_PATTERNS,
        epochs=stack.SPARSE_EPOCHS,
    )
    stack._strip_features(training_rows)
    model = {
        "schema": MODEL_SCHEMA,
        "status": "complete",
        "training_rows_sha256": sparse.sha256_file(args.training_rows),
        "development_labels_read": 0,
        "sparse_oof": sparse_oof_report,
        "cross_fitted_meta": {"folds": fold_reports, "rows": len(cross_fitted)},
        "thresholds": thresholds,
        "threshold_trials": threshold_trials,
        "meta_vocabulary": vocabulary,
        "meta_weights": final_meta_weights.tolist(),
        "meta_training": final_meta_report,
        "sparse_training": {
            "epochs": stack.SPARSE_EPOCHS,
            "learning_rate": stack.SPARSE_LEARNING_RATE,
            "balanced_patterns": stack.SPARSE_BALANCED_PATTERNS,
            "history": sparse_history,
            "nonzero_weights": [
                [index, value]
                for index, value in enumerate(final_sparse_weights)
                if value != 0.0
            ],
        },
    }
    model_sha = sparse._atomic_json(args.model_output, model)

    selected_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for identity in sorted(development_rows):
        source = development_rows[identity]
        candidates = [owner[identity] for owner in owners]
        features = [
            sparse.candidate_features(
                source["question"], source["task"], lineage, candidate
            )
            for lineage, candidate in zip(sparse.LINEAGES, candidates, strict=True)
        ]
        scores = [sparse._score(final_sparse_weights, feature) for feature in features]
        sparse_index = max(range(3), key=lambda index: (scores[index], -index))
        production_index = stack._production_index(candidates, source["task"])
        row = {
            "task": source["task"],
            "selected_lineage": sparse.LINEAGES[sparse_index],
            "production_commit_lineage": sparse.LINEAGES[production_index],
            "scores": scores,
        }
        probabilities = _probabilities(row, vocabulary, final_meta_weights)
        direct_index = max(
            range(len(probabilities)), key=lambda index: (probabilities[index], -index)
        )
        ordered = sorted(probabilities, reverse=True)
        confidence_margin = ordered[0] - ordered[1]
        threshold = thresholds[source["task"]]
        use_direct = direct_index != production_index and confidence_margin >= threshold
        selected_index = direct_index if use_direct else production_index
        selected_source = "direct_commit" if use_direct else "production_commit"
        metadata = {
            "schema": SELECTION_SCHEMA,
            "selected_source": selected_source,
            "selected_lineage": sparse.LINEAGES[selected_index],
            "direct_lineage": sparse.LINEAGES[direct_index],
            "production_commit_lineage": sparse.LINEAGES[production_index],
            "sparse_lineage": sparse.LINEAGES[sparse_index],
            "probabilities": probabilities,
            "confidence_margin": confidence_margin,
            "threshold": threshold,
            "model_sha256": model_sha,
        }
        chosen = dict(candidates[selected_index])
        chosen["calibration_direct_commit_selection"] = metadata
        selected_rows.append(chosen)
        decisions.append(
            {"identity_sha256": identity, "task": source["task"], **metadata}
        )
        counts[selected_source] += 1
        counts[f"lineage:{sparse.LINEAGES[selected_index]}"] += 1

    output_sha = sparse._atomic_lines(args.output, selected_rows)
    decisions_sha = sparse._atomic_lines(args.decisions, decisions)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "interpretation": "calibration_only_cross_fitted_three_owner_commit",
        "rows": len(selected_rows),
        "development_labels_read": 0,
        "thresholds": thresholds,
        "selection_counts": dict(sorted(counts.items())),
        "model": str(args.model_output.resolve()),
        "model_sha256": model_sha,
        "output": str(args.output.resolve()),
        "output_sha256": output_sha,
        "decisions": str(args.decisions.resolve()),
        "decisions_sha256": decisions_sha,
        "training_rows_sha256": sparse.sha256_file(args.training_rows),
        "development_rows_sha256": sparse.sha256_file(args.development_rows),
        "owner_candidate_sha256": {
            lineage: [sparse.sha256_file(path) for path in paths]
            for lineage, paths in zip(
                sparse.LINEAGES,
                (
                    args.current_candidates,
                    args.owner71_candidates,
                    args.owner8_candidates,
                ),
                strict=True,
            )
        },
    }
    sparse._atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-rows", type=Path, required=True)
    parser.add_argument("--development-rows", type=Path, required=True)
    for owner in ("current", "owner71", "owner8"):
        parser.add_argument(
            f"--{owner}-candidates", type=Path, action="append", required=True
        )
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    payload = run(parse_args())
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
