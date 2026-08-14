#!/usr/bin/env python3
"""Train a calibration-only stacked Q36 trajectory router and apply it to development."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import select_q36_mtr_owner_trajectories as production
import train_apply_q36_mtr_sparse_router as sparse

MODEL_SCHEMA = "shohin-q36-mtr-calibration-stacked-router-model-v1"
REPORT_SCHEMA = "shohin-q36-mtr-calibration-stacked-router-report-v1"
SELECTION_SCHEMA = "shohin-q36-mtr-calibration-stacked-router-selection-v1"
FOLDS = 16
SPARSE_EPOCHS = 8
SPARSE_LEARNING_RATE = 0.07
SPARSE_BALANCED_PATTERNS = True
META_REGULARIZATION = 0.1
META_LEARNING_RATE = 0.2
META_STEPS = 800


class Q36MTRCalibrationStackError(RuntimeError):
    """Calibration-stack inputs, OOF training, or application differs."""


def _fold(identity: str) -> int:
    digest = hashlib.sha256(
        f"{sparse.SEED}\0calibration-stack\0{identity}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % FOLDS


def _production_index(candidates: list[dict[str, Any]], task: str) -> int:
    selected = 0
    prepared = []
    for candidate in candidates:
        item = dict(candidate)
        item["task"] = task
        prepared.append(item)
    for challenger in range(1, len(prepared)):
        choice, _ = production._choose(prepared[selected], prepared[challenger])
        if choice == "second":
            selected = challenger
    return selected


def _attach_features(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["_features"] = [
            sparse.candidate_features(row["question"], row["task"], lineage, candidate)
            for lineage, candidate in zip(
                sparse.LINEAGES, row["candidates"], strict=True
            )
        ]


def _oof_sparse(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _attach_features(rows)
    outcomes: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    for fold in range(FOLDS):
        training = [row for row in rows if _fold(row["identity_sha256"]) != fold]
        held_out = [row for row in rows if _fold(row["identity_sha256"]) == fold]
        if not training or not held_out:
            raise Q36MTRCalibrationStackError("calibration-stack fold geometry differs")
        weights, _ = sparse._fit(
            training,
            learning_rate=SPARSE_LEARNING_RATE,
            balanced=SPARSE_BALANCED_PATTERNS,
            epochs=SPARSE_EPOCHS,
        )
        selected_correct = 0
        production_correct = 0
        for row in held_out:
            scores = [sparse._score(weights, features) for features in row["_features"]]
            selected = max(range(3), key=lambda index: (scores[index], -index))
            production_selected = _production_index(row["candidates"], row["task"])
            correctness = [candidate["correct"] for candidate in row["candidates"]]
            selected_correct += int(correctness[selected])
            production_correct += int(correctness[production_selected])
            outcomes.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "fold": fold,
                    "task": row["task"],
                    "selected_lineage": sparse.LINEAGES[selected],
                    "scores": scores,
                    "correct": correctness[selected],
                    "production_commit_lineage": sparse.LINEAGES[production_selected],
                    "production_commit_correct": correctness[production_selected],
                }
            )
        fold_reports.append(
            {
                "fold": fold,
                "training_rows": len(training),
                "held_out_rows": len(held_out),
                "sparse_correct": selected_correct,
                "production_commit_correct": production_correct,
            }
        )
    if len(outcomes) != len(rows) or len(
        {row["identity_sha256"] for row in outcomes}
    ) != len(rows):
        raise Q36MTRCalibrationStackError("calibration-stack OOF coverage differs")
    outcomes.sort(key=lambda row: row["identity_sha256"])
    return outcomes, {
        "schema": "shohin-q36-mtr-calibration-oof-sparse-v1",
        "folds": fold_reports,
        "rows": len(outcomes),
        "sparse_correct": sum(row["correct"] for row in outcomes),
        "production_commit_correct": sum(
            row["production_commit_correct"] for row in outcomes
        ),
        "discordant_rows": sum(
            row["correct"] != row["production_commit_correct"] for row in outcomes
        ),
        "training_excludes_selected_identity_fold": True,
    }


def _margin_bin(scores: list[float]) -> int:
    ordered = sorted(scores, reverse=True)
    margin = ordered[0] - ordered[1]
    if margin < 0.1:
        return 0
    if margin < 0.2:
        return 1
    if margin < 0.5:
        return 2
    return 3


def _categorical(row: dict[str, Any]) -> set[str]:
    margin_bin = _margin_bin(row["scores"])
    return {
        f"task={row['task']}",
        f"sparse={row['selected_lineage']}",
        f"production={row['production_commit_lineage']}",
        f"pair={row['task']}:{row['selected_lineage']}:{row['production_commit_lineage']}",
        f"margin_bin={margin_bin}",
        f"task_margin={row['task']}:{margin_bin}",
    }


def _meta_vector(row: dict[str, Any], vocabulary: dict[str, int]) -> np.ndarray:
    vector = np.zeros(len(vocabulary) + 5, dtype=np.float64)
    for name in _categorical(row):
        if name in vocabulary:
            vector[vocabulary[name]] = 1.0
    scores = row["scores"]
    ordered = sorted(scores, reverse=True)
    vector[len(vocabulary) :] = (
        1.0,
        ordered[0] - ordered[1],
        max(scores),
        min(scores),
        float(np.std(scores)),
    )
    return vector


def _fit_meta(
    outcomes: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
    names = sorted({name for row in outcomes for name in _categorical(row)})
    vocabulary = {name: index for index, name in enumerate(names)}
    discordant = [
        row
        for row in outcomes
        if row["correct"] != row["production_commit_correct"] and row["task"] != "mbpp"
    ]
    if not discordant:
        raise Q36MTRCalibrationStackError("calibration-stack meta target is empty")
    features = np.stack([_meta_vector(row, vocabulary) for row in discordant])
    targets = np.asarray(
        [
            float(row["correct"] and not row["production_commit_correct"])
            for row in discordant
        ],
        dtype=np.float64,
    )
    weights = np.zeros(features.shape[1], dtype=np.float64)
    for _ in range(META_STEPS):
        logits = np.clip(features @ weights, -20.0, 20.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = features.T @ (probabilities - targets) / len(
            discordant
        ) + META_REGULARIZATION * weights / len(discordant)
        weights -= META_LEARNING_RATE * gradient
    if not np.isfinite(weights).all():
        raise Q36MTRCalibrationStackError("calibration-stack meta weights differ")
    training_predictions = features @ weights >= 0.0
    training_correct = sum(
        bool(prediction) == bool(target)
        for prediction, target in zip(training_predictions, targets, strict=True)
    )
    return (
        weights,
        vocabulary,
        {
            "schema": "shohin-q36-mtr-calibration-logistic-meta-v1",
            "discordant_training_rows": len(discordant),
            "training_class_sparse": int(targets.sum()),
            "training_class_production": len(targets) - int(targets.sum()),
            "training_classification_correct": training_correct,
            "regularization": META_REGULARIZATION,
            "learning_rate": META_LEARNING_RATE,
            "steps": META_STEPS,
            "features": names,
            "weight_l2": float(np.linalg.norm(weights)),
        },
    )


def _strip_features(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row.pop("_features", None)


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
        raise Q36MTRCalibrationStackError("calibration-stack owner coverage differs")

    oof_outcomes, oof_report = _oof_sparse(training_rows)
    meta_weights, vocabulary, meta_report = _fit_meta(oof_outcomes)
    final_sparse_weights, final_sparse_history = sparse._fit(
        training_rows,
        learning_rate=SPARSE_LEARNING_RATE,
        balanced=SPARSE_BALANCED_PATTERNS,
        epochs=SPARSE_EPOCHS,
    )
    _strip_features(training_rows)
    model = {
        "schema": MODEL_SCHEMA,
        "status": "complete",
        "training_rows_sha256": sparse.sha256_file(args.training_rows),
        "calibration_oof": oof_report,
        "meta_training": meta_report,
        "sparse_training": {
            "epochs": SPARSE_EPOCHS,
            "learning_rate": SPARSE_LEARNING_RATE,
            "balanced_patterns": SPARSE_BALANCED_PATTERNS,
            "history": final_sparse_history,
            "nonzero_weights": [
                [index, value]
                for index, value in enumerate(final_sparse_weights)
                if value != 0.0
            ],
        },
        "meta_vocabulary": vocabulary,
        "meta_weights": meta_weights.tolist(),
        "development_labels_read": 0,
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
        scores = [
            sparse._score(final_sparse_weights, candidate_features)
            for candidate_features in features
        ]
        sparse_index = max(range(3), key=lambda index: (scores[index], -index))
        production_index = _production_index(candidates, source["task"])
        meta_row = {
            "task": source["task"],
            "selected_lineage": sparse.LINEAGES[sparse_index],
            "production_commit_lineage": sparse.LINEAGES[production_index],
            "scores": scores,
        }
        use_sparse = (
            False
            if source["task"] == "mbpp"
            else bool(float(_meta_vector(meta_row, vocabulary) @ meta_weights) >= 0.0)
        )
        selected_index = sparse_index if use_sparse else production_index
        selected_source = "sparse_router" if use_sparse else "production_commit"
        chosen = dict(candidates[selected_index])
        chosen["calibration_stacked_selection"] = {
            "schema": SELECTION_SCHEMA,
            "selected_source": selected_source,
            "selected_lineage": sparse.LINEAGES[selected_index],
            "sparse_lineage": sparse.LINEAGES[sparse_index],
            "production_commit_lineage": sparse.LINEAGES[production_index],
            "scores": scores,
            "model_sha256": model_sha,
        }
        selected_rows.append(chosen)
        decisions.append(
            {
                "schema": SELECTION_SCHEMA,
                "identity_sha256": identity,
                "task": source["task"],
                "selected_source": selected_source,
                "selected_lineage": sparse.LINEAGES[selected_index],
                "sparse_lineage": sparse.LINEAGES[sparse_index],
                "production_commit_lineage": sparse.LINEAGES[production_index],
                "scores": scores,
                "model_sha256": model_sha,
            }
        )
        counts[selected_source] += 1
    output_sha = sparse._atomic_lines(args.output, selected_rows)
    decisions_sha = sparse._atomic_lines(args.decisions, decisions)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "interpretation": "calibration_only_deployable_stacked_trajectory_router",
        "rows": len(selected_rows),
        "development_labels_read": 0,
        "calibration_oof": oof_report,
        "meta_training": meta_report,
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
