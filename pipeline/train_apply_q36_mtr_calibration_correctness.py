#!/usr/bin/env python3
"""Train absolute owner-correctness heads and conservatively commit a Q36 trajectory."""

from __future__ import annotations

import argparse
from array import array
from collections import Counter
import json
import math
from pathlib import Path
import random
from typing import Any

import train_apply_q36_mtr_calibration_stack as stack
import train_apply_q36_mtr_sparse_router as sparse

MODEL_SCHEMA = "shohin-q36-mtr-calibration-correctness-model-v1"
REPORT_SCHEMA = "shohin-q36-mtr-calibration-correctness-report-v1"
SELECTION_SCHEMA = "shohin-q36-mtr-calibration-correctness-selection-v1"
LEARNING_RATES = (0.03, 0.07, 0.15)
EPOCHS = (4, 8, 12)
BALANCED = (False, True)
THRESHOLDS = (0.0, 0.01, 0.02, 0.04, 0.06, 0.1, 0.15, 0.25, 1.1)


class Q36MTRCalibrationCorrectnessError(RuntimeError):
    """Absolute-correctness inputs, training, or application differs."""


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))


def _attach_features(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["_features"] = [
            sparse.candidate_features(row["question"], row["task"], lineage, candidate)
            for lineage, candidate in zip(
                sparse.LINEAGES, row["candidates"], strict=True
            )
        ]


def _fit(
    rows: list[dict[str, Any]],
    *,
    learning_rate: float,
    epochs: int,
    balanced: bool,
) -> tuple[list[array], dict[str, Any]]:
    weights = [array("d", [0.0]) * sparse.DIMENSION for _ in sparse.LINEAGES]
    accumulators = [array("d", [1e-6]) * sparse.DIMENSION for _ in sparse.LINEAGES]
    positives = [
        sum(bool(row["candidates"][index]["correct"]) for row in rows)
        for index in range(len(sparse.LINEAGES))
    ]
    negatives = [len(rows) - count for count in positives]
    if any(count == 0 for count in positives + negatives):
        raise Q36MTRCalibrationCorrectnessError(
            "correctness-head target geometry differs"
        )
    generator = random.Random(sparse.SEED)
    order = list(range(len(rows)))
    updates = 0
    for _ in range(epochs):
        generator.shuffle(order)
        for row_index in order:
            row = rows[row_index]
            for owner_index, features in enumerate(row["_features"]):
                target = float(row["candidates"][owner_index]["correct"])
                probability = _sigmoid(sparse._score(weights[owner_index], features))
                row_weight = 1.0
                if balanced:
                    denominator = (
                        positives[owner_index] if target else negatives[owner_index]
                    )
                    row_weight = len(rows) / (2.0 * denominator)
                coefficient = row_weight * (target - probability)
                for feature_index, value in features.items():
                    gradient = coefficient * value
                    accumulators[owner_index][feature_index] += gradient * gradient
                    weights[owner_index][feature_index] += (
                        learning_rate
                        * gradient
                        / math.sqrt(accumulators[owner_index][feature_index])
                    )
                updates += 1
    return weights, {
        "rows": len(rows),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "balanced": balanced,
        "positives": positives,
        "negatives": negatives,
        "updates": updates,
    }


def _probabilities(weights: list[array], row: dict[str, Any]) -> list[float]:
    return [
        _sigmoid(sparse._score(owner_weights, features))
        for owner_weights, features in zip(weights, row["_features"], strict=True)
    ]


def _production_index(row: dict[str, Any]) -> int:
    return stack._production_index(row["candidates"], row["task"])


def _validation_correct(weights: list[array], rows: list[dict[str, Any]]) -> int:
    return sum(
        bool(
            row["candidates"][
                max(
                    range(len(sparse.LINEAGES)),
                    key=lambda index: (_probabilities(weights, row)[index], -index),
                )
            ]["correct"]
        )
        for row in rows
    )


def _threshold_metrics(
    predictions: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    correct = production_correct = retained = regressions = interventions = 0
    for row in predictions:
        use_head = (
            row["head_index"] != row["production_index"]
            and row["estimated_gain"] >= threshold
        )
        selected = row["head_index"] if use_head else row["production_index"]
        selected_correct = bool(row["correctness"][selected])
        baseline_correct = bool(row["correctness"][row["production_index"]])
        correct += int(selected_correct)
        production_correct += int(baseline_correct)
        retained += int(selected_correct and baseline_correct)
        regressions += int(baseline_correct and not selected_correct)
        interventions += int(use_head)
    return {
        "threshold": threshold,
        "rows": len(predictions),
        "correct": correct,
        "production_correct": production_correct,
        "interventions": interventions,
        "regressions": regressions,
        "retained": retained,
        "retention_rate": retained / production_correct if production_correct else 1.0,
    }


def _choose_threshold(
    predictions: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    trials = [_threshold_metrics(predictions, threshold) for threshold in THRESHOLDS]
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
    _attach_features(training_rows)
    fit_rows = [row for row in training_rows if row["split"] == "calibration_train"]
    validation_rows = [
        row for row in training_rows if row["split"] == "calibration_development"
    ]
    trials: list[dict[str, Any]] = []
    best: tuple[int, float, bool, int, list[array]] | None = None
    for balanced in BALANCED:
        for learning_rate in LEARNING_RATES:
            for epochs in EPOCHS:
                weights, fit_report = _fit(
                    fit_rows,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    balanced=balanced,
                )
                correct = _validation_correct(weights, validation_rows)
                trial = {**fit_report, "calibration_development_correct": correct}
                trials.append(trial)
                key = (correct, -float(balanced), -learning_rate, -epochs)
                if best is None or key > best[:4]:
                    best = (*key, weights)
    if best is None:
        raise Q36MTRCalibrationCorrectnessError("correctness-head search differs")
    selected = max(
        trials,
        key=lambda trial: (
            trial["calibration_development_correct"],
            -float(trial["balanced"]),
            -trial["learning_rate"],
            -trial["epochs"],
        ),
    )
    selected_weights = best[4]
    validation_predictions: list[dict[str, Any]] = []
    for row in validation_rows:
        probabilities = _probabilities(selected_weights, row)
        head_index = max(
            range(len(probabilities)), key=lambda index: (probabilities[index], -index)
        )
        production_index = _production_index(row)
        validation_predictions.append(
            {
                "task": row["task"],
                "head_index": head_index,
                "production_index": production_index,
                "estimated_gain": probabilities[head_index]
                - probabilities[production_index],
                "correctness": [
                    bool(candidate["correct"]) for candidate in row["candidates"]
                ],
            }
        )
    thresholds: dict[str, float] = {}
    threshold_trials: dict[str, list[dict[str, Any]]] = {}
    for task in sparse.TASKS:
        threshold, task_trials = _choose_threshold(
            [row for row in validation_predictions if row["task"] == task]
        )
        thresholds[task] = threshold
        threshold_trials[task] = task_trials

    final_weights, final_fit = _fit(
        training_rows,
        learning_rate=selected["learning_rate"],
        epochs=selected["epochs"],
        balanced=selected["balanced"],
    )
    model = {
        "schema": MODEL_SCHEMA,
        "status": "complete",
        "development_labels_read": 0,
        "training_rows_sha256": sparse.sha256_file(args.training_rows),
        "model_selection_trials": trials,
        "selected_model": selected,
        "thresholds": thresholds,
        "threshold_trials": threshold_trials,
        "final_fit": final_fit,
        "nonzero_weights": [
            [[index, value] for index, value in enumerate(weights) if value != 0.0]
            for weights in final_weights
        ],
    }
    model_sha = sparse._atomic_json(args.model_output, model)
    owners = [
        sparse.load_development_candidates(paths)
        for paths in (
            args.current_candidates,
            args.owner71_candidates,
            args.owner8_candidates,
        )
    ]
    if any(set(owner) != set(development_rows) for owner in owners):
        raise Q36MTRCalibrationCorrectnessError("correctness-head coverage differs")
    selected_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for identity in sorted(development_rows):
        source = development_rows[identity]
        candidates = [owner[identity] for owner in owners]
        row = {
            "question": source["question"],
            "task": source["task"],
            "candidates": candidates,
        }
        _attach_features([row])
        probabilities = _probabilities(final_weights, row)
        head_index = max(
            range(len(probabilities)), key=lambda index: (probabilities[index], -index)
        )
        production_index = _production_index(row)
        estimated_gain = probabilities[head_index] - probabilities[production_index]
        threshold = thresholds[source["task"]]
        use_head = head_index != production_index and estimated_gain >= threshold
        selected_index = head_index if use_head else production_index
        selected_source = "correctness_head" if use_head else "production_commit"
        metadata = {
            "schema": SELECTION_SCHEMA,
            "selected_source": selected_source,
            "selected_lineage": sparse.LINEAGES[selected_index],
            "head_lineage": sparse.LINEAGES[head_index],
            "production_commit_lineage": sparse.LINEAGES[production_index],
            "probabilities": probabilities,
            "estimated_gain": estimated_gain,
            "threshold": threshold,
            "model_sha256": model_sha,
        }
        chosen = dict(candidates[selected_index])
        chosen["calibration_correctness_selection"] = metadata
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
        "interpretation": "calibration_only_absolute_owner_correctness_commit",
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
    print(json.dumps(run(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
