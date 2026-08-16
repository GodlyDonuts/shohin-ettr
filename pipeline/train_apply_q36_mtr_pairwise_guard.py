#!/usr/bin/env python3
"""Train a text-aware pairwise guard over the Q36 correctness champion."""

from __future__ import annotations

import argparse
from array import array
from collections import Counter
import json
import math
from pathlib import Path
import random
from typing import Any

import train_apply_q36_mtr_calibration_correctness as correctness
import train_apply_q36_mtr_sparse_router as sparse

MODEL_SCHEMA = "shohin-q36-mtr-pairwise-guard-model-v1"
REPORT_SCHEMA = "shohin-q36-mtr-pairwise-guard-report-v1"
SELECTION_SCHEMA = "shohin-q36-mtr-pairwise-guard-selection-v1"
LEARNING_RATES = (0.03, 0.1)
EPOCHS = (4, 8, 12)
BALANCED = (False, True)
THRESHOLDS = (0.25, 0.35, 0.45, 0.5, 0.55, 0.6, 0.65, 0.75, 1.1)
THRESHOLD_MODES = ("task", "task_pair")
MINIMUM_PAIR_GROUP = 20


class Q36MTRPairwiseGuardError(RuntimeError):
    """Pairwise guard inputs, training, or application differs."""


def _pair_features(
    row: dict[str, Any], challenger: int, production: int
) -> dict[int, float]:
    features: dict[int, float] = {}
    for index, value in row["_features"][challenger].items():
        features[index] = features.get(index, 0.0) + value
    for index, value in row["_features"][production].items():
        features[index] = features.get(index, 0.0) - value
    for name in (
        "bias",
        f"task:{row['task']}",
        f"pair:{challenger}:{production}",
        f"task_pair:{row['task']}:{challenger}:{production}",
    ):
        sparse._add(features, f"pair_guard:{name}")
    return {index: value for index, value in features.items() if value != 0.0}


def _pair_examples(
    rows: list[dict[str, Any]],
) -> list[tuple[dict[int, float], float]]:
    examples: list[tuple[dict[int, float], float]] = []
    for row in rows:
        production = correctness._production_index(row)
        production_correct = bool(row["candidates"][production]["correct"])
        for challenger in range(len(sparse.LINEAGES)):
            if challenger == production:
                continue
            challenger_correct = bool(row["candidates"][challenger]["correct"])
            if challenger_correct == production_correct:
                continue
            examples.append(
                (
                    _pair_features(row, challenger, production),
                    float(challenger_correct),
                )
            )
    return examples


def _fit(
    rows: list[dict[str, Any]],
    *,
    learning_rate: float,
    epochs: int,
    balanced: bool,
) -> tuple[array, dict[str, Any]]:
    examples = _pair_examples(rows)
    positives = sum(int(target) for _, target in examples)
    negatives = len(examples) - positives
    if len(examples) < 100 or positives == 0 or negatives == 0:
        raise Q36MTRPairwiseGuardError("pairwise target geometry differs")
    weights = array("d", [0.0]) * sparse.DIMENSION
    accumulators = array("d", [1e-6]) * sparse.DIMENSION
    order = list(range(len(examples)))
    generator = random.Random(sparse.SEED + 17)
    for _ in range(epochs):
        generator.shuffle(order)
        for example_index in order:
            features, target = examples[example_index]
            probability = correctness._sigmoid(sparse._score(weights, features))
            row_weight = 1.0
            if balanced:
                denominator = positives if target else negatives
                row_weight = len(examples) / (2.0 * denominator)
            coefficient = row_weight * (target - probability)
            for feature_index, value in features.items():
                gradient = coefficient * value
                accumulators[feature_index] += gradient * gradient
                weights[feature_index] += (
                    learning_rate * gradient / math.sqrt(accumulators[feature_index])
                )
    return weights, {
        "rows": len(rows),
        "examples": len(examples),
        "positives": positives,
        "negatives": negatives,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "balanced": balanced,
    }


def _probability(
    weights: array, row: dict[str, Any], challenger: int, production: int
) -> float:
    return correctness._sigmoid(
        sparse._score(weights, _pair_features(row, challenger, production))
    )


def _predictions(
    pair_weights: array,
    absolute_weights: list[array],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    predictions = []
    for row in rows:
        probabilities = correctness._probabilities(absolute_weights, row)
        head = max(
            range(len(probabilities)),
            key=lambda index: (probabilities[index], -index),
        )
        production = correctness._production_index(row)
        pair_probability = (
            _probability(pair_weights, row, head, production)
            if head != production
            else 0.0
        )
        predictions.append(
            {
                "task": row["task"],
                "head_index": head,
                "production_index": production,
                "pair_probability": pair_probability,
                "correctness": [
                    bool(candidate["correct"]) for candidate in row["candidates"]
                ],
            }
        )
    return predictions


def _metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    correct = production_correct = regressions = interventions = 0
    for row in rows:
        use_head = (
            row["head_index"] != row["production_index"]
            and row["pair_probability"] >= threshold
        )
        selected = row["head_index"] if use_head else row["production_index"]
        selected_correct = bool(row["correctness"][selected])
        baseline_correct = bool(row["correctness"][row["production_index"]])
        correct += int(selected_correct)
        production_correct += int(baseline_correct)
        regressions += int(baseline_correct and not selected_correct)
        interventions += int(use_head)
    return {
        "threshold": threshold,
        "correct": correct,
        "production_correct": production_correct,
        "regressions": regressions,
        "interventions": interventions,
        "rows": len(rows),
    }


def _choose_threshold(
    rows: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    trials = [_metrics(rows, threshold) for threshold in THRESHOLDS]
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


def _group(row: dict[str, Any], mode: str) -> str:
    if mode == "task":
        return row["task"]
    if mode == "task_pair":
        return f"{row['task']}:{row['head_index']}:{row['production_index']}"
    raise Q36MTRPairwiseGuardError("pairwise threshold mode differs")


def _threshold_map(
    rows: list[dict[str, Any]], mode: str
) -> tuple[dict[str, float], dict[str, list[dict[str, Any]]]]:
    thresholds = {}
    trials = {}
    for group in sorted({_group(row, mode) for row in rows}):
        grouped = [row for row in rows if _group(row, mode) == group]
        if mode != "task" and len(grouped) < MINIMUM_PAIR_GROUP:
            continue
        threshold, group_trials = _choose_threshold(grouped)
        thresholds[group] = threshold
        trials[group] = group_trials
    return thresholds, trials


def _threshold_for(
    row: dict[str, Any], mode: str, thresholds: dict[str, float]
) -> float:
    return thresholds.get(_group(row, mode), thresholds[row["task"]])


def _mapped_metrics(
    rows: list[dict[str, Any]], mode: str, thresholds: dict[str, float]
) -> dict[str, Any]:
    selected = []
    for row in rows:
        item = dict(row)
        item["mapped_threshold"] = _threshold_for(row, mode, thresholds)
        selected.append(item)
    correct = production_correct = regressions = interventions = 0
    for row in selected:
        use_head = (
            row["head_index"] != row["production_index"]
            and row["pair_probability"] >= row["mapped_threshold"]
        )
        selected_index = row["head_index"] if use_head else row["production_index"]
        chosen = bool(row["correctness"][selected_index])
        baseline = bool(row["correctness"][row["production_index"]])
        correct += int(chosen)
        production_correct += int(baseline)
        regressions += int(baseline and not chosen)
        interventions += int(use_head)
    return {
        "mode": mode,
        "correct": correct,
        "production_correct": production_correct,
        "regressions": regressions,
        "interventions": interventions,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = sparse.load_training_rows(args.training_rows)
    development = sparse.load_development_rows(args.development_rows)
    correctness._attach_features(rows)
    fit_rows = [row for row in rows if row["split"] == "calibration_train"]
    validation_rows = [row for row in rows if row["split"] == "calibration_development"]
    absolute_weights, absolute_fit = correctness._fit(
        fit_rows, learning_rate=0.15, epochs=12, balanced=False
    )
    trials = []
    best = None
    for balanced in BALANCED:
        for learning_rate in LEARNING_RATES:
            for epochs in EPOCHS:
                pair_weights, fit_report = _fit(
                    fit_rows,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    balanced=balanced,
                )
                validation = _predictions(
                    pair_weights, absolute_weights, validation_rows
                )
                task_thresholds, task_trials = _threshold_map(validation, "task")
                candidates = []
                all_trials = {"task": task_trials}
                for mode in THRESHOLD_MODES:
                    if mode == "task":
                        thresholds = dict(task_thresholds)
                    else:
                        grouped, grouped_trials = _threshold_map(validation, mode)
                        thresholds = {**task_thresholds, **grouped}
                        all_trials[mode] = grouped_trials
                    metrics = _mapped_metrics(validation, mode, thresholds)
                    candidates.append(
                        {**metrics, "thresholds": dict(sorted(thresholds.items()))}
                    )
                policy = max(
                    candidates,
                    key=lambda item: (
                        item["correct"],
                        -item["regressions"],
                        -item["interventions"],
                        -THRESHOLD_MODES.index(item["mode"]),
                    ),
                )
                trial = {
                    **fit_report,
                    "policy": policy,
                    "threshold_trials": all_trials,
                }
                trials.append(trial)
                key = (
                    policy["correct"],
                    -policy["regressions"],
                    -policy["interventions"],
                    -float(balanced),
                    -learning_rate,
                    -epochs,
                )
                if best is None or key > best[0]:
                    best = (key, trial)
    if best is None:
        raise Q36MTRPairwiseGuardError("pairwise search differs")
    selected = best[1]
    final_weights, final_fit = _fit(
        rows,
        learning_rate=selected["learning_rate"],
        epochs=selected["epochs"],
        balanced=selected["balanced"],
    )
    policy = selected["policy"]
    model = {
        "schema": MODEL_SCHEMA,
        "status": "complete",
        "development_labels_read": 0,
        "training_rows_sha256": sparse.sha256_file(args.training_rows),
        "absolute_fit": absolute_fit,
        "search_trials": trials,
        "selected_fit": {
            key: selected[key] for key in ("learning_rate", "epochs", "balanced")
        },
        "threshold_mode": policy["mode"],
        "thresholds": policy["thresholds"],
        "final_fit": final_fit,
        "nonzero_weights": [
            [index, value] for index, value in enumerate(final_weights) if value != 0.0
        ],
    }
    model_sha = sparse._atomic_json(args.model_output, model)

    owners = correctness._embedded_development_owners(development)
    reused = correctness._load_reused_development_decisions(
        args.reuse_development_decisions
    )
    if set(reused) != set(development):
        raise Q36MTRPairwiseGuardError("pairwise development coverage differs")
    selected_rows = []
    production_rows = []
    decisions = []
    counts: Counter[str] = Counter()
    for identity in sorted(development):
        source = development[identity]
        candidates = [owner[identity] for owner in owners]
        row = {
            "question": source["question"],
            "task": source["task"],
            "candidates": candidates,
        }
        correctness._attach_features([row])
        exact = reused[identity]
        head = sparse.LINEAGES.index(exact["head_lineage"])
        production = sparse.LINEAGES.index(exact["production_commit_lineage"])
        pair_probability = (
            _probability(final_weights, row, head, production)
            if head != production
            else 0.0
        )
        threshold_row = {
            "task": source["task"],
            "head_index": head,
            "production_index": production,
        }
        threshold = _threshold_for(threshold_row, policy["mode"], policy["thresholds"])
        use_head = head != production and pair_probability >= threshold
        selected_index = head if use_head else production
        selected_source = "pairwise_guard" if use_head else "production_commit"
        metadata = {
            "schema": SELECTION_SCHEMA,
            "selected_source": selected_source,
            "selected_lineage": sparse.LINEAGES[selected_index],
            "head_lineage": sparse.LINEAGES[head],
            "production_commit_lineage": sparse.LINEAGES[production],
            "pair_probability": pair_probability,
            "threshold": threshold,
            "model_sha256": model_sha,
        }
        chosen = dict(candidates[selected_index])
        chosen["pairwise_guard_selection"] = metadata
        selected_rows.append(chosen)
        production_rows.append(dict(candidates[production]))
        decisions.append(
            {"identity_sha256": identity, "task": source["task"], **metadata}
        )
        counts[selected_source] += 1
        counts[f"lineage:{sparse.LINEAGES[selected_index]}"] += 1
    output_sha = sparse._atomic_lines(args.output, selected_rows)
    production_sha = sparse._atomic_lines(args.production_output, production_rows)
    decisions_sha = sparse._atomic_lines(args.decisions, decisions)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "interpretation": "calibration_only_text_aware_pairwise_guard",
        "rows": len(selected_rows),
        "development_labels_read": 0,
        "threshold_mode": policy["mode"],
        "thresholds": policy["thresholds"],
        "selection_counts": dict(sorted(counts.items())),
        "model_sha256": model_sha,
        "output_sha256": output_sha,
        "production_output_sha256": production_sha,
        "decisions_sha256": decisions_sha,
        "training_rows_sha256": sparse.sha256_file(args.training_rows),
        "development_rows_sha256": sparse.sha256_file(args.development_rows),
        "reused_development_decisions_sha256": sparse.sha256_file(
            args.reuse_development_decisions
        ),
    }
    sparse._atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-rows", type=Path, required=True)
    parser.add_argument("--development-rows", type=Path, required=True)
    parser.add_argument("--reuse-development-decisions", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--production-output", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(run(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
