#!/usr/bin/env python3
"""Nested-cross-fit an unchanged-control override over Q36 forest consensus."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor

from hf_product_reasoning_eval import TASKS, has_explicit_final_answer
from select_q36_mtr_consensus import ARM_ORDER
from select_q36_mtr_interpolation_retention import (
    _atomic_json,
    _atomic_lines,
    sha256_file,
)
from select_q36_mtr_nested_forest_consensus import (
    _cluster_samples,
    _fit_forest,
    _load_inputs,
    _prediction_for_identity,
)
from select_q36_mtr_nested_pattern_consensus import (
    _choose as _pattern_choose,
    _select_outer_config,
)
from select_q36_mtr_pattern_consensus import _fit as _fit_pattern
from select_q36_mtr_reliability_consensus import SHARDS

ROWS = 1_289
CONTROL_SHARDS = 8
CONTROL_ARM = "unchanged"
SCHEMA = "shohin-q36-mtr-nested-control-forest-v1"
REPORT_SCHEMA = "shohin-q36-mtr-nested-control-forest-report-v1"
N_ESTIMATORS = 500
MIN_SAMPLES_LEAF = 10
MAX_FEATURES = 0.7
RANDOM_STATE = 2026081433
THRESHOLDS = tuple(round(-0.05 + index * 0.005, 3) for index in range(61))
FEATURE_NAMES = (
    "task_bbh_logic",
    "task_math500",
    "task_mbpp",
    "control_log_generated_tokens",
    "control_max_token_exhausted",
    "control_completion_length",
    "control_explicit_final_answer",
    "control_answer_vote_fraction",
    "forest_answer_vote_fraction",
    "control_minus_forest_vote_fraction",
    "control_answer_agreement_mask_fraction",
    "forest_answer_agreement_mask_fraction",
    "control_agrees_with_forest",
    "control_answer_missing",
    "forest_overrode_pattern",
    "forest_margin",
    "forest_probability",
    "pattern_probability",
    "pattern_estimated_reliability",
    "control_answer_present",
    "control_answer_length",
    "control_answer_numeric",
    "control_answer_negative",
    "control_answer_log_magnitude",
    "forest_answer_present",
    "forest_answer_length",
    "forest_answer_numeric",
    "forest_answer_negative",
    "forest_answer_log_magnitude",
)


class Q36MTRNestedControlForestError(RuntimeError):
    """Raised when control-aware forest inputs or fitting differ."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Q36MTRNestedControlForestError(f"unreadable JSON: {path}") from error
    if not isinstance(payload, dict):
        raise Q36MTRNestedControlForestError(f"JSON object differs: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise Q36MTRNestedControlForestError(f"unreadable JSONL: {path}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise Q36MTRNestedControlForestError(f"JSONL row differs: {path}")
    return rows


def _load_control(
    candidate_paths: list[Path], score_paths: list[Path]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if len(candidate_paths) != CONTROL_SHARDS or len(score_paths) != CONTROL_SHARDS:
        raise Q36MTRNestedControlForestError("control shard geometry differs")
    candidates: dict[str, dict[str, Any]] = {}
    for path in candidate_paths:
        for row in _load_jsonl(path):
            identity = row.get("identity_sha256")
            if (
                row.get("schema") != "shohin-q36-mtr-candidate-v1"
                or row.get("arm") != CONTROL_ARM
                or row.get("task") not in TASKS
                or not isinstance(identity, str)
                or len(identity) != 64
                or identity in candidates
                or not isinstance(row.get("completion"), str)
                or not row["completion"].strip()
                or not isinstance(row.get("generated_tokens"), int)
                or row["generated_tokens"] < 0
                or not isinstance(row.get("max_token_exhausted"), bool)
            ):
                raise Q36MTRNestedControlForestError("control candidate differs")
            candidates[identity] = row
    outcomes: dict[str, dict[str, Any]] = {}
    score_candidate_hashes = set()
    for path in score_paths:
        payload = _load_json(path)
        score_candidate_hashes.add(payload.get("candidates_sha256"))
        if (
            payload.get("schema") != "shohin-q36-mtr-draft-preview-v1"
            or payload.get("status") != "complete"
            or payload.get("evaluation_arm") != CONTROL_ARM
            or payload.get("split") != "development"
            or not isinstance(payload.get("outcomes"), list)
        ):
            raise Q36MTRNestedControlForestError("control score report differs")
        for row in payload["outcomes"]:
            identity = row.get("identity_sha256")
            if (
                not isinstance(identity, str)
                or identity in outcomes
                or not isinstance(row.get("correct"), bool)
                or not isinstance(row.get("explicit_final_answer"), bool)
                or not isinstance(row.get("max_token_exhausted"), bool)
                or row.get("task") not in TASKS
            ):
                raise Q36MTRNestedControlForestError("control score outcome differs")
            outcomes[identity] = row
    if score_candidate_hashes != {sha256_file(path) for path in candidate_paths}:
        raise Q36MTRNestedControlForestError("control score candidate custody differs")
    if set(candidates) != set(outcomes) or len(candidates) != ROWS:
        raise Q36MTRNestedControlForestError("control identity coverage differs")
    for identity, candidate in candidates.items():
        outcome = outcomes[identity]
        if (
            outcome["task"] != candidate["task"]
            or outcome["max_token_exhausted"] != candidate["max_token_exhausted"]
        ):
            raise Q36MTRNestedControlForestError("control outcome binding differs")
    return candidates, outcomes


def _load_forest(
    candidates_path: Path, report_path: Path, records: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    report = _load_json(report_path)
    if (
        report.get("schema") != "shohin-q36-mtr-nested-forest-consensus-report-v1"
        or report.get("status") != "complete"
        or report.get("rows") != ROWS
        or report.get("outer_shards") != SHARDS
        or report.get("output_sha256") != sha256_file(candidates_path)
        or not isinstance(report.get("outer_models"), dict)
    ):
        raise Q36MTRNestedControlForestError("forest report differs")
    candidates: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(candidates_path):
        identity = row.get("identity_sha256")
        state = row.get("nested_forest_consensus")
        if (
            not isinstance(identity, str)
            or identity in candidates
            or not isinstance(state, dict)
            or state.get("schema") != "shohin-q36-mtr-nested-forest-consensus-v1"
            or state.get("selected") not in ARM_ORDER
            or state.get("outer_shard") not in range(SHARDS)
        ):
            raise Q36MTRNestedControlForestError("forest candidate differs")
        candidates[identity] = row
    if set(candidates) != set(records) or len(candidates) != ROWS:
        raise Q36MTRNestedControlForestError("forest identity coverage differs")
    for identity, row in candidates.items():
        record = records[identity]
        state = row["nested_forest_consensus"]
        selected = state["selected"]
        if (
            state["outer_shard"] != record["shard"]
            or row.get("completion")
            != record["arms"][selected]["row"].get("completion")
            or row.get("task") != record["task"]
        ):
            raise Q36MTRNestedControlForestError("forest selection binding differs")
    return candidates, report


def _answer_features(answer: str | None) -> tuple[float, ...]:
    if not isinstance(answer, str):
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    try:
        numeric = float(answer.replace(",", ""))
    except ValueError:
        return (1.0, min(len(answer), 100) / 100.0, 0.0, 0.0, 0.0)
    magnitude = math.log1p(abs(numeric)) if math.isfinite(numeric) else 100.0
    return (
        1.0,
        min(len(answer), 100) / 100.0,
        1.0,
        float(numeric < 0.0),
        min(magnitude, 100.0) / 20.0,
    )


def _control_answer(record: dict[str, Any], candidate: dict[str, Any]) -> str | None:
    if record["task"] == "mbpp" or not has_explicit_final_answer(
        candidate["completion"]
    ):
        return None
    return TASKS[record["task"]]["extract"](candidate["completion"])


def _feature_vector(
    record: dict[str, Any],
    candidate: dict[str, Any],
    selected: str,
    forest: dict[str, Any],
) -> np.ndarray:
    control_answer = _control_answer(record, candidate)
    forest_answer = record["arms"][selected]["answer"]
    arm_answers = [record["arms"][arm]["answer"] for arm in ARM_ORDER]
    counts = Counter(answer for answer in arm_answers if answer is not None)
    control_votes = counts.get(control_answer, 0)
    forest_votes = counts.get(forest_answer, 0)
    control_mask = sum(
        (control_answer is not None and answer == control_answer) << index
        for index, answer in enumerate(arm_answers)
    )
    forest_mask = sum(
        (forest_answer is not None and answer == forest_answer) << index
        for index, answer in enumerate(arm_answers)
    )
    values = np.asarray(
        [
            float(record["task"] == "bbh_logic"),
            float(record["task"] == "math500"),
            float(record["task"] == "mbpp"),
            math.log1p(candidate["generated_tokens"]) / 8.0,
            float(candidate["max_token_exhausted"]),
            min(len(candidate["completion"]), 5000) / 5000.0,
            float(has_explicit_final_answer(candidate["completion"])),
            control_votes / len(ARM_ORDER),
            forest_votes / len(ARM_ORDER),
            (control_votes - forest_votes) / len(ARM_ORDER),
            control_mask / ((1 << len(ARM_ORDER)) - 1),
            forest_mask / ((1 << len(ARM_ORDER)) - 1),
            float(control_answer is not None and control_answer == forest_answer),
            float(control_answer is None),
            float(forest["overrode_pattern"]),
            float(forest["forest_margin"]),
            (
                float(forest["forest_probability"])
                if forest["forest_probability"] is not None
                else -1.0
            ),
            (
                float(forest["pattern_probability"])
                if forest["pattern_probability"] is not None
                else -1.0
            ),
            float(forest["pattern_estimated_reliability"]),
            *_answer_features(control_answer),
            *_answer_features(forest_answer),
        ],
        dtype=np.float64,
    )
    if values.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(values)):
        raise Q36MTRNestedControlForestError("control feature geometry differs")
    return values


def _fit_control(matrix: np.ndarray, labels: np.ndarray) -> ExtraTreesRegressor:
    if len(matrix) != len(labels) or not len(matrix) or not np.all(np.isfinite(matrix)):
        raise Q36MTRNestedControlForestError("control training geometry differs")
    model = ExtraTreesRegressor(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(matrix, labels)
    return model


def _choose_threshold(
    predictions: np.ndarray,
    control_correct: np.ndarray,
    forest_correct: np.ndarray,
) -> tuple[float, int]:
    scores = {
        threshold: int(
            np.sum(np.where(predictions > threshold, control_correct, forest_correct))
        )
        for threshold in THRESHOLDS
    }
    selected = max(THRESHOLDS, key=lambda threshold: (scores[threshold], threshold))
    return selected, scores[selected]


def run(
    candidate_paths: dict[str, list[Path]],
    score_paths: dict[str, list[Path]],
    forest_candidates_path: Path,
    forest_report_path: Path,
    control_candidate_paths: list[Path],
    control_score_paths: list[Path],
    output: Path,
    report_path: Path,
) -> dict[str, Any]:
    records = _load_inputs(candidate_paths, score_paths)
    if len(records) != ROWS:
        raise Q36MTRNestedControlForestError("base identity geometry differs")
    forest_candidates, forest_report = _load_forest(
        forest_candidates_path, forest_report_path, records
    )
    control_candidates, control_outcomes = _load_control(
        control_candidate_paths, control_score_paths
    )
    if set(records) != set(control_candidates):
        raise Q36MTRNestedControlForestError("base and control identities differ")

    records_by_shard = {
        shard: [row for row in records.values() if row["shard"] == shard]
        for shard in range(SHARDS)
    }
    outer_configs = {
        shard: _select_outer_config(records_by_shard, shard) for shard in range(SHARDS)
    }
    samples, identity_samples = _cluster_samples(records)
    sample_matrix = np.stack([sample["features"] for sample in samples])
    sample_labels = np.asarray(
        [sample["correct"] for sample in samples], dtype=np.int64
    )
    sample_shards = np.asarray([sample["shard"] for sample in samples])

    outer_models: dict[int, ExtraTreesRegressor] = {}
    outer_thresholds: dict[int, tuple[float, int]] = {}
    outer_training_rows: dict[int, int] = {}
    identities = sorted(records)
    for outer in range(SHARDS):
        config = outer_configs[outer][0]
        forest_threshold = float(
            forest_report["outer_models"][str(outer)]["override_threshold"]
        )
        feature_rows: list[np.ndarray] = []
        deltas: list[int] = []
        groups: list[int] = []
        control_correct: list[int] = []
        forest_correct: list[int] = []
        for inner in range(SHARDS):
            if inner == outer:
                continue
            training = (sample_shards != outer) & (sample_shards != inner)
            validation = sample_shards == inner
            base_model = _fit_forest(sample_matrix[training], sample_labels[training])
            probabilities = np.zeros(len(samples), dtype=np.float64)
            probabilities[validation] = base_model.predict_proba(
                sample_matrix[validation]
            )[:, 1]
            pattern_model = _fit_pattern(
                [
                    row
                    for shard in range(SHARDS)
                    if shard not in (outer, inner)
                    for row in records_by_shard[shard]
                ]
            )
            for identity in identities:
                record = records[identity]
                if record["shard"] != inner:
                    continue
                selected, reliability, pattern_mask = _pattern_choose(
                    record, pattern_model, config
                )
                forest_probability = None
                pattern_probability = None
                margin = -1.0
                overrode = False
                if record["task"] != "mbpp" and identity_samples[identity]:
                    best, pattern, margin = _prediction_for_identity(
                        record,
                        identity_samples[identity],
                        samples,
                        probabilities,
                        selected,
                    )
                    assert best is not None and pattern is not None
                    forest_probability = float(probabilities[best])
                    pattern_probability = float(probabilities[pattern])
                    overrode = (
                        samples[best]["mask"] != pattern_mask
                        and margin > forest_threshold
                    )
                    if overrode:
                        selected = samples[best]["selected"]
                state = {
                    "overrode_pattern": overrode,
                    "forest_margin": margin,
                    "forest_probability": forest_probability,
                    "pattern_probability": pattern_probability,
                    "pattern_estimated_reliability": reliability,
                }
                base_correct = int(record["arms"][selected]["correct"])
                candidate_correct = int(control_outcomes[identity]["correct"])
                feature_rows.append(
                    _feature_vector(
                        record,
                        control_candidates[identity],
                        selected,
                        state,
                    )
                )
                deltas.append(candidate_correct - base_correct)
                groups.append(inner)
                control_correct.append(candidate_correct)
                forest_correct.append(base_correct)
        matrix = np.stack(feature_rows)
        labels = np.asarray(deltas, dtype=np.float64)
        group_array = np.asarray(groups, dtype=np.int64)
        control_array = np.asarray(control_correct, dtype=np.int64)
        forest_array = np.asarray(forest_correct, dtype=np.int64)
        oof_predictions = np.zeros(len(labels), dtype=np.float64)
        for inner in range(SHARDS):
            if inner == outer:
                continue
            inner_model = _fit_control(
                matrix[group_array != inner], labels[group_array != inner]
            )
            oof_predictions[group_array == inner] = inner_model.predict(
                matrix[group_array == inner]
            )
        outer_thresholds[outer] = _choose_threshold(
            oof_predictions, control_array, forest_array
        )
        outer_models[outer] = _fit_control(matrix, labels)
        outer_training_rows[outer] = len(labels)

    outputs = []
    correct = 0
    overrides = 0
    gained = 0
    lost = 0
    domain_correct: Counter[str] = Counter()
    for identity in identities:
        record = records[identity]
        outer = record["shard"]
        forest_row = forest_candidates[identity]
        forest_state = forest_row["nested_forest_consensus"]
        selected = forest_state["selected"]
        prediction = float(
            outer_models[outer].predict(
                _feature_vector(
                    record,
                    control_candidates[identity],
                    selected,
                    forest_state,
                )[None, :]
            )[0]
        )
        threshold, inner_correct = outer_thresholds[outer]
        use_control = prediction > threshold
        source = control_candidates[identity] if use_control else forest_row
        row = dict(source)
        row["nested_control_forest"] = {
            "schema": SCHEMA,
            "selected": CONTROL_ARM if use_control else "nested_forest",
            "outer_shard": outer,
            "predicted_control_delta": prediction,
            "override_threshold": threshold,
            "inner_validation_correct": inner_correct,
            "outer_training_rows": outer_training_rows[outer],
            "heldout_identity_labels_read": 0,
        }
        outputs.append(row)
        forest_value = int(record["arms"][selected]["correct"])
        control_value = int(control_outcomes[identity]["correct"])
        value = control_value if use_control else forest_value
        correct += value
        domain_correct[record["task"]] += value
        overrides += int(use_control)
        gained += int(use_control and control_value and not forest_value)
        lost += int(use_control and forest_value and not control_value)

    output_sha256 = _atomic_lines(output, outputs)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "rows": len(outputs),
        "nested_crossfit_correct": correct,
        "nested_crossfit_accuracy": correct / len(outputs),
        "domain_correct": dict(sorted(domain_correct.items())),
        "control_overrides": overrides,
        "control_only_correct": gained,
        "forest_only_correct_lost": lost,
        "heldout_identity_labels_read": 0,
        "development_labels_used_for_training": True,
        "hyperparameters_selected_inside_outer_training_folds": True,
        "outer_shards": SHARDS,
        "outer_training_shards": SHARDS - 1,
        "inner_training_shards_per_validation": SHARDS - 2,
        "control_model": {
            "implementation": "sklearn.ensemble.ExtraTreesRegressor",
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "n_estimators": N_ESTIMATORS,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "max_features": MAX_FEATURES,
            "random_state": RANDOM_STATE,
            "feature_names": list(FEATURE_NAMES),
        },
        "threshold_grid": list(THRESHOLDS),
        "outer_models": {
            str(outer): {
                "override_threshold": outer_thresholds[outer][0],
                "inner_validation_correct": outer_thresholds[outer][1],
                "training_rows": outer_training_rows[outer],
            }
            for outer in range(SHARDS)
        },
        "input_sha256": {
            "arms": {
                arm: {
                    "candidates": [sha256_file(path) for path in candidate_paths[arm]],
                    "scores": [sha256_file(path) for path in score_paths[arm]],
                }
                for arm in ARM_ORDER
            },
            "forest_candidates": sha256_file(forest_candidates_path),
            "forest_report": sha256_file(forest_report_path),
            "control_candidates": [
                sha256_file(path) for path in control_candidate_paths
            ],
            "control_scores": [sha256_file(path) for path in control_score_paths],
        },
        "output": str(output.resolve()),
        "output_sha256": output_sha256,
    }
    _atomic_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARM_ORDER:
        option = arm.replace("_", "-")
        parser.add_argument(f"--{option}", action="append", type=Path, required=True)
        parser.add_argument(
            f"--{option}-score", action="append", type=Path, required=True
        )
    parser.add_argument("--forest-candidates", type=Path, required=True)
    parser.add_argument("--forest-report", type=Path, required=True)
    parser.add_argument(
        "--control-candidates", action="append", type=Path, required=True
    )
    parser.add_argument("--control-score", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = {arm: getattr(args, arm) for arm in ARM_ORDER}
    scores = {arm: getattr(args, f"{arm}_score") for arm in ARM_ORDER}
    report = run(
        candidates,
        scores,
        args.forest_candidates,
        args.forest_report,
        args.control_candidates,
        args.control_score,
        args.output,
        args.report,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
