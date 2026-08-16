#!/usr/bin/env python3
"""Nested-cross-fit a forest override over Q36 agreement-pattern consensus."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

from select_q36_mtr_consensus import ARM_ORDER
from select_q36_mtr_interpolation_retention import (
    _atomic_json,
    _atomic_lines,
    sha256_file,
)
from select_q36_mtr_nested_pattern_consensus import (
    _choose as _pattern_choose,
    _select_outer_config,
)
from select_q36_mtr_pattern_consensus import _fit as _fit_pattern
from select_q36_mtr_reliability_consensus import SHARDS, _clusters, _load_inputs

SCHEMA = "shohin-q36-mtr-nested-forest-consensus-v1"
REPORT_SCHEMA = "shohin-q36-mtr-nested-forest-consensus-report-v1"
N_ESTIMATORS = 200
MIN_SAMPLES_LEAF = 5
MAX_FEATURES = "sqrt"
RANDOM_STATE = 2026081429
THRESHOLDS = tuple(round(index * 0.005, 3) for index in range(61))
FEATURE_NAMES = (
    "vote_fraction",
    "cluster_count",
    "agreement_mask_fraction",
    "mask_is_numeric_maximum",
    "minimum_log_tokens",
    "mean_log_tokens",
    "maximum_log_tokens",
    "token_standard_deviation",
    "exhausted_fraction",
    "mean_completion_length",
    "mean_wall_seconds",
    *(f"contains_{arm}" for arm in ARM_ORDER),
    *(f"all_tokens_{arm}" for arm in ARM_ORDER),
    *(f"all_exhausted_{arm}" for arm in ARM_ORDER),
    *(f"answer_length_{arm}" for arm in ARM_ORDER),
    *(
        f"agreement_{left}_{right}"
        for left_index, left in enumerate(ARM_ORDER)
        for right in ARM_ORDER[left_index + 1 :]
    ),
)


class Q36MTRNestedForestConsensusError(RuntimeError):
    """Raised when nested forest fitting or its inputs differ."""


def _mask(names: list[str]) -> int:
    return sum(1 << ARM_ORDER.index(name) for name in names)


def _feature_vector(record: dict[str, Any], cluster: dict[str, Any]) -> np.ndarray:
    clusters = _clusters(record)
    masks = tuple(sorted(_mask(item["arms"]) for item in clusters))
    names = cluster["arms"]
    tokens = [record["arms"][arm]["generated_tokens"] for arm in names]
    lengths = [len(record["arms"][arm]["row"]["completion"]) for arm in names]
    wall_seconds = [
        float(record["arms"][arm]["row"].get("wall_seconds", 0.0)) for arm in names
    ]
    mask = _mask(names)
    features = np.asarray(
        [
            len(names) / len(ARM_ORDER),
            len(clusters),
            mask / ((1 << len(ARM_ORDER)) - 1),
            float(mask == max(masks)),
            math.log1p(min(tokens)) / 8.0,
            math.log1p(sum(tokens) / len(tokens)) / 8.0,
            math.log1p(max(tokens)) / 8.0,
            float(np.std(tokens)) / 100.0,
            sum(record["arms"][arm]["max_token_exhausted"] for arm in names)
            / len(names),
            min(sum(lengths) / len(lengths), 1000.0) / 1000.0,
            min(sum(wall_seconds) / len(wall_seconds), 100.0) / 100.0,
            *(float(arm in names) for arm in ARM_ORDER),
            *(record["arms"][arm]["generated_tokens"] / 2048.0 for arm in ARM_ORDER),
            *(float(record["arms"][arm]["max_token_exhausted"]) for arm in ARM_ORDER),
            *(len(record["arms"][arm]["answer"] or "") / 100.0 for arm in ARM_ORDER),
            *(
                float(
                    record["arms"][left]["answer"] is not None
                    and record["arms"][left]["answer"]
                    == record["arms"][right]["answer"]
                )
                for left_index, left in enumerate(ARM_ORDER)
                for right in ARM_ORDER[left_index + 1 :]
            ),
        ],
        dtype=np.float64,
    )
    if features.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(features)):
        raise Q36MTRNestedForestConsensusError("forest feature geometry differs")
    return features


def _cluster_samples(
    records: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    samples: list[dict[str, Any]] = []
    identity_samples: dict[str, list[int]] = {}
    for identity in sorted(records):
        record = records[identity]
        indices = []
        if record["task"] != "mbpp":
            for cluster in _clusters(record):
                index = len(samples)
                samples.append(
                    {
                        "identity": identity,
                        "shard": record["shard"],
                        "mask": _mask(cluster["arms"]),
                        "arms": cluster["arms"],
                        "selected": cluster["arms"][0],
                        "correct": int(cluster["correct"]),
                        "features": _feature_vector(record, cluster),
                    }
                )
                indices.append(index)
        identity_samples[identity] = indices
    if not samples:
        raise Q36MTRNestedForestConsensusError("forest samples are empty")
    return samples, identity_samples


def _fit_forest(matrix: np.ndarray, labels: np.ndarray) -> RandomForestClassifier:
    if len(matrix) != len(labels) or len(set(labels.tolist())) != 2:
        raise Q36MTRNestedForestConsensusError("forest training labels differ")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(matrix, labels)
    return model


def _prediction_for_identity(
    record: dict[str, Any],
    indices: list[int],
    samples: list[dict[str, Any]],
    probabilities: np.ndarray,
    pattern_selected: str,
) -> tuple[int | None, int | None, float]:
    if record["task"] == "mbpp" or not indices:
        return None, None, -1.0
    best = max(
        indices,
        key=lambda index: (
            probabilities[index],
            len(samples[index]["arms"]),
            -ARM_ORDER.index(samples[index]["selected"]),
        ),
    )
    try:
        pattern = next(
            index for index in indices if pattern_selected in samples[index]["arms"]
        )
    except StopIteration as error:
        raise Q36MTRNestedForestConsensusError(
            "pattern selection has no forest cluster"
        ) from error
    return best, pattern, float(probabilities[best] - probabilities[pattern])


def _choose_threshold(rows: list[tuple[int, int, float, bool]]) -> tuple[float, int]:
    if not rows:
        raise Q36MTRNestedForestConsensusError("threshold validation rows are empty")
    scores = {
        threshold: sum(
            forest_correct if differs and margin > threshold else pattern_correct
            for pattern_correct, forest_correct, margin, differs in rows
        )
        for threshold in THRESHOLDS
    }
    selected = max(THRESHOLDS, key=lambda threshold: (scores[threshold], threshold))
    return selected, scores[selected]


def run(
    candidate_paths: dict[str, list[Path]],
    score_paths: dict[str, list[Path]],
    output: Path,
    report_path: Path,
) -> dict[str, Any]:
    records = _load_inputs(candidate_paths, score_paths)
    records_by_shard = {
        shard: [row for row in records.values() if row["shard"] == shard]
        for shard in range(SHARDS)
    }
    outer_configs = {
        shard: _select_outer_config(records_by_shard, shard) for shard in range(SHARDS)
    }
    samples, identity_samples = _cluster_samples(records)
    matrix = np.stack([sample["features"] for sample in samples])
    labels = np.asarray([sample["correct"] for sample in samples], dtype=np.int64)
    sample_shards = np.asarray([sample["shard"] for sample in samples])

    outer_thresholds: dict[int, tuple[float, int]] = {}
    for outer in range(SHARDS):
        config, _ = outer_configs[outer]
        validation_rows = []
        for inner in range(SHARDS):
            if inner == outer:
                continue
            training = (sample_shards != outer) & (sample_shards != inner)
            validation = sample_shards == inner
            forest = _fit_forest(matrix[training], labels[training])
            probabilities = np.zeros(len(samples), dtype=np.float64)
            probabilities[validation] = forest.predict_proba(matrix[validation])[:, 1]
            pattern_model = _fit_pattern(
                [
                    row
                    for shard in range(SHARDS)
                    if shard not in (outer, inner)
                    for row in records_by_shard[shard]
                ]
            )
            for identity in sorted(records):
                record = records[identity]
                if record["shard"] != inner:
                    continue
                if record["task"] == "mbpp":
                    correct = int(record["arms"]["interpolation"]["correct"])
                    validation_rows.append((correct, correct, -1.0, False))
                    continue
                pattern_selected, _, pattern_mask = _pattern_choose(
                    record, pattern_model, config
                )
                if not identity_samples[identity]:
                    correct = int(record["arms"][pattern_selected]["correct"])
                    validation_rows.append((correct, correct, -1.0, False))
                    continue
                best, pattern, margin = _prediction_for_identity(
                    record,
                    identity_samples[identity],
                    samples,
                    probabilities,
                    pattern_selected,
                )
                assert best is not None and pattern is not None
                validation_rows.append(
                    (
                        samples[pattern]["correct"],
                        samples[best]["correct"],
                        margin,
                        samples[best]["mask"] != pattern_mask,
                    )
                )
        outer_thresholds[outer] = _choose_threshold(validation_rows)

    outer_probabilities: dict[int, np.ndarray] = {}
    outer_pattern_models = {}
    for outer in range(SHARDS):
        training = sample_shards != outer
        validation = sample_shards == outer
        forest = _fit_forest(matrix[training], labels[training])
        probabilities = np.zeros(len(samples), dtype=np.float64)
        probabilities[validation] = forest.predict_proba(matrix[validation])[:, 1]
        outer_probabilities[outer] = probabilities
        outer_pattern_models[outer] = _fit_pattern(
            [
                row
                for shard in range(SHARDS)
                if shard != outer
                for row in records_by_shard[shard]
            ]
        )

    outputs = []
    correct = 0
    overrides = 0
    selections: Counter[tuple[str, str]] = Counter()
    domain_correct: Counter[str] = Counter()
    for identity in sorted(records):
        record = records[identity]
        outer = record["shard"]
        config, inner_pattern_correct = outer_configs[outer]
        threshold, inner_forest_correct = outer_thresholds[outer]
        pattern_selected, pattern_reliability, pattern_mask = _pattern_choose(
            record, outer_pattern_models[outer], config
        )
        selected = pattern_selected
        forest_probability = None
        pattern_probability = None
        margin = -1.0
        if record["task"] != "mbpp" and identity_samples[identity]:
            best, pattern, margin = _prediction_for_identity(
                record,
                identity_samples[identity],
                samples,
                outer_probabilities[outer],
                pattern_selected,
            )
            assert best is not None and pattern is not None
            forest_probability = float(outer_probabilities[outer][best])
            pattern_probability = float(outer_probabilities[outer][pattern])
            if samples[best]["mask"] != pattern_mask and margin > threshold:
                selected = samples[best]["selected"]
                overrides += 1
        source = record["arms"][selected]
        row = dict(source["row"])
        row["nested_forest_consensus"] = {
            "schema": SCHEMA,
            "selected": selected,
            "pattern_selected": pattern_selected,
            "outer_shard": outer,
            "pattern_agreement_mask": pattern_mask,
            "pattern_estimated_reliability": pattern_reliability,
            "forest_probability": forest_probability,
            "pattern_probability": pattern_probability,
            "forest_margin": margin,
            "override_threshold": threshold,
            "overrode_pattern": selected != pattern_selected,
            "pattern_alpha": config[0],
            "pattern_weights": list(config[1]),
            "inner_pattern_validation_correct": inner_pattern_correct,
            "inner_forest_validation_correct": inner_forest_correct,
            "heldout_identity_labels_read": 0,
        }
        outputs.append(row)
        row_correct = int(source["correct"])
        correct += row_correct
        domain_correct[row["task"]] += row_correct
        selections[(row["task"], selected)] += 1

    output_sha256 = _atomic_lines(output, outputs)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "rows": len(outputs),
        "nested_crossfit_correct": correct,
        "nested_crossfit_accuracy": correct / len(outputs),
        "domain_correct": dict(sorted(domain_correct.items())),
        "forest_overrides": overrides,
        "outer_shards": SHARDS,
        "outer_training_shards": SHARDS - 1,
        "inner_training_shards_per_validation": SHARDS - 2,
        "heldout_identity_labels_read": 0,
        "development_labels_used_for_training": True,
        "hyperparameters_selected_inside_outer_training_folds": True,
        "forest": {
            "implementation": "sklearn.ensemble.RandomForestClassifier",
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
            str(shard): {
                "pattern_alpha": outer_configs[shard][0][0],
                "pattern_weights": list(outer_configs[shard][0][1]),
                "inner_pattern_validation_correct": outer_configs[shard][1],
                "override_threshold": outer_thresholds[shard][0],
                "inner_forest_validation_correct": outer_thresholds[shard][1],
            }
            for shard in range(SHARDS)
        },
        "selection_counts": {
            f"{task}:{arm}": count for (task, arm), count in sorted(selections.items())
        },
        "input_sha256": {
            arm: {
                "candidates": [sha256_file(path) for path in candidate_paths[arm]],
                "scores": [sha256_file(path) for path in score_paths[arm]],
            }
            for arm in ARM_ORDER
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = {arm: getattr(args, arm) for arm in ARM_ORDER}
    scores = {arm: getattr(args, f"{arm}_score") for arm in ARM_ORDER}
    print(json.dumps(run(candidates, scores, args.output, args.report), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
