#!/usr/bin/env python3
"""Nested-cross-fit a Bayesian agreement-pattern consensus over Q36 trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import product
import json
import math
from pathlib import Path
from typing import Any

from select_q36_mtr_consensus import ARM_ORDER
from select_q36_mtr_interpolation_retention import (
    _atomic_json,
    _atomic_lines,
    sha256_file,
)
from select_q36_mtr_pattern_consensus import _fit
from select_q36_mtr_reliability_consensus import SHARDS, _clusters, _load_inputs

SCHEMA = "shohin-q36-mtr-nested-pattern-consensus-v1"
REPORT_SCHEMA = "shohin-q36-mtr-nested-pattern-consensus-report-v1"
ALPHAS = (10.0, 30.0, 100.0, 300.0)
WEIGHT_VALUES = (0.25, 1.0, 4.0)
CONFIGS = tuple(
    (alpha, weights) for alpha in ALPHAS for weights in product(WEIGHT_VALUES, repeat=3)
)


class Q36MTRNestedPatternConsensusError(RuntimeError):
    """Raised when nested pattern selection or its inputs differ."""


def _smoothed(counts: list[int] | None, prior: float, alpha: float) -> float:
    correct, total = counts or [0, 0]
    return (correct + alpha * prior) / (total + alpha)


def _choose(
    record: dict[str, Any],
    model: dict[str, Any],
    config: tuple[float, tuple[float, float, float]],
) -> tuple[str, float, int]:
    if record["task"] == "mbpp":
        return "interpolation", 1.0, 1 << ARM_ORDER.index("interpolation")
    alpha, weights = config
    task = record["task"]
    prior_counts = model["prior"][task]
    prior = (prior_counts[0] + 1.0) / (prior_counts[1] + 2.0)
    best: tuple[tuple[float, int, int], str, int] | None = None
    for cluster in _clusters(record):
        names = cluster["arms"]
        mask = sum(1 << ARM_ORDER.index(name) for name in names)
        exact = _smoothed(model["exact"].get((task, mask)), prior, alpha)
        size = _smoothed(model["size"].get((task, len(names))), prior, alpha)
        arm = sum(
            _smoothed(model["arm"].get((task, name)), prior, alpha) for name in names
        ) / len(names)
        score = sum(
            weight * math.log(max(value, 1e-12))
            for weight, value in zip(weights, (exact, size, arm), strict=True)
        ) / sum(weights)
        tie_priority = -min(ARM_ORDER.index(name) for name in names)
        key = (score, len(names), tie_priority)
        if best is None or key > best[0]:
            best = (key, names[0], mask)
    if best is None:
        return "hierarchy", 0.0, 1 << ARM_ORDER.index("hierarchy")
    return best[1], best[0][0], best[2]


def _correct(record: dict[str, Any], model: dict[str, Any], config: Any) -> int:
    selected, _, _ = _choose(record, model, config)
    return int(record["arms"][selected]["correct"])


def _select_outer_config(
    records_by_shard: dict[int, list[dict[str, Any]]], outer: int
) -> tuple[tuple[float, tuple[float, float, float]], int]:
    shards = tuple(sorted(records_by_shard))
    if outer not in records_by_shard or len(shards) < 3:
        raise Q36MTRNestedPatternConsensusError("nested shard geometry differs")
    scores = {config: 0 for config in CONFIGS}
    for inner in shards:
        if inner == outer:
            continue
        training = [
            row
            for shard in shards
            if shard not in (outer, inner)
            for row in records_by_shard[shard]
        ]
        model = _fit(training)
        for config in CONFIGS:
            scores[config] += sum(
                _correct(record, model, config) for record in records_by_shard[inner]
            )
    selected = max(CONFIGS, key=lambda config: (scores[config], -CONFIGS.index(config)))
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
    if any(not rows for rows in records_by_shard.values()):
        raise Q36MTRNestedPatternConsensusError("outer shard is empty")
    outer_configs = {
        shard: _select_outer_config(records_by_shard, shard) for shard in range(SHARDS)
    }
    outer_models = {
        shard: _fit(
            [
                row
                for other in range(SHARDS)
                if other != shard
                for row in records_by_shard[other]
            ]
        )
        for shard in range(SHARDS)
    }
    outputs = []
    selections: Counter[tuple[str, str]] = Counter()
    correct = 0
    for identity in sorted(records):
        record = records[identity]
        config, inner_correct = outer_configs[record["shard"]]
        selected, reliability, mask = _choose(
            record, outer_models[record["shard"]], config
        )
        source = record["arms"][selected]
        row = dict(source["row"])
        row["nested_pattern_consensus"] = {
            "schema": SCHEMA,
            "selected": selected,
            "outer_shard": record["shard"],
            "agreement_mask": mask,
            "estimated_reliability": reliability,
            "alpha": config[0],
            "weights": list(config[1]),
            "outer_training_shards": SHARDS - 1,
            "inner_validation_shards": SHARDS - 1,
            "inner_training_shards_per_validation": SHARDS - 2,
            "inner_validation_correct": inner_correct,
            "heldout_identity_labels_read": 0,
        }
        outputs.append(row)
        selections[(row["task"], selected)] += 1
        correct += int(source["correct"])
    output_sha256 = _atomic_lines(output, outputs)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "rows": len(outputs),
        "nested_crossfit_correct": correct,
        "nested_crossfit_accuracy": correct / len(outputs),
        "outer_shards": SHARDS,
        "outer_training_shards": SHARDS - 1,
        "inner_training_shards_per_validation": SHARDS - 2,
        "heldout_identity_labels_read": 0,
        "development_labels_used_for_training": True,
        "hyperparameters_selected_inside_outer_training_folds": True,
        "config_grid": {
            "alphas": list(ALPHAS),
            "weight_values": list(WEIGHT_VALUES),
            "score": "weighted_geometric_mean_of_exact_pattern_vote_size_and_arm_reliability",
        },
        "outer_configs": {
            str(shard): {
                "alpha": config[0],
                "weights": list(config[1]),
                "inner_validation_correct": validation_correct,
            }
            for shard, (config, validation_correct) in outer_configs.items()
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
