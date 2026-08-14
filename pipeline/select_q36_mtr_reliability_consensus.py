#!/usr/bin/env python3
"""Cross-fit a reliability-weighted consensus over Q36 reasoning trajectories."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from hf_q36_mtr_hierarchical_synthesis import ROWS
from select_q36_mtr_consensus import ARM_ORDER, normalized_answer
from select_q36_mtr_interpolation_retention import (
    _atomic_json,
    _atomic_lines,
    sha256_file,
)

SCHEMA = "shohin-q36-mtr-reliability-consensus-v1"
REPORT_SCHEMA = "shohin-q36-mtr-reliability-consensus-report-v1"
SHARDS = 16
L2 = 0.1
STEPS = 1200
LEARNING_RATE = 0.15
FEATURE_NAMES = (
    "bias",
    "vote_fraction",
    "vote_fraction_squared",
    *(f"contains_{arm}" for arm in ARM_ORDER),
    "minimum_log_tokens",
    "mean_log_tokens",
    "maximum_log_tokens",
    "exhausted_fraction",
    "normalized_answer_length",
    "task_bbh_logic",
    "task_math500",
)


class Q36MTRReliabilityConsensusError(RuntimeError):
    """Raised when the cross-fit consensus inputs or model differ."""


def _clusters(record: dict[str, Any]) -> list[dict[str, Any]]:
    task = record["task"]
    groups: dict[str, list[str]] = defaultdict(list)
    for arm in ARM_ORDER:
        answer = record["arms"][arm]["answer"]
        if answer is not None:
            groups[answer].append(arm)
    clusters = []
    for answer, arms in groups.items():
        correctness = {record["arms"][arm]["correct"] for arm in arms}
        if len(correctness) != 1:
            raise Q36MTRReliabilityConsensusError(
                "normalized-answer correctness differs"
            )
        tokens = [record["arms"][arm]["generated_tokens"] for arm in arms]
        vote_fraction = len(arms) / len(ARM_ORDER)
        features = np.asarray(
            [
                1.0,
                vote_fraction,
                vote_fraction**2,
                *(float(arm in arms) for arm in ARM_ORDER),
                math.log1p(min(tokens)) / 8.0,
                math.log1p(sum(tokens) / len(tokens)) / 8.0,
                math.log1p(max(tokens)) / 8.0,
                sum(record["arms"][arm]["max_token_exhausted"] for arm in arms)
                / len(arms),
                min(len(answer), 100) / 100.0,
                float(task == "bbh_logic"),
                float(task == "math500"),
            ],
            dtype=np.float64,
        )
        if features.shape != (len(FEATURE_NAMES),):
            raise Q36MTRReliabilityConsensusError("feature geometry differs")
        clusters.append(
            {
                "answer": answer,
                "arms": arms,
                "features": features,
                "correct": bool(next(iter(correctness))),
            }
        )
    return clusters


def _fit(records: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [
        cluster
        for record in records
        if record["task"] != "mbpp"
        for cluster in _clusters(record)
    ]
    if not samples:
        raise Q36MTRReliabilityConsensusError("training samples are empty")
    matrix = np.stack([sample["features"] for sample in samples])
    labels = np.asarray([sample["correct"] for sample in samples], dtype=np.float64)
    mean = matrix[:, 1:].mean(axis=0)
    scale = matrix[:, 1:].std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = matrix.copy()
    standardized[:, 1:] = (matrix[:, 1:] - mean) / scale
    weights = np.zeros(standardized.shape[1], dtype=np.float64)
    for _ in range(STEPS):
        logits = np.clip(standardized @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = standardized.T @ (probabilities - labels) / len(labels)
        gradient[1:] += L2 * weights[1:]
        weights -= LEARNING_RATE * gradient
    if not all(np.isfinite(value) for value in weights):
        raise Q36MTRReliabilityConsensusError("model weights are not finite")
    return {
        "weights": weights,
        "mean": mean,
        "scale": scale,
        "samples": len(samples),
        "positive_samples": int(labels.sum()),
    }


def _choose(record: dict[str, Any], model: dict[str, Any]) -> tuple[str, float]:
    if record["task"] == "mbpp":
        return "interpolation", 1.0
    best: tuple[tuple[float, int, int], str] | None = None
    for cluster in _clusters(record):
        features = cluster["features"].copy()
        features[1:] = (features[1:] - model["mean"]) / model["scale"]
        logit = float(features @ model["weights"])
        probability = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
        tie_priority = -min(ARM_ORDER.index(arm) for arm in cluster["arms"])
        key = (probability, len(cluster["arms"]), tie_priority)
        if best is None or key > best[0]:
            best = (key, cluster["arms"][0])
    if best is None:
        return "hierarchy", 0.0
    return best[1], best[0][0]


def _load_inputs(
    candidate_paths: dict[str, list[Path]], score_paths: dict[str, list[Path]]
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER:
        if len(candidate_paths[arm]) != SHARDS or len(score_paths[arm]) != SHARDS:
            raise Q36MTRReliabilityConsensusError("input shard geometry differs")
        for shard, (candidate_path, score_path) in enumerate(
            zip(candidate_paths[arm], score_paths[arm], strict=True)
        ):
            candidate_sha256 = sha256_file(candidate_path)
            score = json.loads(score_path.read_text())
            if score.get("candidates_sha256") != candidate_sha256:
                raise Q36MTRReliabilityConsensusError("score candidate hash differs")
            candidates = [
                json.loads(line) for line in candidate_path.read_text().splitlines()
            ]
            outcomes = {
                row["identity_sha256"]: bool(row["correct"])
                for row in score.get("outcomes", [])
            }
            if {row.get("identity_sha256") for row in candidates} != set(outcomes):
                raise Q36MTRReliabilityConsensusError("score coverage differs")
            for row in candidates:
                identity = row["identity_sha256"]
                record = records.setdefault(
                    identity, {"task": row["task"], "shard": shard, "arms": {}}
                )
                if record["task"] != row["task"] or record["shard"] != shard:
                    raise Q36MTRReliabilityConsensusError("identity lineage differs")
                if arm in record["arms"]:
                    raise Q36MTRReliabilityConsensusError("duplicate arm identity")
                record["arms"][arm] = {
                    "row": row,
                    "answer": normalized_answer(row["task"], row["completion"]),
                    "correct": outcomes[identity],
                    "generated_tokens": row["generated_tokens"],
                    "max_token_exhausted": bool(row["max_token_exhausted"]),
                }
    if len(records) != ROWS or any(
        tuple(row["arms"]) != ARM_ORDER for row in records.values()
    ):
        raise Q36MTRReliabilityConsensusError("complete arm coverage differs")
    return records


def run(
    candidate_paths: dict[str, list[Path]],
    score_paths: dict[str, list[Path]],
    output: Path,
    report_path: Path,
) -> dict[str, Any]:
    records = _load_inputs(candidate_paths, score_paths)
    models = {
        shard: _fit([row for row in records.values() if row["shard"] != shard])
        for shard in range(SHARDS)
    }
    counts: Counter[tuple[str, str]] = Counter()
    crossfit_correct = 0
    outputs = []
    for identity in sorted(records):
        record = records[identity]
        selected, probability = _choose(record, models[record["shard"]])
        source = record["arms"][selected]
        row = dict(source["row"])
        row["reliability_consensus"] = {
            "schema": SCHEMA,
            "selected": selected,
            "crossfit_shard": record["shard"],
            "cluster_probability": probability,
            "training_shards": SHARDS - 1,
            "heldout_identity_labels_read": 0,
            "feature_names": list(FEATURE_NAMES),
            "l2": L2,
        }
        outputs.append(row)
        counts[(row["task"], selected)] += 1
        crossfit_correct += int(source["correct"])
    output_sha256 = _atomic_lines(output, outputs)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "rows": len(outputs),
        "crossfit_correct": crossfit_correct,
        "crossfit_accuracy": crossfit_correct / len(outputs),
        "shards": SHARDS,
        "training_shards_per_selection": SHARDS - 1,
        "heldout_identity_labels_read": 0,
        "development_labels_used_for_training": True,
        "hyperparameters_selected_on_development": True,
        "feature_names": list(FEATURE_NAMES),
        "l2": L2,
        "steps": STEPS,
        "learning_rate": LEARNING_RATE,
        "selection_counts": {
            f"{task}:{arm}": count for (task, arm), count in sorted(counts.items())
        },
        "models": {
            str(shard): {
                "weights": model["weights"].tolist(),
                "mean": model["mean"].tolist(),
                "scale": model["scale"].tolist(),
                "samples": model["samples"],
                "positive_samples": model["positive_samples"],
            }
            for shard, model in models.items()
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
