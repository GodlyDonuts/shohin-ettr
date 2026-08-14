#!/usr/bin/env python3
"""Cross-fit a Bayesian agreement-pattern consensus over Q36 trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from select_q36_mtr_consensus import ARM_ORDER
from select_q36_mtr_interpolation_retention import (
    _atomic_json,
    _atomic_lines,
    sha256_file,
)
from select_q36_mtr_reliability_consensus import (
    SHARDS,
    _clusters,
    _load_inputs,
)

SCHEMA = "shohin-q36-mtr-pattern-consensus-v1"
REPORT_SCHEMA = "shohin-q36-mtr-pattern-consensus-report-v1"
ALPHA = 30.0


class Q36MTRPatternConsensusError(RuntimeError):
    """Raised when the agreement-pattern consensus inputs or model differ."""


def _increment(bucket: dict[Any, list[int]], key: Any, correct: bool) -> None:
    counts = bucket.setdefault(key, [0, 0])
    counts[0] += int(correct)
    counts[1] += 1


def _fit(records: list[dict[str, Any]]) -> dict[str, Any]:
    exact: dict[tuple[str, int], list[int]] = {}
    size: dict[tuple[str, int], list[int]] = {}
    arm: dict[tuple[str, str], list[int]] = {}
    prior: dict[str, list[int]] = {}
    samples = 0
    for record in records:
        if record["task"] == "mbpp":
            continue
        for cluster in _clusters(record):
            mask = sum(1 << ARM_ORDER.index(name) for name in cluster["arms"])
            correct = cluster["correct"]
            _increment(exact, (record["task"], mask), correct)
            _increment(size, (record["task"], len(cluster["arms"])), correct)
            _increment(prior, record["task"], correct)
            for name in cluster["arms"]:
                _increment(arm, (record["task"], name), correct)
            samples += 1
    if not samples or set(prior) != {"bbh_logic", "math500"}:
        raise Q36MTRPatternConsensusError("pattern training geometry differs")
    return {
        "exact": exact,
        "size": size,
        "arm": arm,
        "prior": prior,
        "samples": samples,
    }


def _smoothed(counts: list[int] | None, prior: float) -> float:
    correct, total = counts or [0, 0]
    return (correct + ALPHA * prior) / (total + ALPHA)


def _choose(record: dict[str, Any], model: dict[str, Any]) -> tuple[str, float, int]:
    if record["task"] == "mbpp":
        return "interpolation", 1.0, 1 << ARM_ORDER.index("interpolation")
    task = record["task"]
    prior_counts = model["prior"][task]
    prior = (prior_counts[0] + 1.0) / (prior_counts[1] + 2.0)
    best: tuple[tuple[float, int, int], str, int] | None = None
    for cluster in _clusters(record):
        names = cluster["arms"]
        mask = sum(1 << ARM_ORDER.index(name) for name in names)
        exact = _smoothed(model["exact"].get((task, mask)), prior)
        size = _smoothed(model["size"].get((task, len(names))), prior)
        arm_reliability = sum(
            _smoothed(model["arm"].get((task, name)), prior) for name in names
        ) / len(names)
        score = (exact * size * arm_reliability) ** (1.0 / 3.0)
        tie_priority = -min(ARM_ORDER.index(name) for name in names)
        key = (score, len(names), tie_priority)
        if best is None or key > best[0]:
            best = (key, names[0], mask)
    if best is None:
        return "hierarchy", 0.0, 1 << ARM_ORDER.index("hierarchy")
    return best[1], best[0][0], best[2]


def _serialize_counts(bucket: dict[Any, list[int]]) -> dict[str, list[int]]:
    return {
        (
            ":".join(str(part) for part in key) if isinstance(key, tuple) else str(key)
        ): value
        for key, value in sorted(bucket.items(), key=lambda item: str(item[0]))
    }


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
    selections: Counter[tuple[str, str]] = Counter()
    correct = 0
    outputs = []
    for identity in sorted(records):
        record = records[identity]
        selected, reliability, mask = _choose(record, models[record["shard"]])
        source = record["arms"][selected]
        row = dict(source["row"])
        row["pattern_consensus"] = {
            "schema": SCHEMA,
            "selected": selected,
            "crossfit_shard": record["shard"],
            "agreement_mask": mask,
            "estimated_reliability": reliability,
            "training_shards": SHARDS - 1,
            "heldout_identity_labels_read": 0,
            "alpha": ALPHA,
        }
        outputs.append(row)
        selections[(row["task"], selected)] += 1
        correct += int(source["correct"])
    output_sha256 = _atomic_lines(output, outputs)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "rows": len(outputs),
        "crossfit_correct": correct,
        "crossfit_accuracy": correct / len(outputs),
        "shards": SHARDS,
        "training_shards_per_selection": SHARDS - 1,
        "heldout_identity_labels_read": 0,
        "development_labels_used_for_training": True,
        "hyperparameters_selected_on_development": True,
        "alpha": ALPHA,
        "score": "geometric_mean_of_exact_pattern_vote_size_and_arm_reliability",
        "selection_counts": {
            f"{task}:{arm}": count for (task, arm), count in sorted(selections.items())
        },
        "models": {
            str(shard): {
                "samples": model["samples"],
                "exact": _serialize_counts(model["exact"]),
                "size": _serialize_counts(model["size"]),
                "arm": _serialize_counts(model["arm"]),
                "prior": _serialize_counts(model["prior"]),
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
