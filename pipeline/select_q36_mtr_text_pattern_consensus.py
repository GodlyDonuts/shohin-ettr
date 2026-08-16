#!/usr/bin/env python3
"""Nested-cross-fit a text-conditioned agreement-pattern consensus for Q36."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from typing import Any

from select_q36_mtr_consensus import ARM_ORDER
from select_q36_mtr_interpolation_retention import (
    _atomic_json,
    _atomic_lines,
    sha256_file,
)
from select_q36_mtr_nested_pattern_consensus import (
    CONFIGS,
    _select_outer_config,
    _smoothed,
)
from select_q36_mtr_pattern_consensus import _fit
from select_q36_mtr_reliability_consensus import SHARDS, _clusters, _load_inputs

SCHEMA = "shohin-q36-mtr-text-pattern-consensus-v1"
REPORT_SCHEMA = "shohin-q36-mtr-text-pattern-consensus-report-v1"
TOKEN_ALPHAS = (30.0, 50.0, 100.0)
TOKEN_WEIGHTS = (0.01, 0.025, 0.05)
TOKEN_TOP_K = (1, 3, 5)
TOKEN_CONFIGS = tuple(
    (alpha, weight, top_k)
    for alpha in TOKEN_ALPHAS
    for weight in TOKEN_WEIGHTS
    for top_k in TOKEN_TOP_K
)
MAX_TOKENS = 80
STOPWORDS = frozenset("""
    solve following problem make sure put answer only inside boxed reason carefully
    then exact requested option label which what where when this that with from into
    have has are was were will would should could about there their them they for and
    the not but all any one two three four five six seven eight nine ten
    """.split())


class Q36MTRTextPatternConsensusError(RuntimeError):
    """Raised when text-conditioned consensus inputs or fitting differ."""


def _tokens(source_prompt: str) -> tuple[str, ...]:
    if not isinstance(source_prompt, str) or not source_prompt.strip():
        raise Q36MTRTextPatternConsensusError("source prompt is empty")
    return tuple(
        dict.fromkeys(
            token
            for token in re.findall(r"[a-z]{3,}", source_prompt.lower())
            if token not in STOPWORDS
        )
    )[:MAX_TOKENS]


def _load_sources(path: Path) -> tuple[dict[str, tuple[str, ...]], str]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    sources: dict[str, tuple[str, ...]] = {}
    for row in rows:
        if row.get("split") != "development" or row.get("task") not in {
            "bbh_logic",
            "math500",
            "mbpp",
        }:
            raise Q36MTRTextPatternConsensusError("source row projection differs")
        identity = row.get("identity_sha256")
        if not isinstance(identity, str) or identity in sources:
            raise Q36MTRTextPatternConsensusError("source identity differs")
        sources[identity] = _tokens(row.get("source_prompt"))
    return sources, sha256_file(path)


def _fit_text(records: list[dict[str, Any]]) -> dict[str, Any]:
    model = _fit(records)
    token_counts: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    for record in records:
        if record["task"] == "mbpp":
            continue
        for cluster in _clusters(record):
            for arm in cluster["arms"]:
                for token in record["tokens"]:
                    counts = token_counts[(record["task"], arm, token)]
                    counts[0] += int(cluster["correct"])
                    counts[1] += 1
    model["token"] = dict(token_counts)
    return model


def _logit(value: float) -> float:
    bounded = max(1e-6, min(1.0 - 1e-6, value))
    return math.log(bounded / (1.0 - bounded))


def _choose(
    record: dict[str, Any],
    model: dict[str, Any],
    base_config: tuple[float, tuple[float, float, float]],
    token_config: tuple[float, float, int],
) -> tuple[str, float, float, int]:
    if record["task"] == "mbpp":
        return "interpolation", 1.0, 0.0, 1 << ARM_ORDER.index("interpolation")
    alpha, weights = base_config
    token_alpha, token_weight, top_k = token_config
    task = record["task"]
    prior_counts = model["prior"][task]
    prior = (prior_counts[0] + 1.0) / (prior_counts[1] + 2.0)
    best: tuple[tuple[float, int, int], str, float, int] | None = None
    for cluster in _clusters(record):
        names = cluster["arms"]
        mask = sum(1 << ARM_ORDER.index(name) for name in names)
        exact = _smoothed(model["exact"].get((task, mask)), prior, alpha)
        size = _smoothed(model["size"].get((task, len(names))), prior, alpha)
        arm_reliability = sum(
            _smoothed(model["arm"].get((task, name)), prior, alpha) for name in names
        ) / len(names)
        base_score = sum(
            weight * math.log(max(value, 1e-12))
            for weight, value in zip(
                weights, (exact, size, arm_reliability), strict=True
            )
        ) / sum(weights)
        token_deltas = []
        for arm in names:
            arm_prior = _smoothed(model["arm"].get((task, arm)), prior, alpha)
            for token in record["tokens"]:
                correct, total = model["token"].get((task, arm, token), [0, 0])
                token_reliability = (correct + token_alpha * arm_prior) / (
                    total + token_alpha
                )
                token_deltas.append(_logit(token_reliability) - _logit(arm_prior))
        strongest = sorted(token_deltas, key=abs, reverse=True)[:top_k]
        text_score = sum(strongest) / math.sqrt(max(1, len(strongest)))
        score = base_score + token_weight * text_score
        tie_priority = -min(ARM_ORDER.index(name) for name in names)
        key = (score, len(names), tie_priority)
        if best is None or key > best[0]:
            best = (key, names[0], text_score, mask)
    if best is None:
        return "hierarchy", 0.0, 0.0, 1 << ARM_ORDER.index("hierarchy")
    return best[1], best[0][0], best[2], best[3]


def _select_token_config(
    records_by_shard: dict[int, list[dict[str, Any]]],
    outer: int,
    base_config: tuple[float, tuple[float, float, float]],
) -> tuple[tuple[float, float, int], int]:
    scores = {config: 0 for config in TOKEN_CONFIGS}
    for inner in sorted(records_by_shard):
        if inner == outer:
            continue
        model = _fit_text(
            [
                row
                for shard in sorted(records_by_shard)
                if shard not in (outer, inner)
                for row in records_by_shard[shard]
            ]
        )
        for config in TOKEN_CONFIGS:
            for record in records_by_shard[inner]:
                selected, _, _, _ = _choose(record, model, base_config, config)
                scores[config] += int(record["arms"][selected]["correct"])
    selected = max(
        TOKEN_CONFIGS,
        key=lambda config: (scores[config], -TOKEN_CONFIGS.index(config)),
    )
    return selected, scores[selected]


def run(
    candidate_paths: dict[str, list[Path]],
    score_paths: dict[str, list[Path]],
    source_path: Path,
    output: Path,
    report_path: Path,
) -> dict[str, Any]:
    records = _load_inputs(candidate_paths, score_paths)
    sources, source_sha256 = _load_sources(source_path)
    if set(sources) != set(records):
        raise Q36MTRTextPatternConsensusError("source identity coverage differs")
    for identity, record in records.items():
        record["tokens"] = sources[identity]
    records_by_shard = {
        shard: [row for row in records.values() if row["shard"] == shard]
        for shard in range(SHARDS)
    }
    outer_configs = {}
    for outer in range(SHARDS):
        base_config, base_correct = _select_outer_config(records_by_shard, outer)
        token_config, token_correct = _select_token_config(
            records_by_shard, outer, base_config
        )
        outer_configs[outer] = (
            base_config,
            base_correct,
            token_config,
            token_correct,
        )
    models = {
        outer: _fit_text(
            [
                row
                for shard in range(SHARDS)
                if shard != outer
                for row in records_by_shard[shard]
            ]
        )
        for outer in range(SHARDS)
    }
    outputs = []
    selections: Counter[tuple[str, str]] = Counter()
    correct = 0
    for identity in sorted(records):
        record = records[identity]
        base_config, _, token_config, _ = outer_configs[record["shard"]]
        selected, score, text_score, mask = _choose(
            record, models[record["shard"]], base_config, token_config
        )
        source = record["arms"][selected]
        row = dict(source["row"])
        row["text_pattern_consensus"] = {
            "schema": SCHEMA,
            "selected": selected,
            "outer_shard": record["shard"],
            "agreement_mask": mask,
            "combined_score": score,
            "text_score": text_score,
            "base_alpha": base_config[0],
            "base_weights": list(base_config[1]),
            "token_alpha": token_config[0],
            "token_weight": token_config[1],
            "token_top_k": token_config[2],
            "outer_training_shards": SHARDS - 1,
            "inner_training_shards_per_validation": SHARDS - 2,
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
        "heldout_identity_labels_read": 0,
        "development_labels_used_for_training": True,
        "hyperparameters_selected_inside_each_outer_training_fold": True,
        "source_sha256": source_sha256,
        "tokenizer": {
            "pattern": "[a-z]{3,}",
            "max_unique_tokens": MAX_TOKENS,
            "stopwords": sorted(STOPWORDS),
        },
        "base_configurations": len(CONFIGS),
        "token_configurations": len(TOKEN_CONFIGS),
        "outer_configs": {
            str(shard): {
                "base_alpha": values[0][0],
                "base_weights": list(values[0][1]),
                "base_inner_correct": values[1],
                "token_alpha": values[2][0],
                "token_weight": values[2][1],
                "token_top_k": values[2][2],
                "text_inner_correct": values[3],
            }
            for shard, values in outer_configs.items()
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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = {arm: getattr(args, arm) for arm in ARM_ORDER}
    scores = {arm: getattr(args, f"{arm}_score") for arm in ARM_ORDER}
    print(
        json.dumps(
            run(candidates, scores, args.source, args.output, args.report),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
