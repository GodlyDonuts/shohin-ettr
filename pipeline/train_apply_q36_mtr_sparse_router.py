#!/usr/bin/env python3
"""Train and apply a calibration-only sparse router over three Q36 trajectories."""

from __future__ import annotations

import argparse
from array import array
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any

ROW_SCHEMA = "shohin-q36-mtr-setwise-commit-row-v1"
CANDIDATE_SCHEMA = "shohin-q36-mtr-model-draft-v1"
SELECTION_SCHEMA = "shohin-q36-mtr-sparse-router-selection-v1"
MODEL_SCHEMA = "shohin-q36-mtr-sparse-router-model-v1"
REPORT_SCHEMA = "shohin-q36-mtr-sparse-router-report-v1"
LINEAGES = ("current", "owner_71", "owner_8")
TASKS = ("math500", "bbh_logic", "mbpp")
CALIBRATION_SPLITS = ("calibration_train", "calibration_development")
TRAIN_ROWS = 5_824
DEVELOPMENT_ROWS = 1_289
DIMENSION = 1 << 16
SEED = 2026080820
TOKEN_RE = re.compile(r"[A-Za-z_]+|\d+(?:\.\d+)?|[^\s]")
EXPLICIT_RE = re.compile(
    r"(?:the\s+answer\s+is|final\s+answer|\\boxed\s*\{|^\s*def\s+\w+\s*\()",
    re.IGNORECASE | re.MULTILINE,
)


class Q36MTRSparseRouterError(RuntimeError):
    """Sparse-router inputs, training, or output geometry differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise Q36MTRSparseRouterError(f"missing or linked input: {path}")
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise Q36MTRSparseRouterError(f"unreadable input: {path}") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise Q36MTRSparseRouterError(f"empty or malformed input: {path}")
    return rows


def load_training_rows(path: Path) -> list[dict[str, Any]]:
    rows = _jsonl(path)
    identities: set[str] = set()
    patterns: Counter[str] = Counter()
    for row in rows:
        identity = row.get("identity_sha256")
        candidates = row.get("candidates")
        if (
            row.get("schema") != ROW_SCHEMA
            or row.get("split") not in CALIBRATION_SPLITS
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in identities
            or row.get("task") not in TASKS
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
            or not isinstance(candidates, list)
            or len(candidates) != 3
            or [candidate.get("lineage") for candidate in candidates] != list(LINEAGES)
        ):
            raise Q36MTRSparseRouterError("sparse-router training row differs")
        correctness: list[bool] = []
        for candidate in candidates:
            if (
                not isinstance(candidate.get("completion"), str)
                or not candidate["completion"].strip()
                or not isinstance(candidate.get("correct"), bool)
                or isinstance(candidate.get("generated_tokens"), bool)
                or not isinstance(candidate.get("generated_tokens"), int)
                or candidate["generated_tokens"] < 0
                or not isinstance(candidate.get("max_token_exhausted"), bool)
            ):
                raise Q36MTRSparseRouterError(
                    "sparse-router training candidate differs"
                )
            correctness.append(candidate["correct"])
        pattern = "".join("1" if value else "0" for value in correctness)
        if row.get("correctness_pattern") != pattern:
            raise Q36MTRSparseRouterError("sparse-router correctness differs")
        identities.add(identity)
        patterns[pattern] += 1
    if (
        len(rows) != TRAIN_ROWS
        or {row["split"] for row in rows} != set(CALIBRATION_SPLITS)
        or set(patterns) != {f"{index:03b}" for index in range(8)}
    ):
        raise Q36MTRSparseRouterError("sparse-router calibration geometry differs")
    return rows


def load_development_candidates(
    paths: list[Path], *, expected_shards: int = 16
) -> dict[str, dict[str, Any]]:
    if len(paths) != expected_shards:
        raise Q36MTRSparseRouterError("sparse-router candidate shard count differs")
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _jsonl(path):
            if row.get("split") != "development":
                continue
            identity = row.get("identity_sha256")
            if (
                row.get("schema") != CANDIDATE_SCHEMA
                or not isinstance(identity, str)
                or len(identity) != 64
                or identity in result
                or row.get("task") not in TASKS
                or not isinstance(row.get("completion"), str)
                or not row["completion"].strip()
                or isinstance(row.get("generated_tokens"), bool)
                or not isinstance(row.get("generated_tokens"), int)
                or row["generated_tokens"] <= 0
                or not isinstance(row.get("max_token_exhausted"), bool)
            ):
                raise Q36MTRSparseRouterError("sparse-router candidate differs")
            result[identity] = row
    if len(result) != DEVELOPMENT_ROWS:
        raise Q36MTRSparseRouterError("sparse-router development coverage differs")
    return result


def load_development_rows(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in _jsonl(path):
        identity = row.get("identity_sha256")
        candidates = row.get("candidates")
        if (
            row.get("schema") != ROW_SCHEMA
            or row.get("split") != "development"
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in result
            or row.get("task") not in TASKS
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
            or not isinstance(candidates, list)
            or len(candidates) != 3
            or [candidate.get("lineage") for candidate in candidates] != list(LINEAGES)
            or any("correct" in candidate for candidate in candidates)
        ):
            raise Q36MTRSparseRouterError("sparse-router development row differs")
        result[identity] = row
    if len(result) != DEVELOPMENT_ROWS:
        raise Q36MTRSparseRouterError("sparse-router development rows differ")
    return result


def _tokens(text: str, maximum: int) -> list[str]:
    values = [token.lower() for token in TOKEN_RE.findall(text)]
    if len(values) <= maximum:
        return values
    half = maximum // 2
    return values[:half] + values[-half:]


def _hashed(name: str) -> tuple[int, float]:
    digest = hashlib.blake2b(name.encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value & (DIMENSION - 1), 1.0 if value >> 63 == 0 else -1.0


def _add(features: dict[int, float], name: str, value: float = 1.0) -> None:
    index, sign = _hashed(name)
    features[index] = features.get(index, 0.0) + sign * value


def candidate_features(
    question: str, task: str, lineage: str, candidate: dict[str, Any]
) -> dict[int, float]:
    """Project a trajectory into generic and owner-conditioned sparse features."""

    completion = candidate["completion"]
    completion_tokens = _tokens(completion, 128)
    question_tokens = _tokens(question, 64)
    words = max(1, len(TOKEN_RE.findall(completion)))
    overlap = len(set(completion_tokens) & set(question_tokens)) / max(
        1, len(set(question_tokens))
    )
    numeric = {
        "log_chars": math.log1p(len(completion)) / 10.0,
        "log_words": math.log1p(words) / 8.0,
        "log_lines": math.log1p(completion.count("\n") + 1) / 5.0,
        "generated_fraction": min(2.0, candidate.get("generated_tokens", words) / 512),
        "question_overlap": overlap,
        "max_token_exhausted": float(candidate.get("max_token_exhausted", False)),
        "explicit_final": float(bool(EXPLICIT_RE.search(completion))),
        "has_boxed": float("\\boxed" in completion),
        "has_code_fence": float("```" in completion),
    }
    features: dict[int, float] = {}
    for name in (
        "bias",
        f"task:{task}",
        f"owner:{lineage}",
        f"owner_task:{lineage}:{task}",
    ):
        _add(features, name)
    for name, value in numeric.items():
        _add(features, f"numeric:{name}", value)
        _add(features, f"owner_numeric:{lineage}:{name}", value)
        _add(features, f"task_numeric:{task}:{name}", value)
    for token in completion_tokens:
        _add(features, f"completion_unigram:{token}")
        _add(features, f"owner_unigram:{lineage}:{token}")
    for left, right in zip(completion_tokens, completion_tokens[1:]):
        _add(features, f"completion_bigram:{left}\0{right}")
    for token in question_tokens:
        _add(features, f"owner_question:{lineage}:{token}")
    for token in completion_tokens[-32:]:
        _add(features, f"answer_tail:{token}")
        _add(features, f"owner_answer_tail:{lineage}:{token}")
    norm = math.sqrt(sum(value * value for value in features.values()))
    if not math.isfinite(norm) or norm <= 0:
        raise Q36MTRSparseRouterError("sparse-router feature norm differs")
    return {index: value / norm for index, value in features.items()}


def _score(weights: array, features: dict[int, float]) -> float:
    return sum(weights[index] * value for index, value in features.items())


def _softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    scaled = [math.exp(value - maximum) for value in values]
    total = sum(scaled)
    return [value / total for value in scaled]


def _selection_correct(weights: array, rows: list[dict[str, Any]]) -> int:
    correct = 0
    for row in rows:
        scores = [_score(weights, features) for features in row["_features"]]
        selected = max(range(3), key=lambda index: (scores[index], -index))
        correct += int(row["candidates"][selected]["correct"])
    return correct


def _fit(
    rows: list[dict[str, Any]], *, learning_rate: float, balanced: bool, epochs: int
) -> tuple[array, list[dict[str, Any]]]:
    weights = array("d", [0.0]) * DIMENSION
    accumulators = array("d", [1e-6]) * DIMENSION
    patterns = Counter(row["correctness_pattern"] for row in rows)
    mixed_patterns = {pattern for pattern in patterns if pattern not in {"000", "111"}}
    generator = random.Random(SEED)
    history: list[dict[str, Any]] = []
    order = list(range(len(rows)))
    for epoch in range(1, epochs + 1):
        generator.shuffle(order)
        updates = 0
        for row_index in order:
            row = rows[row_index]
            correctness = [candidate["correct"] for candidate in row["candidates"]]
            correct_count = sum(correctness)
            if correct_count in {0, 3}:
                continue
            scores = [_score(weights, features) for features in row["_features"]]
            probabilities = _softmax(scores)
            row_weight = 1.0
            if balanced:
                row_weight = min(
                    4.0,
                    sum(patterns[name] for name in mixed_patterns)
                    / (len(mixed_patterns) * patterns[row["correctness_pattern"]]),
                )
            for candidate_index, features in enumerate(row["_features"]):
                target = float(correctness[candidate_index]) / correct_count
                coefficient = row_weight * (target - probabilities[candidate_index])
                for feature_index, value in features.items():
                    gradient = coefficient * value
                    accumulators[feature_index] += gradient * gradient
                    weights[feature_index] += (
                        learning_rate
                        * gradient
                        / math.sqrt(accumulators[feature_index])
                    )
                updates += 1
        history.append({"epoch": epoch, "updates": updates})
    return weights, history


def train(rows: list[dict[str, Any]]) -> tuple[array, dict[str, Any]]:
    for row in rows:
        row["_features"] = [
            candidate_features(row["question"], row["task"], lineage, candidate)
            for lineage, candidate in zip(LINEAGES, row["candidates"], strict=True)
        ]
    fit_rows = [row for row in rows if row["split"] == "calibration_train"]
    validation = [row for row in rows if row["split"] == "calibration_development"]
    trials: list[dict[str, Any]] = []
    best: tuple[int, float, bool, int, array] | None = None
    for balanced in (False, True):
        for learning_rate in (0.03, 0.07, 0.15):
            weights = array("d", [0.0]) * DIMENSION
            accumulators = array("d", [1e-6]) * DIMENSION
            patterns = Counter(row["correctness_pattern"] for row in fit_rows)
            mixed_patterns = {
                pattern for pattern in patterns if pattern not in {"000", "111"}
            }
            generator = random.Random(SEED)
            order = list(range(len(fit_rows)))
            for epoch in range(1, 13):
                generator.shuffle(order)
                for row_index in order:
                    row = fit_rows[row_index]
                    correctness = [
                        candidate["correct"] for candidate in row["candidates"]
                    ]
                    correct_count = sum(correctness)
                    if correct_count in {0, 3}:
                        continue
                    probabilities = _softmax(
                        [_score(weights, features) for features in row["_features"]]
                    )
                    row_weight = 1.0
                    if balanced:
                        row_weight = min(
                            4.0,
                            sum(patterns[name] for name in mixed_patterns)
                            / (
                                len(mixed_patterns)
                                * patterns[row["correctness_pattern"]]
                            ),
                        )
                    for candidate_index, features in enumerate(row["_features"]):
                        target = float(correctness[candidate_index]) / correct_count
                        coefficient = row_weight * (
                            target - probabilities[candidate_index]
                        )
                        for feature_index, value in features.items():
                            gradient = coefficient * value
                            accumulators[feature_index] += gradient * gradient
                            weights[feature_index] += (
                                learning_rate
                                * gradient
                                / math.sqrt(accumulators[feature_index])
                            )
                validation_correct = _selection_correct(weights, validation)
                trial = {
                    "balanced_patterns": balanced,
                    "learning_rate": learning_rate,
                    "epoch": epoch,
                    "calibration_development_correct": validation_correct,
                    "calibration_development_rows": len(validation),
                }
                trials.append(trial)
                key = (validation_correct, -learning_rate, not balanced, -epoch)
                if best is None or key > (best[0], -best[1], not best[2], -best[3]):
                    best = (
                        validation_correct,
                        learning_rate,
                        balanced,
                        epoch,
                        array("d", weights),
                    )
    if best is None:
        raise Q36MTRSparseRouterError("sparse-router model selection failed")
    _, learning_rate, balanced, best_epoch, _ = best
    final_weights, history = _fit(
        rows,
        learning_rate=learning_rate,
        balanced=balanced,
        epochs=best_epoch,
    )
    for row in rows:
        del row["_features"]
    return final_weights, {
        "seed": SEED,
        "dimension": DIMENSION,
        "model_selection": {
            "learning_rate": learning_rate,
            "balanced_patterns": balanced,
            "epoch": best_epoch,
            "calibration_development_correct": best[0],
            "calibration_development_rows": len(validation),
        },
        "trials": trials,
        "final_training": history,
        "final_training_rows": len(rows),
        "development_labels_read": 0,
    }


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise Q36MTRSparseRouterError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise Q36MTRSparseRouterError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    training_rows = load_training_rows(args.training_rows)
    development_rows = load_development_rows(args.development_rows)
    owners = [
        load_development_candidates(paths)
        for paths in (
            args.current_candidates,
            args.owner71_candidates,
            args.owner8_candidates,
        )
    ]
    if any(set(owner) != set(development_rows) for owner in owners):
        raise Q36MTRSparseRouterError("sparse-router owner identities differ")
    weights, training = train(training_rows)
    nonzero_weights = [
        [index, value] for index, value in enumerate(weights) if value != 0.0
    ]
    model = {
        "schema": MODEL_SCHEMA,
        "status": "complete",
        "training_rows_sha256": sha256_file(args.training_rows),
        "feature_contract": "hashed_question_trajectory_owner_interactions_v1",
        "training": training,
        "nonzero_weights": nonzero_weights,
    }
    model_sha = _atomic_json(args.model_output, model)

    selected_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for identity in sorted(development_rows):
        row = development_rows[identity]
        candidates = [owner[identity] for owner in owners]
        if any(candidate["task"] != row["task"] for candidate in candidates):
            raise Q36MTRSparseRouterError("sparse-router development task differs")
        scores = [
            _score(
                weights,
                candidate_features(row["question"], row["task"], lineage, candidate),
            )
            for lineage, candidate in zip(LINEAGES, candidates, strict=True)
        ]
        selected_index = max(range(3), key=lambda index: (scores[index], -index))
        chosen = dict(candidates[selected_index])
        chosen["sparse_router_selection"] = {
            "schema": SELECTION_SCHEMA,
            "lineages": list(LINEAGES),
            "selected_lineage": LINEAGES[selected_index],
            "scores": scores,
            "model_sha256": model_sha,
        }
        selected_rows.append(chosen)
        selections.append(
            {
                "schema": SELECTION_SCHEMA,
                "identity_sha256": identity,
                "task": row["task"],
                "selected_lineage": LINEAGES[selected_index],
                "scores": scores,
                "model_sha256": model_sha,
            }
        )
        counts[LINEAGES[selected_index]] += 1
    candidates_sha = _atomic_lines(args.output, selected_rows)
    selections_sha = _atomic_lines(args.selections, selections)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "interpretation": "calibration_only_learned_sparse_trajectory_router",
        "rows": len(selected_rows),
        "training_rows": str(args.training_rows.resolve()),
        "training_rows_sha256": sha256_file(args.training_rows),
        "development_rows": str(args.development_rows.resolve()),
        "development_rows_sha256": sha256_file(args.development_rows),
        "development_labels_read": 0,
        "training": training,
        "model": str(args.model_output.resolve()),
        "model_sha256": model_sha,
        "output": str(args.output.resolve()),
        "output_sha256": candidates_sha,
        "selections": str(args.selections.resolve()),
        "selections_sha256": selections_sha,
        "selection_counts": dict(sorted(counts.items())),
        "owner_candidate_sha256": {
            lineage: [sha256_file(path) for path in paths]
            for lineage, paths in zip(
                LINEAGES,
                (
                    args.current_candidates,
                    args.owner71_candidates,
                    args.owner8_candidates,
                ),
                strict=True,
            )
        },
    }
    _atomic_json(args.report, report)
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
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
