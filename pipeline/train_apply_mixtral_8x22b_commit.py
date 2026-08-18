#!/usr/bin/env python3
"""Train and apply the task-label-free Mixtral whole-trajectory commit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any

ARMS = ("unchanged", "self_refinement", "revision")
TASKS = ("math500", "bbh_logic", "mbpp")
CANDIDATE_SCHEMA = "shohin-mixtral-8x22b-fixed-draft-candidate-v1"
EVALUATION_SCHEMA = "shohin-mixtral-8x22b-bf16-tp4-matched-evaluation-v1"
SCREEN_SCORE_SCHEMA = "shohin-mixtral-8x22b-fixed-draft-screen-score-v1"
MODEL_SCHEMA = "shohin-mixtral-8x22b-selective-commit-model-v1"
SELECTION_SCHEMA = "shohin-mixtral-8x22b-selective-commit-selection-v1"
CANDIDATE_OUTPUT_SCHEMA = "shohin-mixtral-8x22b-selective-commit-candidate-v1"
TRAINING_REPORT_SCHEMA = "shohin-mixtral-8x22b-selective-commit-training-v1"
APPLICATION_REPORT_SCHEMA = "shohin-mixtral-8x22b-selective-commit-application-v1"
SCREEN_ROWS = 256
SCREEN_SHARDS = 4
VALIDATION_ROWS = 1023
VALIDATION_SHARDS = 16
SCREEN_SOURCE_SHA256 = (
    "f0b7830814762c6917363642e86edaaf192a8ab2834911c13c0cae9255ceefa9"
)
VALIDATION_SOURCE_SHA256 = (
    "98c25465916f6275c49ccf9cec67db1236cf0c795db67246a774ea392c0cb778"
)
SCREEN_SCORE_SHA256 = "ce51617197a9f8e9a8ffdfa08d900746bf7c6cf3c34898d941ebe004f6cc4e50"
FEATURE_DIMENSION = 1 << 16
LEARNING_RATE = 0.15
EPOCHS = 4
COMMIT_MARGIN = 0.4
FOLDS = 8
SEED = 2026080816
TOKEN_RE = re.compile(r"[A-Za-z_]+|\d+(?:\.\d+)?|[^\s]")


class MixtralCommitError(RuntimeError):
    """The selective-commit data, model, or application differed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MixtralCommitError(f"missing or linked JSON input: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MixtralCommitError(f"unreadable JSON input: {path}") from error
    if not isinstance(payload, dict):
        raise MixtralCommitError(f"JSON object expected: {path}")
    return payload


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise MixtralCommitError(f"missing or linked JSONL input: {path}")
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise MixtralCommitError(f"unreadable JSONL input: {path}") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise MixtralCommitError(f"JSONL rows differ: {path}")
    return rows


def _atomic_json(path: Path, payload: dict[str, Any]) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    return _atomic_bytes(path, encoded)


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    encoded = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode() for row in rows
    )
    return _atomic_bytes(path, encoded)


def _atomic_bytes(path: Path, encoded: bytes) -> str:
    if path.exists() or path.is_symlink():
        raise MixtralCommitError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(encoded).hexdigest()


def load_sources(
    path: Path, *, rows: int, expected_sha256: str
) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise MixtralCommitError("source hash differs")
    result = _jsonl(path)
    identities: set[str] = set()
    for row in result:
        identity = row.get("identity_sha256")
        if (
            not isinstance(identity, str)
            or len(identity) != 64
            or identity in identities
            or row.get("task") not in TASKS
            or not isinstance(row.get("source_prompt"), str)
            or not row["source_prompt"].strip()
        ):
            raise MixtralCommitError("source row differs")
        identities.add(identity)
    if len(result) != rows:
        raise MixtralCommitError("source row count differs")
    return result


def load_candidates(
    root: Path,
    sources: list[dict[str, Any]],
    *,
    shards: int,
    expected_source_sha256: str,
    expected_split: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    if root.is_symlink() or not root.is_dir():
        raise MixtralCommitError("candidate root differs")
    expected_identities = [row["identity_sha256"] for row in sources]
    expected_tasks = [row["task"] for row in sources]
    candidates: dict[str, list[dict[str, Any]]] = {}
    receipts: dict[str, list[str]] = {}
    for arm in ARMS:
        arm_rows: list[dict[str, Any]] = []
        arm_receipts: list[str] = []
        for shard_index in range(shards):
            shard = root / arm / f"shard_{shard_index:02d}"
            candidate_path = shard / "candidates.jsonl"
            report_path = shard / "report.json"
            report = _json(report_path)
            rows = _jsonl(candidate_path)
            shard_size = (len(sources) + shards - 1) // shards
            start = min(len(sources), shard_index * shard_size)
            end = min(len(sources), start + shard_size)
            if (
                report.get("schema") != EVALUATION_SCHEMA
                or report.get("status") != "complete"
                or report.get("arm") != arm
                or report.get("split") != expected_split
                or report.get("source_sha256") != expected_source_sha256
                or report.get("shard_index") != shard_index
                or report.get("shard_count") != shards
                or report.get("row_start") != start
                or report.get("row_end") != end
                or report.get("full_row_count") != len(sources)
                or report.get("candidates_sha256") != sha256_file(candidate_path)
                or report.get("assessor_access_count") != 0
                or report.get("development_labels_read") != 0
                or report.get("sealed_access")
                != {"holdout": 0, "product": 0, "public": 0}
                or len(rows) != end - start
            ):
                raise MixtralCommitError("candidate report differs")
            for offset, row in enumerate(rows, start=start):
                if (
                    row.get("schema") != CANDIDATE_SCHEMA
                    or row.get("arm") != arm
                    or row.get("identity_sha256") != expected_identities[offset]
                    or row.get("task") != expected_tasks[offset]
                    or not isinstance(row.get("completion"), str)
                    or not row["completion"].strip()
                    or isinstance(row.get("generated_tokens"), bool)
                    or not isinstance(row.get("generated_tokens"), int)
                    or row["generated_tokens"] <= 0
                    or not isinstance(row.get("max_token_exhausted"), bool)
                ):
                    raise MixtralCommitError("candidate row differs")
            arm_rows.extend(rows)
            arm_receipts.append(sha256_file(report_path))
        if len(arm_rows) != len(sources):
            raise MixtralCommitError("candidate coverage differs")
        candidates[arm] = arm_rows
        receipts[arm] = arm_receipts
    return candidates, receipts


def _tokens(text: str, maximum: int) -> list[str]:
    values = [token.lower() for token in TOKEN_RE.findall(text)]
    if len(values) <= maximum:
        return values
    half = maximum // 2
    return values[:half] + values[-half:]


def _add(features: dict[int, float], name: str, value: float = 1.0) -> None:
    digest = hashlib.blake2b(name.encode(), digest_size=8).digest()
    raw = int.from_bytes(digest, "little")
    index = raw & (FEATURE_DIMENSION - 1)
    sign = 1.0 if raw >> 63 == 0 else -1.0
    features[index] = features.get(index, 0.0) + sign * value


def candidate_features(
    source_prompt: str, arm: str, candidate: dict[str, Any]
) -> dict[int, float]:
    """Return label-free features; task/benchmark metadata is not accepted."""

    if arm not in ARMS:
        raise MixtralCommitError("commit arm differs")
    completion = candidate["completion"]
    source_tokens = _tokens(source_prompt, 160)
    completion_tokens = _tokens(completion, 192)
    features: dict[int, float] = {}
    _add(features, "bias")
    _add(features, f"arm:{arm}")
    source_scale = 1.0 / math.sqrt(max(1, len(source_tokens)))
    completion_scale = 1.0 / math.sqrt(max(1, len(completion_tokens)))
    for token in source_tokens:
        _add(features, f"source:{token}", source_scale)
    for token in completion_tokens:
        _add(features, f"completion:{token}", completion_scale)
        _add(features, f"arm_completion:{arm}:{token}", completion_scale)
    for left, right in zip(completion_tokens, completion_tokens[1:]):
        _add(features, f"arm_bigram:{arm}:{left}\0{right}", completion_scale)
    markers = {
        "source_python_shape": bool(
            re.search(
                r"\b(def|function|python|return|list|integer|array|string)\b",
                source_prompt,
                re.IGNORECASE,
            )
        ),
        "source_creation_shape": bool(
            re.search(
                r"\b(write|implement|program|code)\b", source_prompt, re.IGNORECASE
            )
        ),
        "completion_def": bool(
            re.search(r"^\s*def\s+\w+\s*\(", completion, re.MULTILINE)
        ),
        "completion_code_fence": "```" in completion,
        "completion_boxed": "\\boxed" in completion,
        "completion_explicit_final": bool(
            re.search(r"\b(final answer|answer is)\b", completion, re.IGNORECASE)
        ),
        "max_token_exhausted": candidate["max_token_exhausted"],
    }
    for name, value in markers.items():
        _add(features, f"marker:{name}", float(value))
        _add(features, f"arm_marker:{arm}:{name}", float(value))
    numeric = {
        "source_log_chars": math.log1p(len(source_prompt)) / 10.0,
        "completion_log_chars": math.log1p(len(completion)) / 10.0,
        "completion_log_tokens": math.log1p(len(completion_tokens)) / 8.0,
        "generated_fraction": min(2.0, candidate["generated_tokens"] / 512.0),
        "source_completion_length_ratio": min(
            4.0, len(completion) / max(1, len(source_prompt))
        ),
    }
    for name, value in numeric.items():
        _add(features, f"numeric:{name}", value)
        _add(features, f"arm_numeric:{arm}:{name}", value)
    norm = math.sqrt(sum(value * value for value in features.values()))
    if not math.isfinite(norm) or norm <= 0:
        raise MixtralCommitError("commit feature norm differs")
    return {index: value / norm for index, value in features.items()}


def _probability(weights: dict[int, float], features: dict[int, float]) -> float:
    score = sum(weights.get(index, 0.0) * value for index, value in features.items())
    score = max(-30.0, min(30.0, score))
    return 1.0 / (1.0 + math.exp(-score))


def fit(
    rows: list[dict[str, Any]],
    *,
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
) -> dict[int, float]:
    weights: dict[int, float] = {}
    accumulators: dict[int, float] = {}
    order = [
        (row_index, arm_index)
        for row_index in range(len(rows))
        for arm_index in range(3)
    ]
    generator = random.Random(SEED)
    for _ in range(epochs):
        generator.shuffle(order)
        for row_index, arm_index in order:
            row = rows[row_index]
            features = row["features"][arm_index]
            probability = _probability(weights, features)
            coefficient = float(row["correct"][arm_index]) - probability
            for index, value in features.items():
                gradient = coefficient * value
                accumulator = accumulators.get(index, 1e-6) + gradient * gradient
                accumulators[index] = accumulator
                weights[index] = weights.get(index, 0.0) + (
                    learning_rate * gradient / math.sqrt(accumulator)
                )
    return weights


def select_arm(
    weights: dict[int, float], features: list[dict[int, float]]
) -> tuple[int, list[float]]:
    probabilities = [_probability(weights, item) for item in features]
    adapted = 1 if probabilities[1] >= probabilities[2] else 2
    selected = (
        adapted if probabilities[adapted] - probabilities[0] >= COMMIT_MARGIN else 0
    )
    return selected, probabilities


def _training_rows(
    sources: list[dict[str, Any]],
    candidates: dict[str, list[dict[str, Any]]],
    score: dict[str, Any],
) -> list[dict[str, Any]]:
    outcomes = score.get("outcomes")
    if (
        score.get("schema") != SCREEN_SCORE_SCHEMA
        or score.get("status") != "complete"
        or score.get("rows") != SCREEN_ROWS
        or not isinstance(outcomes, list)
        or len(outcomes) != SCREEN_ROWS
    ):
        raise MixtralCommitError("screen score differs")
    by_identity = {row.get("identity_sha256"): row for row in outcomes}
    if len(by_identity) != SCREEN_ROWS:
        raise MixtralCommitError("screen outcome coverage differs")
    result = []
    for row_index, source in enumerate(sources):
        identity = source["identity_sha256"]
        outcome = by_identity.get(identity)
        correct = outcome.get("correct") if isinstance(outcome, dict) else None
        if (
            not isinstance(correct, dict)
            or set(correct) != set(ARMS)
            or any(not isinstance(correct[arm], bool) for arm in ARMS)
            or outcome.get("task") != source["task"]
        ):
            raise MixtralCommitError("screen outcome row differs")
        result.append(
            {
                "identity_sha256": identity,
                "task": source["task"],
                "correct": [correct[arm] for arm in ARMS],
                "features": [
                    candidate_features(
                        source["source_prompt"], arm, candidates[arm][row_index]
                    )
                    for arm in ARMS
                ],
            }
        )
    return result


def _metrics(rows: list[dict[str, Any]], selections: list[int]) -> dict[str, Any]:
    correct = 0
    unchanged_correct = 0
    retained = 0
    wins = 0
    losses = 0
    domains: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    counts: Counter[str] = Counter()
    for row, selected in zip(rows, selections, strict=True):
        selected_correct = row["correct"][selected]
        baseline_correct = row["correct"][0]
        correct += int(selected_correct)
        unchanged_correct += int(baseline_correct)
        retained += int(baseline_correct and selected_correct)
        wins += int(selected_correct and not baseline_correct)
        losses += int(baseline_correct and not selected_correct)
        counts[ARMS[selected]] += 1
        domains[row["task"]][0] += int(selected_correct)
        domains[row["task"]][1] += 1
    return {
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows),
        "unchanged_correct": unchanged_correct,
        "unchanged_correct_retained": retained,
        "unchanged_correct_retention": retained / unchanged_correct,
        "wins_over_unchanged": wins,
        "losses_from_unchanged": losses,
        "selection_counts": dict(sorted(counts.items())),
        "domains": {
            task: {"correct": values[0], "total": values[1]}
            for task, values in sorted(domains.items())
        },
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    if sha256_file(args.screen_score) != SCREEN_SCORE_SHA256:
        raise MixtralCommitError("screen score hash differs")
    sources = load_sources(
        args.screen_source,
        rows=SCREEN_ROWS,
        expected_sha256=SCREEN_SOURCE_SHA256,
    )
    candidates, candidate_receipts = load_candidates(
        args.candidate_root,
        sources,
        shards=SCREEN_SHARDS,
        expected_source_sha256=SCREEN_SOURCE_SHA256,
        expected_split="external_validation_screen",
    )
    score = _json(args.screen_score)
    rows = _training_rows(sources, candidates, score)
    oof_selected = [0] * len(rows)
    oof_probabilities: list[list[float] | None] = [None] * len(rows)
    for fold in range(FOLDS):
        training_indices = [
            index
            for index, row in enumerate(rows)
            if int(row["identity_sha256"][:8], 16) % FOLDS != fold
        ]
        validation_indices = [
            index
            for index, row in enumerate(rows)
            if int(row["identity_sha256"][:8], 16) % FOLDS == fold
        ]
        if not training_indices or not validation_indices:
            raise MixtralCommitError("cross-fit fold geometry differs")
        weights = fit([rows[index] for index in training_indices])
        for index in validation_indices:
            selected, probabilities = select_arm(weights, rows[index]["features"])
            oof_selected[index] = selected
            oof_probabilities[index] = probabilities
    oof_metrics = _metrics(rows, oof_selected)
    oof_rows = [
        {
            "schema": SELECTION_SCHEMA,
            "split": "screen_out_of_fold",
            "identity_sha256": row["identity_sha256"],
            "task": row["task"],
            "fold": int(row["identity_sha256"][:8], 16) % FOLDS,
            "selected_arm": ARMS[selected],
            "probabilities": probabilities,
            "correct": row["correct"][selected],
            "unchanged_correct": row["correct"][0],
        }
        for row, selected, probabilities in zip(
            rows, oof_selected, oof_probabilities, strict=True
        )
    ]
    oof_sha256 = _atomic_lines(args.oof_selections, oof_rows)
    weights = fit(rows)
    model = {
        "schema": MODEL_SCHEMA,
        "status": "complete",
        "feature_contract": "task_label_free_hashed_source_and_complete_trajectory_v1",
        "feature_dimension": FEATURE_DIMENSION,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "commit_margin": COMMIT_MARGIN,
        "seed": SEED,
        "arms": list(ARMS),
        "screen_source_sha256": SCREEN_SOURCE_SHA256,
        "screen_score_sha256": SCREEN_SCORE_SHA256,
        "screen_candidate_report_sha256s": candidate_receipts,
        "screen_rows": SCREEN_ROWS,
        "nonzero_weights": [[index, weights[index]] for index in sorted(weights)],
        "oof_metrics": oof_metrics,
        "oof_selections_sha256": oof_sha256,
        "validation_labels_read": 0,
        "task_label_used_as_feature": False,
    }
    model_sha256 = _atomic_json(args.model_output, model)
    report = {
        "schema": TRAINING_REPORT_SCHEMA,
        "status": "complete",
        "model_sha256": model_sha256,
        "model": str(args.model_output.resolve()),
        "oof_selections_sha256": oof_sha256,
        "oof_metrics": oof_metrics,
        "hyperparameters": {
            "feature_dimension": FEATURE_DIMENSION,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "commit_margin": COMMIT_MARGIN,
            "folds": FOLDS,
            "seed": SEED,
        },
        "task_label_used_as_feature": False,
        "validation_labels_read": 0,
    }
    _atomic_json(args.report, report)
    return report


def _load_model(path: Path) -> tuple[dict[str, Any], dict[int, float]]:
    model = _json(path)
    expected = {
        "schema": MODEL_SCHEMA,
        "status": "complete",
        "feature_contract": "task_label_free_hashed_source_and_complete_trajectory_v1",
        "feature_dimension": FEATURE_DIMENSION,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "commit_margin": COMMIT_MARGIN,
        "seed": SEED,
        "arms": list(ARMS),
        "screen_source_sha256": SCREEN_SOURCE_SHA256,
        "screen_score_sha256": SCREEN_SCORE_SHA256,
        "screen_rows": SCREEN_ROWS,
        "validation_labels_read": 0,
        "task_label_used_as_feature": False,
    }
    if any(model.get(key) != value for key, value in expected.items()):
        raise MixtralCommitError("commit model contract differs")
    raw_weights = model.get("nonzero_weights")
    if not isinstance(raw_weights, list):
        raise MixtralCommitError("commit weights differ")
    weights: dict[int, float] = {}
    for item in raw_weights:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or isinstance(item[0], bool)
            or not isinstance(item[0], int)
            or not 0 <= item[0] < FEATURE_DIMENSION
            or item[0] in weights
            or isinstance(item[1], bool)
            or not isinstance(item[1], (int, float))
            or not math.isfinite(item[1])
        ):
            raise MixtralCommitError("commit weight row differs")
        weights[item[0]] = float(item[1])
    if not weights:
        raise MixtralCommitError("commit model is empty")
    return model, weights


def apply(args: argparse.Namespace) -> dict[str, Any]:
    model, weights = _load_model(args.model)
    sources = load_sources(
        args.source,
        rows=VALIDATION_ROWS,
        expected_sha256=VALIDATION_SOURCE_SHA256,
    )
    candidates, candidate_receipts = load_candidates(
        args.candidate_root,
        sources,
        shards=VALIDATION_SHARDS,
        expected_source_sha256=VALIDATION_SOURCE_SHA256,
        expected_split="external_validation_confirmation",
    )
    selected_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    model_sha256 = sha256_file(args.model)
    for row_index, source in enumerate(sources):
        features = [
            candidate_features(source["source_prompt"], arm, candidates[arm][row_index])
            for arm in ARMS
        ]
        selected, probabilities = select_arm(weights, features)
        arm = ARMS[selected]
        candidate = candidates[arm][row_index]
        selected_rows.append(
            {
                "schema": CANDIDATE_OUTPUT_SCHEMA,
                "arm": "selective_commit",
                "selected_arm": arm,
                "identity_sha256": source["identity_sha256"],
                "task": source["task"],
                "completion": candidate["completion"],
                "generated_tokens": candidate["generated_tokens"],
                "max_token_exhausted": candidate["max_token_exhausted"],
                "model_sha256": model_sha256,
            }
        )
        selection_rows.append(
            {
                "schema": SELECTION_SCHEMA,
                "split": "external_validation_confirmation",
                "identity_sha256": source["identity_sha256"],
                "selected_arm": arm,
                "probabilities": probabilities,
                "model_sha256": model_sha256,
            }
        )
        counts[arm] += 1
    output_sha256 = _atomic_lines(args.output, selected_rows)
    selections_sha256 = _atomic_lines(args.selections, selection_rows)
    report = {
        "schema": APPLICATION_REPORT_SCHEMA,
        "status": "complete",
        "rows": VALIDATION_ROWS,
        "source_sha256": VALIDATION_SOURCE_SHA256,
        "candidate_report_sha256s": candidate_receipts,
        "model": str(args.model.resolve()),
        "model_sha256": model_sha256,
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
        "selections": str(args.selections.resolve()),
        "selections_sha256": selections_sha256,
        "selection_counts": dict(sorted(counts.items())),
        "assessor_access_count": 0,
        "validation_labels_read": 0,
        "task_label_used_as_feature": False,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
        "model_oof_metrics": model["oof_metrics"],
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--screen-source", type=Path, required=True)
    train_parser.add_argument("--screen-score", type=Path, required=True)
    train_parser.add_argument("--candidate-root", type=Path, required=True)
    train_parser.add_argument("--model-output", type=Path, required=True)
    train_parser.add_argument("--oof-selections", type=Path, required=True)
    train_parser.add_argument("--report", type=Path, required=True)
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--model", type=Path, required=True)
    apply_parser.add_argument("--source", type=Path, required=True)
    apply_parser.add_argument("--candidate-root", type=Path, required=True)
    apply_parser.add_argument("--output", type=Path, required=True)
    apply_parser.add_argument("--selections", type=Path, required=True)
    apply_parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = train(args) if args.mode == "train" else apply(args)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
