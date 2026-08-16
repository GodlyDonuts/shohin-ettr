#!/usr/bin/env python3
"""Fit a development-only five-arm nonlinear selector and score a partition."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from pcf1_code_sandbox import (
    atomic_json as sandbox_atomic_json,
    qualify_allocation,
    qualify_mbpp_assessor_setups,
    score_completion,
)
from score_q36_mtr_external import (
    ARMS,
    CANDIDATE_SCHEMA,
    PARTITIONS,
    TASKS,
    _load_jsonl,
    _mcnemar_exact,
    load_assessors,
    sha256_file,
)
from score_q36_mtr_external_consensus import normalized_answer

SCHEMA = "shohin-q36-mtr-external-forest-score-v1"
SELECTION_SCHEMA = "shohin-q36-mtr-external-forest-selection-v1"
DEVELOPMENT_ROWS = 1_289
DEVELOPMENT_PATHS = {
    "unchanged": 8,
    "self_refinement": 8,
    "revision": 1,
    "draft_hidden": 1,
    "interpolation": 16,
}
OUTER_SHARDS = 16
RANDOM_FEATURES = 128
RIDGE_PENALTY = 1.0e-2
RANDOM_STATE = 2026081436
FEATURE_TASKS = ("bbh_logic", "math500", "mbpp")
RETENTION_THRESHOLDS = {
    "bbh_logic": 0.115,
    "math500": 0.035,
    "mbpp": 0.300,
}


class Q36MTRExternalForestError(RuntimeError):
    """External forest training data, target data, or score differs."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRExternalForestError("external forest output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise Q36MTRExternalForestError("external forest selections exist")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            encoded = json.dumps(row, sort_keys=True) + "\n"
            handle.write(encoded)
            digest.update(encoded.encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def load_candidate_group(
    arm: str,
    paths: list[Path],
    expected_rows: int,
    expected_paths: int,
    *,
    development: bool = False,
) -> dict[str, dict[str, Any]]:
    if arm not in ARMS or len(paths) != expected_paths:
        raise Q36MTRExternalForestError("external forest candidate geometry differs")
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _load_jsonl(path):
            identity = row.get("identity_sha256")
            current_schema = (
                row.get("schema") == CANDIDATE_SCHEMA and row.get("arm") == arm
            )
            legacy_interpolation = (
                development
                and arm == "interpolation"
                and row.get("schema") == "shohin-q36-mtr-model-draft-v1"
                and row.get("arm") is None
            )
            if (
                not (current_schema or legacy_interpolation)
                or row.get("task") not in TASKS
                or not isinstance(identity, str)
                or len(identity) != 64
                or identity in rows
                or not isinstance(row.get("completion"), str)
                or not row["completion"].strip()
                or not isinstance(row.get("generated_tokens"), int)
                or row["generated_tokens"] < 0
                or not isinstance(row.get("max_token_exhausted"), bool)
            ):
                raise Q36MTRExternalForestError("external forest candidate differs")
            rows[identity] = row
    if len(rows) != expected_rows or {row["task"] for row in rows.values()} != set(
        TASKS
    ):
        raise Q36MTRExternalForestError("external forest candidate coverage differs")
    return rows


def load_outcomes(
    arm: str,
    candidate_paths: list[Path],
    score_paths: list[Path],
    identities: set[str],
) -> dict[str, bool]:
    candidate_hashes = {sha256_file(path) for path in candidate_paths}
    score_hashes: set[str] = set()
    outcomes: dict[str, bool] = {}
    for path in score_paths:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise Q36MTRExternalForestError(
                "external forest score unreadable"
            ) from error
        evaluation_arm = report.get("evaluation_arm")
        if (
            report.get("schema") != "shohin-q36-mtr-draft-preview-v1"
            or report.get("status") != "complete"
            or report.get("split") != "development"
            or not isinstance(report.get("outcomes"), list)
            or (
                evaluation_arm != arm
                and not (arm == "interpolation" and evaluation_arm is None)
            )
        ):
            raise Q36MTRExternalForestError("external forest score report differs")
        score_hashes.add(str(report.get("candidates_sha256")))
        for row in report["outcomes"]:
            identity = row.get("identity_sha256")
            if (
                not isinstance(identity, str)
                or identity in outcomes
                or not isinstance(row.get("correct"), bool)
                or row.get("task") not in TASKS
            ):
                raise Q36MTRExternalForestError("external forest score outcome differs")
            outcomes[identity] = row["correct"]
    if score_hashes != candidate_hashes or set(outcomes) != identities:
        raise Q36MTRExternalForestError("external forest score custody differs")
    return outcomes


def _answer(task: str, completion: str) -> str | None:
    try:
        return normalized_answer(task, completion)
    except (ValueError, TypeError):
        return None


def feature_vector(
    identity: str, arm: str, candidates: dict[str, dict[str, dict[str, Any]]]
) -> list[float]:
    rows = {name: candidates[name][identity] for name in ARMS}
    task = rows[arm]["task"]
    if any(row["task"] != task for row in rows.values()):
        raise Q36MTRExternalForestError("external forest task binding differs")
    answers = {name: _answer(task, row["completion"]) for name, row in rows.items()}
    counts = Counter(answer for answer in answers.values() if answer is not None)
    selected_answer = answers[arm]
    values = [float(task == name) for name in FEATURE_TASKS]
    values.extend(float(arm == name) for name in ARMS)
    for name in ARMS:
        row = rows[name]
        values.extend(
            (
                math.log1p(row["generated_tokens"]) / 8.0,
                float(row["max_token_exhausted"]),
                math.log1p(len(row["completion"])) / 10.0,
                float(answers[name] is not None),
            )
        )
    values.extend(
        (
            (
                counts.get(selected_answer, 0) / len(ARMS)
                if selected_answer is not None
                else 0.0
            ),
            max(counts.values(), default=0) / len(ARMS),
            len(counts) / len(ARMS),
            sum(answer is None for answer in answers.values()) / len(ARMS),
        )
    )
    values.extend(
        float(selected_answer is not None and selected_answer == answers[name])
        for name in ARMS
    )
    if selected_answer is None:
        values.extend((0.0, 0.0, 0.0, 0.0))
    else:
        try:
            numeric = float(selected_answer.replace(",", ""))
            is_numeric = math.isfinite(numeric)
        except ValueError:
            numeric = 0.0
            is_numeric = False
        values.extend(
            (
                min(len(selected_answer), 100) / 100.0,
                float(is_numeric),
                float(numeric < 0.0 if is_numeric else False),
                (min(math.log1p(abs(numeric)), 100.0) / 20.0 if is_numeric else 0.0),
            )
        )
    if len(values) != 41 or not all(math.isfinite(value) for value in values):
        raise Q36MTRExternalForestError("external forest feature geometry differs")
    return values


class _RandomFeatureRidge:
    """Small deterministic nonlinear readout requiring only NumPy."""

    def __init__(self, matrix: Any, labels: Any, random_state: int) -> None:
        import numpy as np

        if matrix.ndim != 2 or labels.shape != (matrix.shape[0],):
            raise Q36MTRExternalForestError("external selector fit geometry differs")
        self.mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        self.scale = np.where(scale > 1.0e-6, scale, 1.0)
        normalized = (matrix - self.mean) / self.scale
        generator = np.random.default_rng(random_state)
        self.projection = generator.normal(
            0.0,
            1.0 / math.sqrt(matrix.shape[1]),
            size=(matrix.shape[1], RANDOM_FEATURES),
        )
        self.bias = generator.uniform(-math.pi, math.pi, size=RANDOM_FEATURES)
        hidden = np.tanh(normalized @ self.projection + self.bias)
        design = np.concatenate(
            (np.ones((matrix.shape[0], 1)), normalized, hidden), axis=1
        )
        gram = design.T @ design
        penalty = np.eye(gram.shape[0]) * RIDGE_PENALTY
        penalty[0, 0] = 0.0
        try:
            self.coefficients = np.linalg.solve(gram + penalty, design.T @ labels)
        except np.linalg.LinAlgError as error:
            raise Q36MTRExternalForestError(
                "external selector fit is singular"
            ) from error

    def predict(self, matrix: Any) -> Any:
        import numpy as np

        if matrix.ndim != 2 or matrix.shape[1] != self.mean.shape[0]:
            raise Q36MTRExternalForestError(
                "external selector prediction geometry differs"
            )
        normalized = (matrix - self.mean) / self.scale
        hidden = np.tanh(normalized @ self.projection + self.bias)
        design = np.concatenate(
            (np.ones((matrix.shape[0], 1)), normalized, hidden), axis=1
        )
        return design @ self.coefficients


def _fit(matrix: Any, labels: Any, random_state: int) -> Any:
    return _RandomFeatureRidge(matrix, labels, random_state)


def _choose(
    identity: str,
    predictions: dict[tuple[str, str], float],
    candidates: dict[str, dict[str, dict[str, Any]]],
) -> str:
    best = max(
        ARMS,
        key=lambda arm: (predictions[(identity, arm)], -ARMS.index(arm)),
    )
    task = candidates["unchanged"][identity]["task"]
    advantage = predictions[(identity, best)] - predictions[(identity, "interpolation")]
    return best if advantage > RETENTION_THRESHOLDS[task] else "interpolation"


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    if (
        args.output.exists()
        or args.selections.exists()
        or args.sandbox_receipt.exists()
    ):
        raise Q36MTRExternalForestError("external forest output exists")
    expected = PARTITIONS.get(args.split)
    if expected != (args.expected_rows, args.shard_count):
        raise Q36MTRExternalForestError("external forest partition differs")

    development_paths = {
        arm: getattr(args, f"development_{arm}_candidates") for arm in ARMS
    }
    development = {
        arm: load_candidate_group(
            arm,
            paths,
            DEVELOPMENT_ROWS,
            DEVELOPMENT_PATHS[arm],
            development=True,
        )
        for arm, paths in development_paths.items()
    }
    development_ids = set(development["unchanged"])
    if any(set(rows) != development_ids for rows in development.values()):
        raise Q36MTRExternalForestError("external forest development identities differ")
    labels = {
        arm: load_outcomes(
            arm,
            development_paths[arm],
            getattr(args, f"development_{arm}_scores"),
            development_ids,
        )
        for arm in ARMS
    }

    keys = [(identity, arm) for identity in sorted(development_ids) for arm in ARMS]
    features = {key: feature_vector(key[0], key[1], development) for key in keys}
    outer = {
        identity: int(identity[:8], 16) % OUTER_SHARDS for identity in development_ids
    }
    oof: dict[tuple[str, str], float] = {}
    for shard in range(OUTER_SHARDS):
        train_keys = [key for key in keys if outer[key[0]] != shard]
        test_keys = [key for key in keys if outer[key[0]] == shard]
        model = _fit(
            np.asarray([features[key] for key in train_keys]),
            np.asarray([float(labels[key[1]][key[0]]) for key in train_keys]),
            RANDOM_STATE + shard,
        )
        predictions = model.predict(np.asarray([features[key] for key in test_keys]))
        oof.update({key: float(value) for key, value in zip(test_keys, predictions)})
    development_choices = {
        identity: _choose(identity, oof, development)
        for identity in sorted(development_ids)
    }
    development_correct = sum(
        labels[development_choices[identity]][identity] for identity in development_ids
    )
    interpolation_correct = sum(labels["interpolation"].values())

    target_paths = {arm: getattr(args, f"target_{arm}_candidates") for arm in ARMS}
    target = {
        arm: load_candidate_group(arm, paths, args.expected_rows, args.shard_count)
        for arm, paths in target_paths.items()
    }
    target_ids = set(target["unchanged"])
    if any(set(rows) != target_ids for rows in target.values()):
        raise Q36MTRExternalForestError("external forest target identities differ")
    final_model = _fit(
        np.asarray([features[key] for key in keys]),
        np.asarray([float(labels[key[1]][key[0]]) for key in keys]),
        RANDOM_STATE,
    )
    target_keys = [(identity, arm) for identity in sorted(target_ids) for arm in ARMS]
    target_matrix = np.asarray(
        [feature_vector(key[0], key[1], target) for key in target_keys]
    )
    target_predictions = {
        key: float(value)
        for key, value in zip(target_keys, final_model.predict(target_matrix))
    }
    choices = {
        identity: _choose(identity, target_predictions, target)
        for identity in sorted(target_ids)
    }

    assessors = load_assessors(args.assessors, args.expected_rows)
    if set(assessors) != target_ids:
        raise Q36MTRExternalForestError("external forest assessor coverage differs")
    sandbox = qualify_allocation()
    sandbox_sha256 = sandbox_atomic_json(args.sandbox_receipt, sandbox)
    setup_receipts = qualify_mbpp_assessor_setups(
        [row["assessor"] for row in assessors.values() if row["task"] == "mbpp"]
    )
    correct: dict[str, bool] = {}
    baseline_correct: dict[str, bool] = {}
    task_correct: Counter[str] = Counter()
    task_rows = Counter(row["task"] for row in assessors.values())
    selections: list[dict[str, Any]] = []
    for identity in sorted(target_ids):
        selected = choices[identity]
        candidate = target[selected][identity]
        baseline = target["interpolation"][identity]
        result = score_completion(
            assessors[identity]["assessor"], candidate["completion"]
        )
        baseline_result = score_completion(
            assessors[identity]["assessor"], baseline["completion"]
        )
        if not isinstance(result.get("correct"), bool) or not isinstance(
            baseline_result.get("correct"), bool
        ):
            raise Q36MTRExternalForestError("external forest score differs")
        correct[identity] = result["correct"]
        baseline_correct[identity] = baseline_result["correct"]
        task_correct[candidate["task"]] += int(result["correct"])
        selections.append(
            {
                "schema": SELECTION_SCHEMA,
                "identity_sha256": identity,
                "task": candidate["task"],
                "selected_arm": selected,
                "selected_prediction": target_predictions[(identity, selected)],
                "interpolation_prediction": target_predictions[
                    (identity, "interpolation")
                ],
                "completion": candidate["completion"],
            }
        )
    selections_sha256 = _atomic_lines(args.selections, selections)
    selector_only = sum(
        correct[identity] and not baseline_correct[identity] for identity in target_ids
    )
    baseline_only = sum(
        baseline_correct[identity] and not correct[identity] for identity in target_ids
    )
    total_correct = sum(correct.values())
    baseline_total = sum(baseline_correct.values())
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "split": args.split,
        "rows": args.expected_rows,
        "development": {
            "rows": DEVELOPMENT_ROWS,
            "outer_shards": OUTER_SHARDS,
            "crossfit_correct": development_correct,
            "interpolation_correct": interpolation_correct,
            "gain_over_interpolation_count": development_correct
            - interpolation_correct,
        },
        "model": {
            "type": "deterministic_random_feature_ridge",
            "random_features": RANDOM_FEATURES,
            "ridge_penalty": RIDGE_PENALTY,
            "random_state": RANDOM_STATE,
            "retention_thresholds": RETENTION_THRESHOLDS,
            "development_labels_only": True,
        },
        "target": {
            "correct": total_correct,
            "total": args.expected_rows,
            "accuracy": total_correct / args.expected_rows,
            "interpolation_correct": baseline_total,
            "gain_over_interpolation_count": total_correct - baseline_total,
            "selection_counts": dict(sorted(Counter(choices.values()).items())),
            "domains": {
                task: {"correct": task_correct[task], "total": task_rows[task]}
                for task in TASKS
            },
            "paired_vs_interpolation": {
                "selector_only_correct": selector_only,
                "interpolation_only_correct": baseline_only,
                "mcnemar_exact_two_sided_p": _mcnemar_exact(
                    selector_only, baseline_only
                ),
            },
        },
        "development_candidate_sha256s": {
            arm: [sha256_file(path) for path in paths]
            for arm, paths in development_paths.items()
        },
        "development_score_sha256s": {
            arm: [
                sha256_file(path) for path in getattr(args, f"development_{arm}_scores")
            ]
            for arm in ARMS
        },
        "target_candidate_sha256s": {
            arm: [sha256_file(path) for path in paths]
            for arm, paths in target_paths.items()
        },
        "assessors_sha256": sha256_file(args.assessors),
        "selections_sha256": selections_sha256,
        "sandbox_receipt_sha256": sandbox_sha256,
        "sandbox_probe_sha256": sandbox.get("probe_sha256"),
        "mbpp_setup_qualification_count": len(setup_receipts),
    }
    _atomic_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for arm in ARMS:
        option = arm.replace("_", "-")
        parser.add_argument(
            f"--development-{option}-candidates",
            type=Path,
            action="append",
            required=True,
        )
        parser.add_argument(
            f"--development-{option}-scores",
            type=Path,
            action="append",
            required=True,
        )
        parser.add_argument(
            f"--target-{option}-candidates",
            type=Path,
            action="append",
            required=True,
        )
    parser.add_argument("--assessors", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(PARTITIONS), required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--sandbox-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report["target"], sort_keys=True))


if __name__ == "__main__":
    main()
