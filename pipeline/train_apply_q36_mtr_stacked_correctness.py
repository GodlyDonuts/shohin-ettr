#!/usr/bin/env python3
"""Train a nonlinear stacked correctness selector for Q36 owner trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from hf_product_reasoning_eval import extract_short_answer, _normalize_short_answer
import train_apply_q36_mtr_calibration_correctness as correctness
import train_apply_q36_mtr_sparse_router as sparse

MODEL_SCHEMA = "shohin-q36-mtr-stacked-correctness-model-v1"
REPORT_SCHEMA = "shohin-q36-mtr-stacked-correctness-report-v1"
SELECTION_SCHEMA = "shohin-q36-mtr-stacked-correctness-selection-v1"
FOLDS = 5
STACKED_SEED = 2026080816
LOGISTIC_C = 3.0
META_LEAF_NODES = (4, 8, 16)
META_MIN_SAMPLES = (20, 50, 100)
META_ITERATIONS = 150
META_LEARNING_RATE = 0.05
META_L2 = 1.0


class Q36MTRStackedCorrectnessError(RuntimeError):
    """Stacked selector input, model, or output geometry differs."""


def _fold(identity: str) -> int:
    return int(identity[:16], 16) % FOLDS


def _document(row: dict[str, Any], owner: int) -> str:
    candidate = row["candidates"][owner]
    return (
        f"TASK={row['task']} OWNER={sparse.LINEAGES[owner]}\n"
        f"QUESTION\n{row['question']}\n"
        f"CANDIDATE\n{candidate['completion']}"
    )


def _vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=50_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=70_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def _answer(task: str, completion: str) -> str | None:
    if task == "mbpp":
        return None
    return _normalize_short_answer(extract_short_answer(completion))


def scalar_features(
    row: dict[str, Any], probabilities: list[float] | np.ndarray, owner: int
) -> list[float]:
    """Return compact nonlinear features without any correctness labels."""
    if len(probabilities) != len(sparse.LINEAGES) or not 0 <= owner < len(
        sparse.LINEAGES
    ):
        raise Q36MTRStackedCorrectnessError("stacked probability geometry differs")
    candidates = row["candidates"]
    completions = [str(candidate["completion"]) for candidate in candidates]
    answers = [_answer(row["task"], completion) for completion in completions]
    answer_counts = Counter(answer for answer in answers if answer is not None)
    answer = answers[owner]
    completion = completions[owner]
    lower = completion.lower()
    tokens = re.findall(r"[A-Za-z_]+|\d+(?:\.\d+)?|\S", completion)
    lengths = [len(value) for value in completions]
    ordered = sorted((float(value) for value in probabilities), reverse=True)
    task_features = [float(row["task"] == task) for task in sparse.TASKS]
    owner_features = [float(owner == index) for index in range(len(sparse.LINEAGES))]
    return [
        *(float(value) for value in probabilities),
        float(probabilities[owner]),
        max(float(value) for value in probabilities) - float(probabilities[owner]),
        ordered[0] - ordered[1],
        *task_features,
        *owner_features,
        math.log1p(len(completion)),
        math.log1p(len(tokens)),
        float(bool(candidates[owner].get("max_token_exhausted", False))),
        float(lower.count("boxed")),
        float(lower.count("answer")),
        float(lower.count("however")),
        float(lower.count("mistake")),
        float(lower.count("check")),
        float(lower.count("therefore")),
        sum(character.isdigit() for character in completion) / max(1, len(completion)),
        completion.count("\\") / max(1, len(completion)),
        float(answer is not None),
        math.log1p(len(answer or "")),
        float(answer is not None and answer_counts[answer] >= 2),
        float(answer is not None and answer_counts[answer] == 3),
        math.log1p(lengths[owner] / max(1, min(lengths))),
        math.log1p(max(lengths) / max(1, lengths[owner])),
    ]


def _logistic_model() -> LogisticRegression:
    return LogisticRegression(
        C=LOGISTIC_C,
        max_iter=500,
        solver="liblinear",
        random_state=STACKED_SEED,
    )


def _meta_model(
    leaf_nodes: int, minimum_samples: int
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=META_ITERATIONS,
        learning_rate=META_LEARNING_RATE,
        max_leaf_nodes=leaf_nodes,
        min_samples_leaf=minimum_samples,
        l2_regularization=META_L2,
        random_state=STACKED_SEED,
    )


def _atomic_joblib(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise Q36MTRStackedCorrectnessError("stacked model output exists")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        joblib.dump(payload, temporary_path, compress=3)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return sparse.sha256_file(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(STACKED_SEED)
    np.random.seed(STACKED_SEED)
    rows = sparse.load_training_rows(args.training_rows)
    development = sparse.load_development_rows(args.development_rows)
    development_rows = [development[identity] for identity in sorted(development)]
    documents = [
        _document(row, owner) for row in rows for owner in range(len(sparse.LINEAGES))
    ]
    development_documents = [
        _document(row, owner)
        for row in development_rows
        for owner in range(len(sparse.LINEAGES))
    ]
    vectorizer = _vectorizer()
    matrix = vectorizer.fit_transform(documents)
    development_matrix = vectorizer.transform(development_documents)
    labels = np.asarray(
        [
            [int(bool(candidate["correct"])) for candidate in row["candidates"]]
            for row in rows
        ],
        dtype=np.int8,
    )
    folds = np.asarray([_fold(row["identity_sha256"]) for row in rows])
    oof_probabilities = np.zeros((len(rows), len(sparse.LINEAGES)), dtype=float)
    development_probabilities = np.zeros(
        (len(development_rows), len(sparse.LINEAGES)), dtype=float
    )
    logistic_models: list[list[LogisticRegression]] = []
    for owner in range(len(sparse.LINEAGES)):
        owner_indices = np.asarray(
            [len(sparse.LINEAGES) * index + owner for index in range(len(rows))]
        )
        development_indices = np.asarray(
            [
                len(sparse.LINEAGES) * index + owner
                for index in range(len(development_rows))
            ]
        )
        owner_models = []
        fold_development = []
        for fold in range(FOLDS):
            fit = np.where(folds != fold)[0]
            held_out = np.where(folds == fold)[0]
            model = _logistic_model()
            model.fit(matrix[owner_indices[fit]], labels[fit, owner])
            oof_probabilities[held_out, owner] = model.predict_proba(
                matrix[owner_indices[held_out]]
            )[:, 1]
            fold_development.append(
                model.predict_proba(development_matrix[development_indices])[:, 1]
            )
            owner_models.append(model)
        development_probabilities[:, owner] = np.mean(fold_development, axis=0)
        logistic_models.append(owner_models)
    meta_matrix = np.asarray(
        [
            scalar_features(row, oof_probabilities[index], owner)
            for index, row in enumerate(rows)
            for owner in range(len(sparse.LINEAGES))
        ]
    )
    development_meta = np.asarray(
        [
            scalar_features(row, development_probabilities[index], owner)
            for index, row in enumerate(development_rows)
            for owner in range(len(sparse.LINEAGES))
        ]
    )
    flat_labels = labels.reshape(-1)
    meta_training_rows = np.where(folds != 0)[0]
    meta_validation_rows = np.where(folds == 0)[0]
    meta_training_indices = np.concatenate(
        [len(sparse.LINEAGES) * meta_training_rows + owner for owner in range(3)]
    )
    search = []
    best: tuple[tuple[int, int, int], int, int] | None = None
    for leaf_nodes in META_LEAF_NODES:
        for minimum_samples in META_MIN_SAMPLES:
            model = _meta_model(leaf_nodes, minimum_samples)
            model.fit(
                meta_matrix[meta_training_indices], flat_labels[meta_training_indices]
            )
            probabilities = np.column_stack(
                [
                    model.predict_proba(
                        meta_matrix[len(sparse.LINEAGES) * meta_validation_rows + owner]
                    )[:, 1]
                    for owner in range(len(sparse.LINEAGES))
                ]
            )
            selected = probabilities.argmax(axis=1)
            correct = int(labels[meta_validation_rows, selected].sum())
            trial = {
                "leaf_nodes": leaf_nodes,
                "minimum_samples": minimum_samples,
                "correct": correct,
                "rows": len(meta_validation_rows),
            }
            search.append(trial)
            key = (correct, -leaf_nodes, -minimum_samples)
            if best is None or key > best[0]:
                best = (key, leaf_nodes, minimum_samples)
    if best is None:
        raise Q36MTRStackedCorrectnessError("stacked model search differs")
    _, selected_leaf_nodes, selected_minimum_samples = best
    meta_model = _meta_model(selected_leaf_nodes, selected_minimum_samples)
    meta_model.fit(meta_matrix, flat_labels)
    selected_probabilities = np.column_stack(
        [
            meta_model.predict_proba(
                development_meta[
                    len(sparse.LINEAGES) * np.arange(len(development_rows)) + owner
                ]
            )[:, 1]
            for owner in range(len(sparse.LINEAGES))
        ]
    )
    selected_indices = selected_probabilities.argmax(axis=1)
    model_sha256 = _atomic_joblib(
        args.model_output,
        {
            "schema": MODEL_SCHEMA,
            "vectorizer": vectorizer,
            "logistic_models": logistic_models,
            "meta_model": meta_model,
            "selected_leaf_nodes": selected_leaf_nodes,
            "selected_minimum_samples": selected_minimum_samples,
            "sklearn_version": sklearn.__version__,
        },
    )
    candidates = []
    decisions = []
    counts: Counter[str] = Counter()
    development_owners = correctness._embedded_development_owners(development)
    for index, (row, selected_index) in enumerate(
        zip(development_rows, selected_indices, strict=True)
    ):
        candidate = dict(
            development_owners[int(selected_index)][row["identity_sha256"]]
        )
        metadata = {
            "schema": SELECTION_SCHEMA,
            "selected_lineage": sparse.LINEAGES[int(selected_index)],
            "probabilities": selected_probabilities[index].tolist(),
            "text_probabilities": development_probabilities[index].tolist(),
            "model_sha256": model_sha256,
            "development_labels_read": 0,
        }
        candidate["stacked_correctness_selection"] = metadata
        candidates.append(candidate)
        decisions.append(
            {
                "identity_sha256": row["identity_sha256"],
                "task": row["task"],
                **metadata,
            }
        )
        counts[sparse.LINEAGES[int(selected_index)]] += 1
    output_sha256 = sparse._atomic_lines(args.output, candidates)
    decisions_sha256 = sparse._atomic_lines(args.decisions, decisions)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "interpretation": "calibration_only_oof_text_nonlinear_stacked_correctness",
        "rows": len(candidates),
        "development_labels_read": 0,
        "training_rows_sha256": sparse.sha256_file(args.training_rows),
        "development_rows_sha256": sparse.sha256_file(args.development_rows),
        "model_sha256": model_sha256,
        "output_sha256": output_sha256,
        "decisions_sha256": decisions_sha256,
        "selection_counts": dict(sorted(counts.items())),
        "logistic_c": LOGISTIC_C,
        "folds": FOLDS,
        "search": search,
        "selected_leaf_nodes": selected_leaf_nodes,
        "selected_minimum_samples": selected_minimum_samples,
        "sklearn_version": sklearn.__version__,
    }
    sparse._atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-rows", type=Path, required=True)
    parser.add_argument("--development-rows", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(run(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
