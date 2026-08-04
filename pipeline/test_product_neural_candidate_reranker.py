"""Focused tests for frozen-feature correctness-head mechanics."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from product_candidate_reranker import CandidateReranker, FEATURE_NAMES, SCHEMA as SHAPE_SCHEMA
from product_neural_candidate_reranker import (
    FEATURE_SCHEMA,
    FrozenFeatureCorrectnessHead,
    fit,
    group_indices,
    identity_is_validation,
    select,
    selection_metrics,
)


def _metadata(identity: str, sample: int, correct: bool) -> dict:
    return {
        "identity_sha256": identity,
        "task": "math500",
        "sample_index": sample,
        "prediction": str(sample),
        "correct": correct,
        "empty_completion": False,
    }


def test_grouping_restores_sample_order() -> None:
    rows = [_metadata("x", 1, False), _metadata("x", 0, True)]
    grouped = group_indices(rows)
    assert grouped["x"] == [1, 0]


def test_validation_split_is_deterministic() -> None:
    assert identity_is_validation("fixed", 31) == identity_is_validation("fixed", 31)


def test_empty_candidate_never_wins_selection() -> None:
    rows = [_metadata("x", 0, True), _metadata("x", 1, False)]
    rows[1]["empty_completion"] = True
    metrics = selection_metrics(
        group_indices(rows), rows, torch.tensor([0.0, 100.0]), validation=None, seed=31
    )
    assert metrics["selected_correct"] == 1


def test_correctness_head_emits_one_score_per_row() -> None:
    model = FrozenFeatureCorrectnessHead(12, 4, 16)
    scores = model(torch.randn(3, 12), torch.randn(3, 4))
    assert scores.shape == (3,)


def test_fit_save_and_select_round_trip(tmp_path: Path) -> None:
    identities: list[str] = []
    validation = train = 0
    candidate = 0
    while validation < 2 or train < 12:
        identity = f"row-{candidate}"
        candidate += 1
        is_validation = identity_is_validation(identity, 31)
        if is_validation and validation < 2:
            validation += 1
            identities.append(identity)
        elif not is_validation and train < 12:
            train += 1
            identities.append(identity)
    metadata: list[dict] = []
    hidden: list[torch.Tensor] = []
    shape: list[torch.Tensor] = []
    for identity in identities:
        for sample in range(2):
            correct = sample == 1
            metadata.append(_metadata(identity, sample, correct))
            feature = torch.zeros(12)
            feature[0] = 1.0 if correct else -1.0
            hidden.append(feature)
            shape.append(torch.zeros(len(FEATURE_NAMES)))
    feature_path = tmp_path / "features.pt"
    torch.save(
        {
            "schema": FEATURE_SCHEMA,
            "model_revision": "model",
            "adapter_sha256": "adapter",
            "candidate_sha256": "candidates",
            "layer_offsets": (-1,),
            "pooling": ("last",),
            "tail_tokens": 32,
            "shape_feature_names": FEATURE_NAMES,
            "hidden_features": torch.stack(hidden).half(),
            "shape_features": torch.stack(shape),
            "metadata": metadata,
        },
        feature_path,
    )
    shape_model = CandidateReranker(len(FEATURE_NAMES), 16)
    shape_path = tmp_path / "shape.pt"
    torch.save(
        {
            "schema": SHAPE_SCHEMA,
            "feature_names": FEATURE_NAMES,
            "hidden_size": 16,
            "state_dict": shape_model.state_dict(),
            "feature_mean": torch.zeros(len(FEATURE_NAMES)),
            "feature_scale": torch.ones(len(FEATURE_NAMES)),
        },
        shape_path,
    )
    model_path = tmp_path / "model.pt"
    report = fit(
        argparse.Namespace(
            features=[feature_path],
            shape_model=shape_path,
            output=model_path,
            hidden_size=16,
            epochs=4,
            patience=4,
            batch_size=16,
            eval_batch_size=32,
            learning_rate=1e-2,
            weight_decay=0.0,
            bce_weight=0.25,
            seed=31,
            device="cpu",
        )
    )
    assert report["validation_metrics"]["selected_correct"] == 2
    selection_path = tmp_path / "selection.json"
    selected = select(
        argparse.Namespace(
            features=[feature_path],
            model=model_path,
            output=selection_path,
            eval_batch_size=32,
            device="cpu",
        )
    )
    assert selected["selected_correct"] == len(identities)
