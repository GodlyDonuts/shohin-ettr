from __future__ import annotations

import math

import train_apply_q36_mtr_stacked_correctness as module


def _row() -> dict:
    return {
        "task": "math500",
        "candidates": [
            {
                "completion": "The answer is \\boxed{7}.",
                "max_token_exhausted": False,
            },
            {
                "completion": "Check carefully. Therefore \\boxed{7}.",
                "max_token_exhausted": False,
            },
            {
                "completion": "However, the answer is \\boxed{8}.",
                "max_token_exhausted": True,
            },
        ],
    }


def test_scalar_features_are_finite_and_label_free() -> None:
    row = _row()
    features = module.scalar_features(row, [0.2, 0.7, 0.1], 1)
    assert len(features) == 29
    assert all(math.isfinite(value) for value in features)
    assert features[-4] == 1.0  # the selected answer has owner support


def test_scalar_features_detect_exhaustion_and_disagreement() -> None:
    features = module.scalar_features(_row(), [0.2, 0.7, 0.1], 2)
    assert features[14] == 1.0
    assert features[-4] == 0.0


def test_fold_is_deterministic() -> None:
    identity = "f" * 64
    assert module._fold(identity) == int(identity[:16], 16) % module.FOLDS


def test_document_binds_task_owner_question_and_candidate() -> None:
    row = _row()
    row["question"] = "What is seven?"
    document = module._document(row, 1)
    assert "TASK=math500 OWNER=owner_71" in document
    assert "What is seven?" in document
    assert "Check carefully" in document
