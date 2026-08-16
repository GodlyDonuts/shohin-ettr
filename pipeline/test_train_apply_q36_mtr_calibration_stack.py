from __future__ import annotations

import hashlib

import pytest

import train_apply_q36_mtr_calibration_stack as module


def _identity(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _candidate(lineage: str, correct: bool) -> dict:
    completion = "The answer is 42." if correct else "unfinished attempt"
    return {
        "lineage": lineage,
        "completion": completion,
        "correct": correct,
        "generated_tokens": 6,
        "max_token_exhausted": not correct,
    }


def test_production_index_applies_candidate_rule_sequentially() -> None:
    candidates = [
        _candidate("current", False),
        _candidate("owner_71", True),
        _candidate("owner_8", False),
    ]
    assert module._production_index(candidates, "math500") == 1


def test_oof_sparse_excludes_every_held_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "FOLDS", 2)
    monkeypatch.setattr(module, "SPARSE_EPOCHS", 1)
    rows = []
    for index in range(12):
        pattern = f"{index % 8:03b}"
        rows.append(
            {
                "identity_sha256": _identity(f"row-{index}"),
                "task": "math500",
                "question": f"Compute value {index}.",
                "correctness_pattern": pattern,
                "candidates": [
                    _candidate(lineage, bit == "1")
                    for lineage, bit in zip(
                        module.sparse.LINEAGES, pattern, strict=True
                    )
                ],
            }
        )
    outcomes, report = module._oof_sparse(rows)
    assert len(outcomes) == len(rows)
    assert report["training_excludes_selected_identity_fold"] is True
    assert len(report["folds"]) == 2
    assert all(
        item["training_rows"] + item["held_out_rows"] == 12 for item in report["folds"]
    )


def test_meta_model_is_finite_and_retains_feature_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "META_STEPS", 10)
    outcomes = []
    for index in range(12):
        outcomes.append(
            {
                "task": "math500" if index % 2 else "bbh_logic",
                "selected_lineage": module.sparse.LINEAGES[index % 3],
                "production_commit_lineage": module.sparse.LINEAGES[(index + 1) % 3],
                "scores": [0.1 * index, 0.2, -0.1],
                "correct": index % 3 == 0,
                "production_commit_correct": index % 3 == 1,
            }
        )
    weights, vocabulary, report = module._fit_meta(outcomes)
    assert len(weights) == len(vocabulary) + 5
    assert report["discordant_training_rows"] > 0
    assert report["training_class_sparse"] > 0
    assert report["training_class_production"] > 0
