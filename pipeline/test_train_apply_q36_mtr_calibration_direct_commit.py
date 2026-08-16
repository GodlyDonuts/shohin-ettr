from __future__ import annotations

import train_apply_q36_mtr_calibration_direct_commit as module


def _row(index: int, task: str = "math500") -> dict:
    direct = index % 3
    production = (index + 1) % 3
    correctness = [owner == direct for owner in range(3)]
    return {
        "identity_sha256": f"{index:064x}",
        "fold": index % 2,
        "task": task,
        "selected_lineage": module.sparse.LINEAGES[direct],
        "production_commit_lineage": module.sparse.LINEAGES[production],
        "scores": [1.0 if owner == direct else 0.1 * owner for owner in range(3)],
        "candidate_correctness": correctness,
    }


def test_direct_meta_fit_is_finite_and_selects_owner() -> None:
    rows = [_row(index) for index in range(120)]
    vocabulary = module._vocabulary(rows)
    weights, report = module._fit(rows, vocabulary)
    probabilities = module._probabilities(rows[0], vocabulary, weights)
    assert weights.shape == (len(vocabulary) + 11, 3)
    assert len(probabilities) == 3
    assert abs(sum(probabilities) - 1.0) < 1e-12
    assert report["rows"] == 120


def test_cross_fit_excludes_held_fold(monkeypatch) -> None:
    monkeypatch.setattr(module.stack, "FOLDS", 2)
    monkeypatch.setattr(module, "STEPS", 20)
    rows = [_row(index) for index in range(240)]
    vocabulary = module._vocabulary(rows)
    predictions, folds = module._cross_fit(rows, vocabulary)
    assert len(predictions) == 240
    assert len(folds) == 2
    assert all(fold["training_rows"] == 120 for fold in folds)
    assert all(fold["held_out_rows"] == 120 for fold in folds)


def test_threshold_selection_prefers_retention_on_tie() -> None:
    rows = [
        {
            "direct_index": 1,
            "production_index": 0,
            "confidence_margin": 0.12,
            "candidate_correctness": [True, False, False],
        },
        {
            "direct_index": 1,
            "production_index": 0,
            "confidence_margin": 0.08,
            "candidate_correctness": [False, True, False],
        },
    ]
    threshold, trials = module._choose_threshold(rows)
    assert threshold == 1.1
    assert len(trials) == len(module.THRESHOLDS)
    assert module._threshold_metrics(rows, threshold)["regressions"] == 0
