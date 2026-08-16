from __future__ import annotations

from array import array

import train_apply_q36_mtr_pairwise_guard as module


def test_pair_features_change_sign_with_candidate_order(monkeypatch) -> None:
    monkeypatch.setattr(module.sparse, "DIMENSION", 32)
    row = {
        "task": "math500",
        "_features": [{1: 2.0}, {1: -1.0}, {2: 1.0}],
    }
    forward = module._pair_features(row, 1, 0)
    reverse = module._pair_features(row, 0, 1)
    assert forward[1] == -3.0
    assert reverse[1] == 3.0


def test_pair_probability_is_finite(monkeypatch) -> None:
    monkeypatch.setattr(module.sparse, "DIMENSION", 32)
    row = {
        "task": "bbh_logic",
        "_features": [{1: 1.0}, {2: 1.0}, {3: 1.0}],
    }
    weights = array("d", [0.0]) * module.sparse.DIMENSION
    probability = module._probability(weights, row, 1, 0)
    assert probability == 0.5


def test_threshold_prefers_retention_on_tie() -> None:
    rows = [
        {
            "head_index": 1,
            "production_index": 0,
            "pair_probability": 0.6,
            "correctness": [True, False, False],
        },
        {
            "head_index": 1,
            "production_index": 0,
            "pair_probability": 0.4,
            "correctness": [False, True, False],
        },
    ]
    threshold, _ = module._choose_threshold(rows)
    assert threshold == 1.1


def test_pair_group_falls_back_for_small_support() -> None:
    rows = [
        {
            "task": "math500",
            "head_index": 1,
            "production_index": 0,
            "pair_probability": 0.7,
            "correctness": [False, True, False],
        }
        for _ in range(5)
    ]
    task, _ = module._threshold_map(rows, "task")
    pair, _ = module._threshold_map(rows, "task_pair")
    assert pair == {}
    assert (
        module._threshold_for(rows[0], "task_pair", {**task, **pair}) == task["math500"]
    )
