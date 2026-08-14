from __future__ import annotations

from pathlib import Path

import train_apply_q36_mtr_calibration_correctness as module


def _features(index: int) -> dict[int, float]:
    return {0: 1.0, 1 + index: 1.0}


def _row(index: int, owner: int) -> dict:
    return {
        "task": "math500",
        "candidates": [{"correct": candidate == owner} for candidate in range(3)],
        "_features": [_features(candidate) for candidate in range(3)],
    }


def test_absolute_correctness_heads_learn_owner_probabilities(monkeypatch) -> None:
    monkeypatch.setattr(module.sparse, "DIMENSION", 8)
    rows = [_row(index, index % 3) for index in range(180)]
    weights, report = module._fit(rows, learning_rate=0.1, epochs=6, balanced=False)
    probabilities = module._probabilities(weights, rows[0])
    assert len(weights) == 3
    assert len(probabilities) == 3
    assert all(0.0 < value < 1.0 for value in probabilities)
    assert report["updates"] == 180 * 3 * 6


def test_threshold_selection_prefers_conservative_retention_on_tie() -> None:
    rows = [
        {
            "head_index": 1,
            "production_index": 0,
            "estimated_gain": 0.05,
            "correctness": [True, False, False],
        },
        {
            "head_index": 1,
            "production_index": 0,
            "estimated_gain": 0.03,
            "correctness": [False, True, False],
        },
    ]
    threshold, trials = module._choose_threshold(rows)
    assert threshold == 1.1
    assert len(trials) == len(module.THRESHOLDS)
    assert module._threshold_metrics(rows, threshold)["regressions"] == 0


def test_sigmoid_is_finite_at_extremes() -> None:
    assert 0.0 < module._sigmoid(-1e9) < 0.5
    assert 0.5 < module._sigmoid(1e9) < 1.0


def test_lineage_thresholds_fall_back_when_support_is_small() -> None:
    rows = [
        {
            "task": "math500",
            "head_index": index % 2,
            "production_index": 2,
            "estimated_gain": 0.05,
            "correctness": [True, False, False],
        }
        for index in range(10)
    ]
    task_thresholds, _ = module._threshold_map(rows, "task")
    grouped, _ = module._threshold_map(rows, "task_head")
    thresholds = {**task_thresholds, **grouped}
    assert grouped == {}
    assert (
        module._threshold_for(rows[0], "task_head", thresholds)
        == task_thresholds["math500"]
    )


def test_supported_lineage_thresholds_override_task_threshold() -> None:
    rows = [
        {
            "task": "bbh_logic",
            "head_index": 1,
            "production_index": 0,
            "estimated_gain": 0.05,
            "correctness": [index % 2 == 0, index % 2 == 1, False],
        }
        for index in range(module.MINIMUM_THRESHOLD_GROUP)
    ]
    task_thresholds, _ = module._threshold_map(rows, "task")
    grouped, _ = module._threshold_map(rows, "task_head")
    thresholds = {**task_thresholds, **grouped}
    key = "bbh_logic:1"
    assert key in grouped
    assert module._threshold_for(rows[0], "task_head", thresholds) == thresholds[key]


def test_job_emits_matched_production_baseline() -> None:
    job = (
        Path(__file__).resolve().parent
        / "jobs"
        / "q36_mtr_calibration_correctness.sbatch"
    ).read_text(encoding="utf-8")
    assert "PRODUCTION_OUTPUT" in job
    assert '--production-output "$PRODUCTION_OUTPUT"' in job
    assert 'chmod a-w "$MODEL_OUTPUT" "$OUTPUT" "$PRODUCTION_OUTPUT"' in job
