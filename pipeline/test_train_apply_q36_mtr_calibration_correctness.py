from __future__ import annotations

import json
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


def test_production_confidence_bins_are_fixed_and_distinct() -> None:
    base = {
        "task": "math500",
        "head_index": 1,
        "production_index": 2,
    }
    groups = [
        module._threshold_group(
            {**base, "production_probability": probability},
            "task_head_production_confidence",
        )
        for probability in (0.1, 0.25, 0.5, 0.75, 0.99)
    ]
    assert groups == [
        "math500:1:2:production_confidence_0",
        "math500:1:2:production_confidence_1",
        "math500:1:2:production_confidence_2",
        "math500:1:2:production_confidence_3",
        "math500:1:2:production_confidence_3",
    ]


def test_high_confidence_production_can_learn_stricter_retention() -> None:
    rows = []
    for index in range(module.MINIMUM_THRESHOLD_GROUP):
        rows.append(
            {
                "task": "bbh_logic",
                "head_index": 1,
                "production_index": 0,
                "production_probability": 0.9,
                "estimated_gain": 0.05,
                "correctness": [True, index == 0, False],
            }
        )
    for index in range(module.MINIMUM_THRESHOLD_GROUP):
        rows.append(
            {
                "task": "bbh_logic",
                "head_index": 1,
                "production_index": 0,
                "production_probability": 0.1,
                "estimated_gain": 0.05,
                "correctness": [False, index != 0, False],
            }
        )
    grouped, _ = module._threshold_map(rows, "task_head_production_confidence")
    low = "bbh_logic:1:0:production_confidence_0"
    high = "bbh_logic:1:0:production_confidence_3"
    assert grouped[low] == 0.04
    assert grouped[high] == 1.1


def test_embedded_development_projection_is_deterministic() -> None:
    identity = "a" * 64
    rows = {
        identity: {
            "task": "math500",
            "candidates": [
                {"lineage": lineage, "completion": f"answer {index}"}
                for index, lineage in enumerate(module.sparse.LINEAGES)
            ],
        }
    }
    owners = module._embedded_development_owners(rows)
    assert [owner[identity]["lineage"] for owner in owners] == list(
        module.sparse.LINEAGES
    )
    assert all(owner[identity]["generated_tokens"] == 2 for owner in owners)
    assert all(
        owner[identity]["schema"] == module.sparse.CANDIDATE_SCHEMA for owner in owners
    )


def test_reused_decisions_preserve_exact_probabilities(
    tmp_path: Path, monkeypatch
) -> None:
    identity = "b" * 64
    path = tmp_path / "decisions.jsonl"
    row = {
        "schema": module.SELECTION_SCHEMA,
        "identity_sha256": identity,
        "task": "math500",
        "head_lineage": "owner_8",
        "production_commit_lineage": "current",
        "probabilities": [0.7, 0.2, 0.9],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(module.sparse, "DEVELOPMENT_ROWS", 1)
    loaded = module._load_reused_development_decisions(path)
    assert loaded[identity]["probabilities"] == [0.7, 0.2, 0.9]
    assert loaded[identity]["production_commit_lineage"] == "current"


def test_job_emits_matched_production_baseline() -> None:
    job = (
        Path(__file__).resolve().parent
        / "jobs"
        / "q36_mtr_calibration_correctness.sbatch"
    ).read_text(encoding="utf-8")
    assert "PRODUCTION_OUTPUT" in job
    assert '--production-output "$PRODUCTION_OUTPUT"' in job
    assert 'chmod a-w "$MODEL_OUTPUT" "$OUTPUT" "$PRODUCTION_OUTPUT"' in job
