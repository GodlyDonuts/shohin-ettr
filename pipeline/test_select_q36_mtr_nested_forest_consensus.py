from __future__ import annotations

import numpy as np

import select_q36_mtr_nested_forest_consensus as module


def _record(task: str = "math500") -> dict:
    answers = {
        "hierarchy": "A",
        "interpolation": "B",
        "direct": "B",
        "offset_one": "C",
        "level_two": "B",
        "challenger": "A",
    }
    return {
        "task": task,
        "shard": 0,
        "arms": {
            arm: {
                "answer": answer if task != "mbpp" else None,
                "correct": answer == "B",
                "generated_tokens": 10 + index,
                "max_token_exhausted": False,
                "row": {
                    "completion": f"answer {answer}",
                    "wall_seconds": 1.0 + index,
                },
            }
            for index, (arm, answer) in enumerate(answers.items())
        },
    }


def test_forest_feature_geometry_is_frozen() -> None:
    record = _record()
    cluster = next(
        cluster
        for cluster in module._clusters(record)
        if cluster["arms"] == ["interpolation", "direct", "level_two"]
    )
    features = module._feature_vector(record, cluster)
    assert len(module.FEATURE_NAMES) == 50
    assert features.shape == (50,)
    assert np.all(np.isfinite(features))
    assert features[module.FEATURE_NAMES.index("vote_fraction")] == 0.5


def test_threshold_selection_is_conservative_on_ties() -> None:
    threshold, correct = module._choose_threshold(
        [
            (1, 1, 0.10, True),
            (0, 0, 0.20, True),
            (1, 1, -1.0, False),
        ]
    )
    assert threshold == 0.3
    assert correct == 2


def test_prediction_uses_probability_then_cluster_size() -> None:
    record = _record()
    samples, identity_samples = module._cluster_samples({"id": record})
    probabilities = np.asarray([0.8, 0.8, 0.2])
    best, pattern, margin = module._prediction_for_identity(
        record,
        identity_samples["id"],
        samples,
        probabilities,
        "hierarchy",
    )
    assert best is not None and pattern is not None
    assert samples[best]["selected"] == "interpolation"
    assert samples[pattern]["selected"] == "hierarchy"
    assert margin == 0.0


def test_forest_and_threshold_constants_are_frozen() -> None:
    assert module.N_ESTIMATORS == 200
    assert module.MIN_SAMPLES_LEAF == 5
    assert module.MAX_FEATURES == "sqrt"
    assert module.RANDOM_STATE == 2026081429
    assert module.THRESHOLDS[0] == 0.0
    assert module.THRESHOLDS[-1] == 0.3
    assert len(module.THRESHOLDS) == 61
