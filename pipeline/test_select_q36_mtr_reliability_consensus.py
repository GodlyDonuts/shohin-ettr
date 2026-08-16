from __future__ import annotations

import numpy as np

import select_q36_mtr_reliability_consensus as module


def _record(correct: dict[str, bool] | None = None) -> dict:
    answers = {
        "hierarchy": "A",
        "interpolation": "B",
        "direct": "B",
        "offset_one": "C",
        "level_two": "B",
        "challenger": "A",
    }
    correct = correct or {arm: answers[arm] == "B" for arm in module.ARM_ORDER}
    return {
        "task": "bbh_logic",
        "shard": 0,
        "arms": {
            arm: {
                "answer": answers[arm],
                "correct": correct[arm],
                "generated_tokens": 10 + index,
                "max_token_exhausted": False,
            }
            for index, arm in enumerate(module.ARM_ORDER)
        },
    }


def test_clusters_group_equal_answers_and_bind_features() -> None:
    clusters = module._clusters(_record())
    assert [cluster["arms"] for cluster in clusters] == [
        ["hierarchy", "challenger"],
        ["interpolation", "direct", "level_two"],
        ["offset_one"],
    ]
    assert all(
        cluster["features"].shape == (len(module.FEATURE_NAMES),)
        for cluster in clusters
    )
    assert [cluster["correct"] for cluster in clusters] == [False, True, False]


def test_clusters_reject_inconsistent_correctness_for_same_answer() -> None:
    record = _record()
    record["arms"]["direct"]["correct"] = False
    try:
        module._clusters(record)
    except module.Q36MTRReliabilityConsensusError as error:
        assert "correctness" in str(error)
    else:
        raise AssertionError("inconsistent normalized answer was accepted")


def test_fit_and_choose_are_deterministic() -> None:
    records = [_record() for _ in range(8)]
    first = module._fit(records)
    second = module._fit(records)
    assert np.array_equal(first["weights"], second["weights"])
    selected, probability = module._choose(_record(), first)
    assert selected == "interpolation"
    assert 0.5 < probability <= 1.0


def test_mbpp_uses_interpolation_without_model_features() -> None:
    record = _record()
    record["task"] = "mbpp"
    assert module._choose(record, {}) == ("interpolation", 1.0)
