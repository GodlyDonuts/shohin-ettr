from __future__ import annotations

import select_q36_mtr_pattern_consensus as module


def _record(task: str = "bbh_logic") -> dict:
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
                "generated_tokens": 10,
                "max_token_exhausted": False,
            }
            for arm, answer in answers.items()
        },
    }


def test_pattern_model_prefers_reliable_agreement_cluster() -> None:
    model = module._fit(
        [_record(task) for _ in range(20) for task in ("bbh_logic", "math500")]
    )
    selected, reliability, mask = module._choose(_record(), model)
    assert selected == "interpolation"
    assert reliability > 0.5
    expected = sum(
        1 << module.ARM_ORDER.index(arm)
        for arm in ("interpolation", "direct", "level_two")
    )
    assert mask == expected


def test_pattern_model_is_deterministic() -> None:
    records = [_record(task) for _ in range(3) for task in ("bbh_logic", "math500")]
    first = module._fit(records)
    second = module._fit(records)
    assert first == second
    assert module._choose(_record(), first) == module._choose(_record(), second)


def test_pattern_model_preserves_interpolation_for_code() -> None:
    selected, reliability, mask = module._choose(_record("mbpp"), {})
    assert selected == "interpolation"
    assert reliability == 1.0
    assert mask == 1 << module.ARM_ORDER.index("interpolation")


def test_smoothed_unseen_pattern_backs_off_to_prior() -> None:
    assert module._smoothed(None, 0.4) == 0.4
