from __future__ import annotations

import select_q36_mtr_nested_pattern_consensus as module


def _record(task: str, shard: int, preferred: str = "B") -> dict:
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
        "shard": shard,
        "arms": {
            arm: {
                "answer": answer if task != "mbpp" else None,
                "correct": answer == preferred,
                "generated_tokens": 10,
                "max_token_exhausted": False,
            }
            for arm, answer in answers.items()
        },
    }


def test_nested_pattern_prefers_reliable_agreement_cluster() -> None:
    records = [
        _record(task, shard) for shard in range(3) for task in ("bbh_logic", "math500")
    ]
    model = module._fit(records)
    selected, reliability, mask = module._choose(
        _record("bbh_logic", 3), model, (30.0, (1.0, 1.0, 0.25))
    )
    assert selected == "interpolation"
    assert reliability < 0
    assert mask == sum(
        1 << module.ARM_ORDER.index(arm)
        for arm in ("interpolation", "direct", "level_two")
    )


def test_nested_pattern_selects_config_without_outer_labels() -> None:
    records_by_shard = {
        shard: [
            _record("bbh_logic", shard),
            _record("math500", shard),
        ]
        for shard in range(3)
    }
    first = module._select_outer_config(records_by_shard, 0)
    records_by_shard[0] = [
        _record("bbh_logic", 0, "A"),
        _record("math500", 0, "A"),
    ]
    second = module._select_outer_config(records_by_shard, 0)
    assert first == second
    assert first[0] in module.CONFIGS


def test_nested_pattern_preserves_interpolation_for_code() -> None:
    selected, reliability, mask = module._choose(
        _record("mbpp", 0), {}, (10.0, (4.0, 4.0, 0.25))
    )
    assert selected == "interpolation"
    assert reliability == 1.0
    assert mask == 1 << module.ARM_ORDER.index("interpolation")


def test_config_grid_is_frozen_and_deterministic() -> None:
    assert len(module.CONFIGS) == 108
    assert module.CONFIGS[0] == (10.0, (0.25, 0.25, 0.25))
    assert module.CONFIGS[-1] == (300.0, (4.0, 4.0, 4.0))
