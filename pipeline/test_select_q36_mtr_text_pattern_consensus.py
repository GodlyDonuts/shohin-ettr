from __future__ import annotations

import select_q36_mtr_text_pattern_consensus as module


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
        "tokens": ("algebra", "minimum"),
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


def test_tokens_are_bounded_deduplicated_and_drop_instructions() -> None:
    result = module._tokens(
        "Solve the following problem. Algebra algebra minimum geometry."
    )
    assert result == ("algebra", "minimum", "geometry")


def test_text_model_prefers_reliable_agreement_cluster() -> None:
    records = [
        _record(task, shard) for shard in range(3) for task in ("bbh_logic", "math500")
    ]
    model = module._fit_text(records)
    selected, score, text_score, mask = module._choose(
        _record("math500", 3),
        model,
        (30.0, (1.0, 4.0, 0.25)),
        (50.0, 0.025, 3),
    )
    assert selected == "interpolation"
    assert score < 0
    assert text_score > 0
    assert mask == sum(
        1 << module.ARM_ORDER.index(arm)
        for arm in ("interpolation", "direct", "level_two")
    )


def test_token_config_selection_does_not_read_outer_labels() -> None:
    records_by_shard = {
        shard: [
            _record("bbh_logic", shard),
            _record("math500", shard),
        ]
        for shard in range(3)
    }
    base_config = (30.0, (1.0, 4.0, 0.25))
    first = module._select_token_config(records_by_shard, 0, base_config)
    records_by_shard[0] = [
        _record("bbh_logic", 0, "A"),
        _record("math500", 0, "A"),
    ]
    second = module._select_token_config(records_by_shard, 0, base_config)
    assert first == second
    assert first[0] in module.TOKEN_CONFIGS


def test_text_model_preserves_interpolation_for_code() -> None:
    selected, score, text_score, mask = module._choose(
        _record("mbpp", 0),
        {},
        (30.0, (1.0, 4.0, 0.25)),
        (50.0, 0.025, 3),
    )
    assert selected == "interpolation"
    assert score == 1.0
    assert text_score == 0.0
    assert mask == 1 << module.ARM_ORDER.index("interpolation")


def test_token_grid_is_frozen() -> None:
    assert len(module.TOKEN_CONFIGS) == 27
    assert module.TOKEN_CONFIGS[0] == (30.0, 0.01, 1)
    assert module.TOKEN_CONFIGS[-1] == (100.0, 0.05, 5)
