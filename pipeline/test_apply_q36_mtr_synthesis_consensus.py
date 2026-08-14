from __future__ import annotations

import apply_q36_mtr_synthesis_consensus as module


def _candidate(answer: str, task: str = "math500") -> dict:
    return {"completion": rf"Reasoning. \boxed{{{answer}}}", "task": task}


def test_synthesis_is_selected_when_owner_confirms_answer(monkeypatch) -> None:
    monkeypatch.setattr(module.stack, "_production_index", lambda candidates, task: 0)
    selected, metadata = module.choose(
        [_candidate("1"), _candidate("2"), _candidate("3"), _candidate("2")],
        "math500",
    )
    assert selected == 3
    assert metadata["reason"] == "synthesis_confirmed_by_owner"


def test_owner_consensus_overrides_unconfirmed_synthesis(monkeypatch) -> None:
    monkeypatch.setattr(module.stack, "_production_index", lambda candidates, task: 2)
    selected, metadata = module.choose(
        [_candidate("4"), _candidate("4"), _candidate("3"), _candidate("9")],
        "bbh_logic",
    )
    assert selected == 0
    assert metadata["reason"] == "owner_consensus"


def test_code_retains_production(monkeypatch) -> None:
    monkeypatch.setattr(module.stack, "_production_index", lambda candidates, task: 1)
    selected, metadata = module.choose(
        [_candidate("1", "mbpp") for _ in range(4)], "mbpp"
    )
    assert selected == 1
    assert metadata["reason"] == "production_fallback"
