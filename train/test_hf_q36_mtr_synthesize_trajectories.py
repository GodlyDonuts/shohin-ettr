from __future__ import annotations

import hashlib

import pytest

import hf_q36_mtr_synthesize_trajectories as module


def _candidate(value: str) -> dict:
    return {"completion": f"reasoning attempt {value}"}


def test_synthesis_prompt_contains_all_attempts_without_lineage_names() -> None:
    identity = hashlib.sha256(b"identity").hexdigest()
    prompt, order = module.synthesis_prompt(
        "Compute the answer.",
        identity,
        [_candidate("A"), _candidate("B"), _candidate("C")],
    )
    assert len(order) == 3
    assert set(order) == set(module.sparse.LINEAGES)
    assert all(f"reasoning attempt {value}" in prompt for value in "ABC")
    assert "owner_71" not in prompt
    assert "owner_8" not in prompt
    assert prompt.count("Original problem:\nCompute the answer.") == 2


def test_synthesis_rotation_is_deterministic_and_balanced() -> None:
    rotations = [module._rotation(f"{index:064x}") for index in range(600)]
    assert rotations == [module._rotation(f"{index:064x}") for index in range(600)]
    assert set(rotations) == {0, 1, 2}
    assert max(rotations.count(value) for value in range(3)) < 240


def test_synthesis_rejects_missing_candidate() -> None:
    with pytest.raises(module.Q36MTRSynthesisError):
        module.synthesis_prompt("Question", "0" * 64, [_candidate("A")])


def test_synthesis_offsets_cover_all_cyclic_orders() -> None:
    identity = hashlib.sha256(b"offset-identity").hexdigest()
    candidates = [_candidate("A"), _candidate("B"), _candidate("C")]
    orders = [
        module.synthesis_prompt("Question", identity, candidates, offset)[1]
        for offset in range(3)
    ]
    assert len({tuple(order) for order in orders}) == 3
    assert all(set(order) == set(module.sparse.LINEAGES) for order in orders)


def test_synthesis_rejects_rotation_outside_three_cycles() -> None:
    with pytest.raises(module.Q36MTRSynthesisError, match="rotation"):
        module._rotation("0" * 64, 3)
