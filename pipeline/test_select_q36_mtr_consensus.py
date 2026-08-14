from __future__ import annotations

import select_q36_mtr_consensus as module


def _row(task: str, completion: str, identity: str = "0" * 64) -> dict:
    return {
        "identity_sha256": identity,
        "task": task,
        "completion": completion,
    }


def test_consensus_selects_plurality_with_frozen_tie_order() -> None:
    rows = {
        arm: _row("bbh_logic", value)
        for arm, value in zip(
            module.ARM_ORDER,
            ("A", "B", "B", "C", "B", "A"),
            strict=True,
        )
    }
    assert module.choose(rows) == ("interpolation", 3)
    rows["direct"] = _row("bbh_logic", "A")
    assert module.choose(rows) == ("hierarchy", 3)


def test_consensus_uses_interpolation_for_executable_code() -> None:
    rows = {arm: _row("mbpp", arm) for arm in module.ARM_ORDER}
    assert module.choose(rows) == ("interpolation", 1)


def test_consensus_rejects_identity_mismatch() -> None:
    rows = {arm: _row("math500", r"\boxed{1}") for arm in module.ARM_ORDER}
    rows["challenger"] = _row("math500", r"\boxed{1}", "1" * 64)
    try:
        module.choose(rows)
    except module.Q36MTRConsensusError as error:
        assert "identity" in str(error)
    else:
        raise AssertionError("identity mismatch was accepted")
