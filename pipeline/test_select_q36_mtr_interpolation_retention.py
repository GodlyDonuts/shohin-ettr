from __future__ import annotations

import select_q36_mtr_interpolation_retention as module


def _row(task: str, tokens: int, identity: str = "0" * 64) -> dict:
    return {
        "identity_sha256": identity,
        "task": task,
        "generated_tokens": tokens,
    }


def test_selection_rule_is_label_free_and_conservative() -> None:
    assert module.choose(_row("bbh_logic", 20), _row("bbh_logic", 2)) == "hierarchy"
    assert module.choose(_row("mbpp", 2), _row("mbpp", 20)) == "interpolation"
    assert module.choose(_row("math500", 8), _row("math500", 8)) == "interpolation"
    assert module.choose(_row("math500", 8), _row("math500", 9)) == "hierarchy"


def test_selection_rejects_identity_mismatch() -> None:
    try:
        module.choose(_row("math500", 8), _row("math500", 8, "1" * 64))
    except module.Q36MTRInterpolationRetentionError as error:
        assert "identity" in str(error)
    else:
        raise AssertionError("identity mismatch was accepted")
