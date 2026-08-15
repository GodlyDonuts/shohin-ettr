from __future__ import annotations

import score_q36_mtr_external_consensus as module


def _rows(answers):
    return {
        arm: {
            "identity_sha256": "a" * 64,
            "task": "math500",
            "completion": rf"\boxed{{{answer}}}",
        }
        for arm, answer in zip(module.ARMS, answers, strict=True)
    }


def test_plurality_prefers_interpolation_on_tie():
    rows = _rows(("1", "2", "2", "3", "1"))
    assert module.choose("plurality", "math500", rows) == "interpolation"


def test_conservative_unchanged_requires_three_challengers():
    assert (
        module.choose(
            "conservative_unchanged",
            "math500",
            _rows(("1", "2", "2", "2", "2")),
        )
        == "interpolation"
    )
    assert (
        module.choose(
            "conservative_unchanged",
            "math500",
            _rows(("1", "2", "2", "3", "3")),
        )
        == "unchanged"
    )


def test_aligned_agreement_and_retention():
    rows = _rows(("1", "2", "3", "3", "3"))
    assert module.choose("aligned_agreement", "math500", rows) == "interpolation"
    rows = _rows(("1", "1", "2", "3", "2"))
    assert module.choose("interpolation_retention", "math500", rows) == "unchanged"


def test_code_uses_interpolation_control():
    rows = {
        arm: {"identity_sha256": "a" * 64, "task": "mbpp", "completion": "pass"}
        for arm in module.ARMS
    }
    for rule in module.RULES:
        assert module.choose(rule, "mbpp", rows) == "interpolation"
