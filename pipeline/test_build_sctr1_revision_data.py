from __future__ import annotations

from build_sctr1_revision_data import selective_target


def _pair() -> dict:
    return {
        "candidates": [
            {"lineage": "base", "completion": "verified", "correct": True},
            {"lineage": "expert", "completion": "wrong", "correct": False},
        ]
    }


def test_verified_draft_is_preserved_without_reserialization() -> None:
    target, kind = selective_target(
        _pair(),
        {"task": "math500", "question": "q", "answer": "answer"},
        {"correct": True, "completion": "already right"},
    )
    assert target == "<KEEP>"
    assert kind == "keep_verified_draft"


def test_incorrect_draft_receives_complete_verified_revision() -> None:
    target, kind = selective_target(
        _pair(),
        {"task": "math500", "question": "q", "answer": "answer"},
        {"correct": False, "completion": "wrong"},
    )
    assert target == "<REVISE>\nverified"
    assert kind == "revise_verified_candidate"
