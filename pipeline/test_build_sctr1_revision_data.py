from __future__ import annotations

from build_sctr1_revision_data import selective_target, shuffled_source_commands


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


def test_shuffled_commands_preserve_stratum_counts_and_change_assignments() -> None:
    sources = {
        f"{index:064x}": {
            "task": "math500",
            "presentations": 1,
            "command": "keep" if index < 2 else "revise",
        }
        for index in range(6)
    }
    assigned = shuffled_source_commands(sources)
    assert sorted(assigned.values()) == sorted(
        source["command"] for source in sources.values()
    )
    assert any(assigned[key] != sources[key]["command"] for key in sources)
