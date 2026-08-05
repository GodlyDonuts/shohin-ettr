import hashlib

from build_visible_code_repair_bank import (
    SCHEMA,
    VisibleCodeRepairError,
    build_repair_rows,
)


def _bank(identity: str) -> dict:
    return {
        "identity_sha256": identity,
        "task": "mbpp",
        "text": "Return x + 1.",
        "test_list": ["assert inc(1) == 2"],
    }


def _candidate(identity: str, index: int, passed: bool) -> dict:
    return {
        "identity_sha256": identity,
        "task": "mbpp",
        "sample_index": index,
        "completion": "def inc(x):\n    return x",
        "execution": {"passed": passed, "stderr": "AssertionError"},
    }


def _selection(identity: str, correct: bool = False) -> dict:
    return {
        "schema": "shohin-visible-code-candidate-selection-v1",
        "results": [
            {
                "identity_sha256": identity,
                "task": "mbpp",
                "selected_sample_index": 0,
                "selected_correct": correct,
            }
        ],
    }


def test_builds_visible_only_repair_prompt() -> None:
    rows = build_repair_rows(
        [_bank("a")],
        [_candidate("a", 0, False)],
        _selection("a"),
        diagnostic_chars=100,
    )
    assert len(rows) == 1
    assert rows[0]["repair_schema"] == SCHEMA
    assert rows[0]["original_identity_sha256"] == "a"
    assert rows[0]["root_identity_sha256"] == "a"
    assert rows[0]["original_task_text"] == "Return x + 1."
    assert rows[0]["repair_round"] == 1
    assert "Previous solution" in rows[0]["text"]
    assert "AssertionError" in rows[0]["text"]
    assert rows[0]["test_list"] == ["assert inc(1) == 2"]


def test_second_round_recovers_root_task_without_nested_prompt() -> None:
    first = build_repair_rows(
        [_bank("a")],
        [_candidate("a", 0, False)],
        _selection("a"),
        diagnostic_chars=100,
    )[0]
    second_identity = first["identity_sha256"]
    second_selection = _selection(second_identity)
    second_candidate = _candidate(second_identity, 0, False)
    second_candidate["completion"] = "def inc(x):\n    return x + 2"
    rows = build_repair_rows(
        [first],
        [second_candidate],
        second_selection,
        diagnostic_chars=100,
    )
    assert len(rows) == 1
    assert rows[0]["repair_round"] == 2
    assert rows[0]["root_identity_sha256"] == "a"
    assert rows[0]["original_task_text"] == "Return x + 1."
    assert rows[0]["text"].count("Repair the previous Python solution") == 1
    assert "def inc(x):\n    return x + 2" in rows[0]["text"]


def test_derives_evaluator_identity_for_canonical_bank_row() -> None:
    identity = hashlib.sha256(b"mbpp\0Return x + 1.").hexdigest()
    bank = _bank(identity)
    bank.pop("identity_sha256")
    rows = build_repair_rows(
        [bank],
        [_candidate(identity, 0, False)],
        _selection(identity),
        diagnostic_chars=100,
    )
    assert rows[0]["original_identity_sha256"] == identity


def test_drops_already_correct_selection() -> None:
    assert (
        build_repair_rows(
            [_bank("a")],
            [_candidate("a", 0, True)],
            _selection("a", correct=True),
            diagnostic_chars=100,
        )
        == []
    )


def test_rejects_passing_candidate_marked_failed() -> None:
    try:
        build_repair_rows(
            [_bank("a")],
            [_candidate("a", 0, True)],
            _selection("a"),
            diagnostic_chars=100,
        )
    except VisibleCodeRepairError as exc:
        assert "passing candidate" in str(exc)
    else:
        raise AssertionError("expected a fail-closed selection error")
