from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.select_opencode_reasoning2_candidates import (
    OpenCodeReasoningCandidateError,
    _read_candidate_files,
    candidate_from_row,
    select_candidates,
)


def _row(**changes):
    row = {
        "id": "candidate-a",
        "question_id": "question-a",
        "question": "-",
        "r1_generation": "<think>"
        + ("reason " * 50)
        + "</think>\n```python\nprint(3)\n```",
        "qwq_critique": "",
        "solution": "print(3)",
        "judgement": "right",
        "pass_rate": 1.0,
        "source": "CodeChef",
        "license": "apache-2.0",
        "dataset": "apps",
        "split": "train",
        "difficulty": "interview",
        "index": "7",
    }
    row.update(changes)
    return row


def test_candidate_accepts_complete_embedded_verified_python() -> None:
    candidate, drop = candidate_from_row(
        _row(),
        allowed_datasets=frozenset({"apps"}),
        min_response_chars=128,
        max_response_chars=10_000,
        max_code_chars=1_000,
    )
    assert drop is None
    assert candidate is not None
    assert candidate["source_index"] == 7
    assert candidate["verification"] == "pending_source_test_replay"


@pytest.mark.parametrize(
    ("changes", "drop"),
    [
        ({"judgement": "wrong"}, "judgement"),
        ({"pass_rate": 0.99}, "pass_rate"),
        ({"split": "test"}, "source_identity"),
        ({"r1_generation": "reason without tags"}, "reasoning"),
        ({"r1_generation": "<think>reason</think>"}, "reasoning"),
        ({"solution": "def broken("}, "solution"),
        ({"solution": "print(4)"}, "solution_not_embedded"),
        ({"license": ""}, "license"),
    ],
)
def test_candidate_rejects_untrusted_rows(changes, drop) -> None:
    candidate, actual = candidate_from_row(
        _row(**changes),
        allowed_datasets=frozenset({"apps"}),
        min_response_chars=16,
        max_response_chars=10_000,
        max_code_chars=1_000,
    )
    assert candidate is None
    assert actual == drop


def test_selection_keeps_shortest_complete_candidate_per_problem() -> None:
    long = _row(id="long")
    short = _row(
        id="short",
        r1_generation="<think>" + ("x " * 140) + "</think>\n```python\nprint(3)\n```",
    )
    rows, counters = select_candidates(
        [long, short], min_response_chars=32, max_response_chars=10_000
    )
    assert len(rows) == 1
    assert rows[0]["ocr2_id"] == "short"
    assert counters["duplicate_candidate_rows"] == 1
    assert counters["candidate_replacements"] == 1


def test_merge_rejects_duplicate_identity_inside_one_shard(tmp_path: Path) -> None:
    rows, _ = select_candidates([_row()], min_response_chars=32)
    path = tmp_path / "part-000.jsonl"
    payload = json.dumps(rows[0]) + "\n"
    path.write_text(payload + payload, encoding="utf-8")
    with pytest.raises(OpenCodeReasoningCandidateError, match="duplicate identity"):
        _read_candidate_files([path])


def test_merge_prefers_shorter_cross_shard_candidate(tmp_path: Path) -> None:
    long, _ = select_candidates([_row(id="long")], min_response_chars=32)
    short, _ = select_candidates(
        [
            _row(
                id="short",
                r1_generation="<think>"
                + ("x " * 140)
                + "</think>\n```python\nprint(3)\n```",
            )
        ],
        min_response_chars=32,
    )
    first = tmp_path / "part-000.jsonl"
    second = tmp_path / "part-001.jsonl"
    first.write_text(json.dumps(long[0]) + "\n", encoding="utf-8")
    second.write_text(json.dumps(short[0]) + "\n", encoding="utf-8")
    rows, counters = _read_candidate_files([first, second])
    assert len(rows) == 1
    assert rows[0]["ocr2_id"] == "short"
    assert counters["cross_shard_duplicates"] == 1
    assert counters["cross_shard_replacements"] == 1
