#!/usr/bin/env python3
"""Tests for direct internal-draft/revision interaction."""

import json

import pytest

from hf_idr_interact import IDRInteractionError, load_question_rows, revision_prompt


def test_revision_prompt_preserves_source_and_draft():
    prompt = revision_prompt("What is 17 * 23?", "17 * 20 = 340", "math")
    assert prompt.count("What is 17 * 23?") == 2
    assert "17 * 20 = 340" in prompt
    assert "\\boxed{}" in prompt
    assert "a critique" in prompt


def test_response_modes_have_distinct_output_contracts():
    general = revision_prompt("Explain it.", "Draft.", "general")
    code = revision_prompt("Write it.", "def f(): pass", "code")
    assert "complete corrected answer" in general
    assert "executable code" in code


@pytest.mark.parametrize(
    ("question", "draft", "mode"),
    [("", "draft", "general"), ("question", "", "general"), ("q", "d", "bad")],
)
def test_revision_prompt_fails_closed(question, draft, mode):
    with pytest.raises(IDRInteractionError):
        revision_prompt(question, draft, mode)


def test_load_question_rows_preserves_modes(tmp_path):
    path = tmp_path / "questions.jsonl"
    rows = [
        {"id": "algebra", "question": "Solve x + 3 = 7.", "response_mode": "math"},
        {"id": "program", "question": "Write f().", "response_mode": "code"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert load_question_rows(path) == rows


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [{"id": "x", "question": "", "response_mode": "general"}],
        [
            {"id": "x", "question": "one", "response_mode": "general"},
            {"id": "x", "question": "two", "response_mode": "general"},
        ],
        [{"id": "x", "question": "one", "response_mode": "unknown"}],
    ],
)
def test_load_question_rows_fails_closed(tmp_path, rows):
    path = tmp_path / "questions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(IDRInteractionError):
        load_question_rows(path)
