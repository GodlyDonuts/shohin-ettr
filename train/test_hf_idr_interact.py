#!/usr/bin/env python3
"""Tests for direct internal-draft/revision interaction."""

import pytest

from hf_idr_interact import IDRInteractionError, revision_prompt


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
