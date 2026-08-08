from __future__ import annotations

from build_idr1_revision_data import (
    EVAL_SCHEMA,
    TRAIN_SCHEMA,
    internal_revision_prompt,
)


def test_idr1_schemas_are_distinct() -> None:
    assert TRAIN_SCHEMA != EVAL_SCHEMA
    assert "idr1" in TRAIN_SCHEMA and "idr1" in EVAL_SCHEMA


def test_internal_revision_prompt_has_one_model_owned_draft() -> None:
    prompt = internal_revision_prompt("problem", "draft", "math500")
    assert prompt.count("Internal draft:") == 1
    assert prompt.count("Candidate") == 0
    assert prompt.count("Original problem:\nproblem") == 2
    assert "\\boxed{}" in prompt


def test_internal_revision_prompt_preserves_code_format() -> None:
    prompt = internal_revision_prompt("write f", "def f(): pass", "mbpp")
    assert "only executable Python code" in prompt
