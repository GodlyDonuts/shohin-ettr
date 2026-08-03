from __future__ import annotations

from pipeline.audit_product_reasoning_token_mix import (
    question_response,
    truncate_lengths,
)


def test_question_response_uses_training_schema_fallbacks() -> None:
    assert question_response({"problem": "p", "solution": "s"}) == ("p", "s")
    assert question_response({"prompt": "p", "completion": "c"}) == ("p", "c")
    assert question_response({"question": "", "response": "r"}) is None


def test_truncate_lengths_matches_response_first_training_budget() -> None:
    assert truncate_lengths(
        100,
        200,
        max_sequence_length=1024,
        workspace_slots=0,
    ) == (100, 201, False, False)
    assert truncate_lengths(
        100,
        2000,
        max_sequence_length=1024,
        workspace_slots=0,
    ) == (9, 1016, True, True)
    assert truncate_lengths(
        1000,
        800,
        max_sequence_length=1024,
        workspace_slots=0,
    ) == (224, 801, False, True)
