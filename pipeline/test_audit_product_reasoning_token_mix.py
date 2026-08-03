from __future__ import annotations

import json
from pathlib import Path

from pipeline.audit_product_reasoning_token_mix import (
    audit,
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


def test_audit_counts_fully_untruncated_rows_and_targets(tmp_path: Path) -> None:
    class Tokenizer:
        name_or_path = "fake"

        def apply_chat_template(self, messages, **kwargs):
            del kwargs
            return messages[-1]["content"]

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return text.split()

    data = tmp_path / "rows.jsonl"
    rows = [
        {"question": "short", "response": "one two", "training_group": "math"},
        {
            "question": "short",
            "response": " ".join(["long"] * 40),
            "training_group": "math",
        },
    ]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = audit(
        data=data,
        tokenizer=Tokenizer(),
        model_revision="test",
        max_sequence_length=16,
        workspace_slots=0,
    )
    metrics = report["groups"]["math"]
    assert metrics["fully_untruncated_rows"] == 1
    assert metrics["fully_untruncated_target_tokens"] == 3
