from __future__ import annotations

import pytest

from diverge_vcr1_data import (
    DRAFT_OPEN,
    VCR1DataError,
    tokenize_correction_example,
)


class CharacterTokenizer:
    chat_template = None
    is_fast = True
    eos_token_id = 0
    pad_token_id = 0

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) + 1 for character in text]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ):
        assert not add_special_tokens and return_offsets_mapping
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def test_correction_tokenization_preserves_disjoint_segments() -> None:
    tokens = tokenize_correction_example(
        CharacterTokenizer(),
        "What is 7 + 5?",
        "I think it is 13.",
        "The sum is 12. Final answer: 12",
        max_sequence_length=2000,
        workspace_slots=8,
    )
    assert tokens is not None
    assert any(tokens.question_mask)
    assert any(tokens.draft_mask)
    assert not any(
        question and draft
        for question, draft in zip(tokens.question_mask, tokens.draft_mask, strict=True)
    )
    assert tokens.response_ids[-1] == 0


def test_correction_tokenization_rejects_truncation() -> None:
    tokens = tokenize_correction_example(
        CharacterTokenizer(),
        "Q",
        "D" * 200,
        "T" * 200,
        max_sequence_length=64,
        workspace_slots=8,
    )
    assert tokens is None


def test_reserved_markers_fail_closed() -> None:
    with pytest.raises(VCR1DataError, match="reserved"):
        tokenize_correction_example(
            CharacterTokenizer(),
            f"bad {DRAFT_OPEN}",
            "draft",
            "target",
            max_sequence_length=1000,
            workspace_slots=8,
        )
