from __future__ import annotations

import pytest

from gpt_oss_harmony import (
    FINAL_MARKER,
    GptOssHarmonyError,
    REASONING_EFFORT,
    RETURN_MARKER,
    extract_final_completion,
    render_prompt,
    tokenize_training_example,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["reasoning_effort"] == REASONING_EFFORT
        question = messages[0]["content"]
        prefix = (
            "<|start|>system<|message|>Reasoning: low<|end|>"
            f"<|start|>user<|message|>{question}<|end|>"
            "<|start|>assistant"
        )
        if kwargs["add_generation_prompt"]:
            return prefix
        return prefix + FINAL_MARKER + messages[1]["content"] + RETURN_MARKER

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode())

    def decode(self, ids, **kwargs):
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        return bytes(ids).decode()


def test_prompt_and_training_target_share_exact_prefix() -> None:
    tokenizer = FakeTokenizer()
    prompt = render_prompt(tokenizer, "What is 2+2?")
    prompt_ids, target_ids = tokenize_training_example(
        tokenizer,
        "What is 2+2?",
        "4",
        max_sequence_length=512,
    )
    assert bytes(prompt_ids).decode() == prompt
    assert bytes(target_ids).decode() == FINAL_MARKER + "4" + RETURN_MARKER


def test_training_refuses_truncation_instead_of_changing_targets() -> None:
    with pytest.raises(GptOssHarmonyError, match="token geometry"):
        tokenize_training_example(
            FakeTokenizer(),
            "What is 2+2?",
            "4",
            max_sequence_length=8,
        )


def test_final_projection_drops_analysis_and_special_channels() -> None:
    raw = (
        "<|channel|>analysis<|message|>reasoning<|end|>"
        + FINAL_MARKER
        + "  4  "
        + RETURN_MARKER
    )
    completion, receipt = extract_final_completion(FakeTokenizer(), list(raw.encode()))
    assert completion == "4"
    assert receipt["analysis_channel_present"] is True
    assert receipt["final_channel_present"] is True
    assert receipt["final_channel_terminated"] is True
    assert receipt["empty_final_completion"] is False


def test_missing_final_channel_is_explicitly_malformed() -> None:
    completion, receipt = extract_final_completion(
        FakeTokenizer(), list(b"<|channel|>analysis<|message|>unfinished")
    )
    assert completion == ""
    assert receipt["final_channel_present"] is False
    assert receipt["empty_final_completion"] is True
