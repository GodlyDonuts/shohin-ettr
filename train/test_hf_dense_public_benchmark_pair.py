from __future__ import annotations

from types import SimpleNamespace

import pytest

from hf_dense_public_benchmark_pair import (
    DenseBenchmarkGenerationError,
    matched_render_prompt,
    model_context_limit,
)


class Tokenizer:
    chat_template = None
    model_max_length = 4096


def test_matched_rendering_is_checkpoint_independent() -> None:
    tokenizer = Tokenizer()
    rendered = matched_render_prompt(tokenizer, "problem")
    assert "System: You are a careful reasoning assistant" in rendered
    assert "User: problem" in rendered
    assert rendered.endswith("Assistant:")


def test_context_limit_uses_most_conservative_real_bound() -> None:
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8192))
    assert model_context_limit(model, Tokenizer()) == 4096


def test_missing_context_limit_fails_closed() -> None:
    model = SimpleNamespace(config=SimpleNamespace())
    tokenizer = SimpleNamespace(model_max_length=10**30)
    with pytest.raises(DenseBenchmarkGenerationError, match="unavailable"):
        model_context_limit(model, tokenizer)
