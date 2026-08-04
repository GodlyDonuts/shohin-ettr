"""Focused tests for candidate hidden-feature extraction."""

from __future__ import annotations

import torch

from hf_product_candidate_features import bounded_token_rows, pool_hidden_states


def test_bounded_rows_preserve_completion_tail() -> None:
    tokens, start, truncated = bounded_token_rows(list(range(100)), list(range(200, 300)), 128)
    assert truncated is True
    assert len(tokens) == 128
    assert tokens[-1] == 299
    assert start == 96


def test_pooling_width_matches_layers_times_three() -> None:
    hidden = tuple(torch.randn(2, 8, 5) for _ in range(5))
    pooled = pool_hidden_states(hidden, (-1, -2), [8, 7], [4, 3], 2)
    assert pooled.shape == (2, 2 * 3 * 5)
