from __future__ import annotations

import pytest

from score_product_reasoning_training_fit import ProductFitError, _weighted_mean


def test_weighted_mean_uses_charged_tokens() -> None:
    assert _weighted_mean([1.0, 3.0], [3, 1]) == 1.5


def test_weighted_mean_rejects_empty_accounting() -> None:
    with pytest.raises(ProductFitError):
        _weighted_mean([], [])
