from __future__ import annotations

import pytest

import audit_q36_mtr_training_tokens as module


def test_summary_records_exact_over_limit_geometry() -> None:
    report = module.summarize(
        [(0, 100, 60, 40), (1, 4097, 3000, 1097), (2, 4220, 3100, 1120)],
        4096,
    )
    assert report["maximum_total_tokens"] == 4220
    assert report["over_limit_count"] == 1
    assert report["over_limit"][0]["row_index"] == 2


def test_summary_rejects_empty_geometry() -> None:
    with pytest.raises(ValueError, match="geometry"):
        module.summarize([], 4096)
