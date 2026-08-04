"""Tests for immutable product-evaluation rescoring."""

from __future__ import annotations

from rescore_product_eval_report import rescore_report


def test_rescore_report_preserves_old_score_and_repairs_labels() -> None:
    payload = {
        "task": "math500",
        "total": 2,
        "correct": 0,
        "accuracy": 0.0,
        "results": [
            {"prediction": "1/4", "gold": r"\frac{1}{4}", "correct": False},
            {"prediction": "2", "gold": "3", "correct": False},
        ],
    }
    report = rescore_report(payload)
    assert report["original_correct"] == 0
    assert report["correct"] == 1
    assert report["accuracy"] == 0.5
    assert report["label_transitions"] == {"0->0": 1, "0->1": 1}
