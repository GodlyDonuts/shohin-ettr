"""Tests for non-destructive product-candidate rescoring."""

from __future__ import annotations

from rescore_product_candidates import rescore


def test_math_rescore_repairs_numeric_surface_false_negative() -> None:
    rows = [
        {
            "task": "math500",
            "identity_sha256": "x",
            "sample_index": 0,
            "prediction": "1/4",
            "gold": r"\frac{1}{4}",
            "correct": False,
            "completion": "unchanged",
        }
    ]
    output, report = rescore(rows)
    assert output[0]["correct"] is True
    assert output[0]["completion"] == "unchanged"
    assert report["label_transitions"] == {"0->1": 1}
