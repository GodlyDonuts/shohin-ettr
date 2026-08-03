from __future__ import annotations

from fetch_product_reasoning_evals import normalize_aime, normalize_bbh


def test_aime_normalization_requires_complete_board() -> None:
    rows = [
        {"id": index, "problem": f"p{index}", "answer": str(index), "year": "2024"}
        for index in range(30)
    ]
    normalized = normalize_aime(list(reversed(rows)))
    assert len(normalized) == 30
    assert normalized[0]["id"] == 0


def test_bbh_normalization_binds_task_and_revision() -> None:
    rows = [{"input": f"q{index}", "target": "(A)"} for index in range(100)]
    normalized = normalize_bbh("logical_deduction_three_objects", rows)
    assert len(normalized) == 100
    assert normalized[0]["task"] == "logical_deduction_three_objects"
    assert normalized[0]["target"] == "(A)"
