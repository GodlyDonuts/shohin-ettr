"""Tests for held-out self-consistency bank construction and selection."""

from __future__ import annotations

from build_product_self_consistency_bank import build_rows
from hf_product_reasoning_eval import TASKS
from select_product_self_consistency import (
    canonical_prediction,
    choose_modal_candidate,
    select,
)


def _candidate(index: int, prediction: str | None, correct: bool) -> dict:
    return {
        "identity_sha256": "identity",
        "task": "gsm8k",
        "sample_index": index,
        "prediction": prediction,
        "correct": correct,
        "completion": f"candidate {index}",
    }


def test_bank_uses_exact_evaluator_selection_and_prompt() -> None:
    source = [
        {"question": f"What is {index}+1?", "answer": f"#### {index + 1}"}
        for index in range(6)
    ]
    rows = build_rows("gsm8k", source, 3, 17)
    assert len(rows) == 3
    assert len({row["identity_sha256"] for row in rows}) == 3
    assert all(row["training_group"] == "sealed_self_consistency_eval" for row in rows)
    assert all(row["answer"].startswith("#### ") for row in rows)
    assert all(TASKS["gsm8k"]["gold"](row) is not None for row in rows)


def test_math_bank_round_trips_through_evaluator_gold_parser() -> None:
    rows = build_rows(
        "math500",
        [{"problem": "Compute one half.", "solution": r"Thus \boxed{\frac{1}{2}}."}],
        1,
        17,
    )
    assert rows[0]["answer"] == r"\boxed{\frac{1}{2}}"
    assert TASKS["math500"]["gold"](rows[0]) == r"\frac{1}{2}"


def test_modal_selector_normalizes_numeric_equivalence() -> None:
    candidates = [
        _candidate(0, "7.0", False),
        _candidate(1, "8", False),
        _candidate(2, "7", True),
        _candidate(3, "7.00", True),
    ]
    assert canonical_prediction("gsm8k", "7.00") == "7/1"
    assert choose_modal_candidate(candidates)["sample_index"] == 0


def test_ties_choose_first_sample_and_empty_answers_do_not_vote() -> None:
    candidates = [
        _candidate(0, "3", False),
        _candidate(1, None, True),
        _candidate(2, "4", True),
        _candidate(3, None, False),
    ]
    assert choose_modal_candidate(candidates)["sample_index"] == 0


def test_report_separates_first_selected_and_oracle() -> None:
    candidates = [
        _candidate(0, "2", False),
        _candidate(1, "3", True),
        _candidate(2, "3", True),
        _candidate(3, "4", False),
    ]
    report = select(candidates)
    assert report["first_correct"] == 0
    assert report["selected_correct"] == 1
    assert report["oracle_correct"] == 1
    assert report["selector_reads_gold"] is False
