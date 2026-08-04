"""Focused tests for the cheap candidate-reranking control."""

from __future__ import annotations

from product_candidate_reranker import FEATURE_NAMES, feature_vector, group_candidates


def _candidate(index: int, prediction: str, correct: bool) -> dict:
    return {
        "identity_sha256": "row",
        "task": "gsm8k",
        "sample_index": index,
        "prediction": prediction,
        "completion": f"Reasoning. Therefore the final answer is: {prediction}",
        "question": "What is 2 + 3?",
        "correct": correct,
        "gold": "5",
        "explicit_final_answer": True,
        "max_token_exhausted": False,
        "draft_max_token_exhausted": False,
        "generated_tokens": 12,
        "finalization_generated_tokens": 0,
        "finalization": None,
    }


def test_features_never_read_gold_or_correctness() -> None:
    group = [_candidate(0, "5", True), _candidate(1, "7", False)]
    before = feature_vector(group[0], group)
    mutated = [dict(row) for row in group]
    mutated[0]["correct"] = False
    mutated[0]["gold"] = "unrelated secret"
    mutated[1]["correct"] = True
    assert feature_vector(mutated[0], mutated) == before
    assert len(before) == len(FEATURE_NAMES)


def test_grouping_reorders_candidates_by_sample_index() -> None:
    rows = [_candidate(1, "7", False), _candidate(0, "5", True)]
    grouped = group_candidates(rows)
    assert [row["sample_index"] for row in grouped["row"]] == [0, 1]
