from hf_product_reasoning_rollouts import choose_positive, score_completion


def test_score_completion_requires_correct_explicit_math_answer() -> None:
    row = {"task": "math500", "answer": r"\boxed{2}"}
    assert score_completion(row, r"Reasoning. Final answer: \boxed{2}")["correct"]
    assert not score_completion(row, "Reasoning reaches 2 but never commits.")["correct"]
    assert not score_completion(row, r"Final answer: \boxed{3}")["correct"]


def test_choose_positive_prefers_shortest_verified_trajectory() -> None:
    candidates = [
        {"correct": True, "generated_tokens": 80, "completion": "long", "sample_index": 0},
        {"correct": False, "generated_tokens": 10, "completion": "wrong", "sample_index": 1},
        {"correct": True, "generated_tokens": 20, "completion": "short", "sample_index": 2},
    ]
    assert choose_positive(candidates)["completion"] == "short"
    assert choose_positive([candidates[1]]) is None
