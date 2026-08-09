import inspect

import pytest

from hf_product_reasoning_rollouts import (
    ProductRolloutError,
    choose_positive,
    combine_finalization,
    render_rollout_prompt,
    run,
    score_completion,
    validate_generation_geometry,
)


def test_score_completion_requires_correct_explicit_math_answer() -> None:
    row = {"task": "math500", "answer": r"\boxed{2}"}
    assert score_completion(row, r"Reasoning. Final answer: \boxed{2}")["correct"]
    assert not score_completion(row, "Reasoning reaches 2 but never commits.")[
        "correct"
    ]
    assert not score_completion(row, r"Final answer: \boxed{3}")["correct"]


def test_score_completion_uses_normalized_science_answer() -> None:
    row = {
        "task": "bbh_logic",
        "answer": r"\boxed{0}",
        "expected_answer_normalized": "0",
    }
    score = score_completion(row, r"Therefore the answer is \boxed{0}.")
    assert score["gold"] == "0"
    assert score["correct"]


def test_score_completion_executes_mbpp_candidate() -> None:
    row = {
        "task": "mbpp",
        "text": "Return the sum of two integers.",
        "test_setup_code": "",
        "test_list": ["assert add(2, 3) == 5", "assert add(-2, 2) == 0"],
    }
    passing = score_completion(row, "def add(a, b):\n    return a + b", code_timeout=1)
    failing = score_completion(row, "def add(a, b):\n    return a - b", code_timeout=1)
    assert passing["correct"]
    assert passing["execution"]["passed"]
    assert not failing["correct"]


def test_choose_positive_requires_uninterrupted_autonomous_trajectory() -> None:
    candidates = [
        {
            "correct": True,
            "generated_tokens": 20,
            "completion": "truncated answer is 2 then restart",
            "sample_index": 0,
            "draft_max_token_exhausted": True,
            "finalization": None,
            "finalization_max_token_exhausted": False,
        },
        {
            "correct": False,
            "generated_tokens": 10,
            "completion": "wrong",
            "sample_index": 1,
        },
        {
            "correct": True,
            "generated_tokens": 100,
            "completion": r"exhausted derivation\n\n\boxed{2}",
            "sample_index": 2,
            "draft_max_token_exhausted": True,
            "finalization": r"\boxed{2}",
            "finalization_max_token_exhausted": False,
        },
        {
            "correct": True,
            "generated_tokens": 80,
            "completion": r"complete derivation\n\n\boxed{2}",
            "sample_index": 3,
            "draft_max_token_exhausted": False,
            "finalization": None,
            "finalization_max_token_exhausted": False,
        },
    ]
    assert choose_positive(candidates)["completion"].endswith(r"\boxed{2}")
    assert choose_positive(candidates)["sample_index"] == 3
    assert choose_positive([candidates[2]]) is None
    assert choose_positive([candidates[1]]) is None


def test_choose_positive_uses_shortest_within_same_completion_class() -> None:
    candidates = [
        {
            "correct": True,
            "generated_tokens": 80,
            "completion": "long",
            "sample_index": 0,
            "draft_max_token_exhausted": False,
        },
        {
            "correct": True,
            "generated_tokens": 20,
            "completion": "short",
            "sample_index": 1,
            "draft_max_token_exhausted": False,
        },
    ]
    assert choose_positive(candidates)["completion"] == "short"


def test_combine_finalization_only_appends_explicit_recovery() -> None:
    assert combine_finalization("draft", True, r"\boxed{7}") == "draft\n\n\\boxed{7}"
    assert combine_finalization("draft", False, r"\boxed{7}") == "draft"
    assert combine_finalization("draft", True, "still reasoning") == "draft"


def test_direct_bare_prompt_does_not_add_boxed_answer_wrapper() -> None:
    class Tokenizer:
        chat_template = None

    rendered = render_rollout_prompt(
        Tokenizer(),
        "Return only executable Python code.",
        adapter=False,
        enable_thinking=False,
        bare_prompt_style="direct",
    )
    assert rendered == "User: Return only executable Python code.\n\nAssistant:"
    assert "boxed" not in rendered


def test_rollout_generation_uses_resolved_adapter_mode() -> None:
    source = inspect.getsource(run)
    assert "rendered,\n            adapter," in source
    assert "finalize_rendered,\n                    adapter," in source
    assert "_render_prompt(\n                        tokenizer" not in source


def test_rollout_forwards_and_records_backbone_quantization() -> None:
    source = inspect.getsource(run)
    assert "quantization=args.quantization" in source
    assert 'else args.quantization' in source


def test_rollout_generation_geometry_separates_deployment_and_sampling() -> None:
    validate_generation_geometry("greedy", 1)
    validate_generation_geometry("qwen-thinking", 2)
    with pytest.raises(
        ProductRolloutError, match="greedy rollout collection requires one"
    ):
        validate_generation_geometry("greedy", 2)
    with pytest.raises(ProductRolloutError, match="requires 2--8 samples"):
        validate_generation_geometry("qwen-thinking", 1)
