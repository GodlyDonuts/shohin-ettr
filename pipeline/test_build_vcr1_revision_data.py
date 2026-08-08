from __future__ import annotations

from build_vcr1_revision_data import (
    canonical_target,
    revision_prompt,
    source_task_prompt,
    training_target,
)


def _pair(outcome: str) -> dict:
    correctness = {
        "base_only": (True, False),
        "both_correct": (True, True),
        "both_wrong": (False, False),
        "expert_only": (False, True),
    }[outcome]
    return {
        "outcome_class": outcome,
        "candidates": [
            {
                "lineage": "base",
                "completion": "base completion",
                "correct": correctness[0],
            },
            {
                "lineage": "expert",
                "completion": "expert completion",
                "correct": correctness[1],
            },
        ],
    }


def test_targets_use_verified_candidate_or_source_repair() -> None:
    source = {"task": "math500", "answer": "\\boxed{7}", "question": "q"}
    assert training_target(_pair("base_only"), source) == (
        "base completion",
        "verified_candidate",
    )
    assert training_target(_pair("expert_only"), source) == (
        "expert completion",
        "verified_candidate",
    )
    assert training_target(_pair("both_wrong"), source) == (
        "\\boxed{7}",
        "source_verified_repair",
    )


def test_code_prompt_and_target_preserve_tests() -> None:
    source = {
        "task": "mbpp",
        "text": "Write f.",
        "test_list": ["assert f() == 1"],
        "code": "def f():\n    return 1",
    }
    prompt = source_task_prompt(source)
    assert "assert f() == 1" in prompt
    assert canonical_target(source).startswith("def f")


def test_revision_prompt_requests_solution_not_verdict() -> None:
    prompt = revision_prompt("problem", "attempt a", "attempt b")
    assert prompt.count("\nproblem") == 2
    assert "Candidate A" in prompt and "Candidate B" in prompt
    assert "complete final solution" in prompt
