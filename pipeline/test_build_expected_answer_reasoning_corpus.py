from __future__ import annotations

from build_expected_answer_reasoning_corpus import (
    expected_normalized,
    parse_requirements,
    select_rows,
)


def fixture(answer: str, response_answer: str, problem_type: str = "has_answer_extracted"):
    return {
        "problem": "Determine the exact integer obtained by adding twelve and thirty.",
        "generated_solution": (
            "We add the two supplied integers carefully, independently verify the "
            "arithmetic, and report the requested exact result. " * 2
        ) + rf"Therefore the final answer is \boxed{{{response_answer}}}.",
        "expected_answer": answer,
        "problem_type": problem_type,
        "problem_source": "fixture",
        "generation_model": "teacher",
    }


def test_expected_answer_normalization() -> None:
    assert expected_normalized(r"\boxed{\dfrac{1}{2}}") == r"\frac{1}{2}"
    assert expected_normalized(" C ") == "c"


def test_selection_and_requirements() -> None:
    rows = [
        fixture("42", "42"),
        fixture("42", "41"),
        fixture("42", "42", "converted_proof"),
    ]
    selected, counters = select_rows(
        rows,
        dataset_id="fixture",
        prompt_field="problem",
        response_field="generated_solution",
        answer_field="expected_answer",
        domain="math",
        source_field="problem_source",
        license_field=None,
        metadata_fields=["generation_model", "problem_type"],
        requirements={"problem_type": "has_answer_extracted"},
        eval_exact=set(),
        eval_ngrams=set(),
        maximum_rows=100,
        seed=31,
    )
    assert len(selected) == 1
    assert selected[0]["verification"] == "expected_answer_match_v1"
    assert selected[0]["expected_answer_normalized"] == "42"
    assert counters["response_answer_mismatch"] == 1
    assert counters["required_field_rejected"] == 1


def test_requirement_parser() -> None:
    assert parse_requirements(["problem_type=has_answer_extracted"]) == {
        "problem_type": "has_answer_extracted"
    }


if __name__ == "__main__":
    test_expected_answer_normalization()
    test_selection_and_requirements()
    test_requirement_parser()
