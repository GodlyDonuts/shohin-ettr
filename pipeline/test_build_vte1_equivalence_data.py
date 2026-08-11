"""Tests for VTE1's verified transaction equivalence sets."""

from __future__ import annotations

import pytest

from build_vte1_equivalence_data import VTE1DataError, verified_equivalence_set
from kcr1_branch_transducer import execute_transaction


def _math() -> dict[str, str]:
    return {
        "training_group": "math",
        "expected_answer_normalized": "5",
        "verification": "expected_answer_match_v1",
        "response": r"Reasoning. \boxed{5}",
    }


def test_exact_draft_allows_keep_and_restart() -> None:
    source = _math()
    verified = source["response"]
    values = verified_equivalence_set(source, verified, verified)
    assert values[0] == "<KEEP>"
    assert any(value.startswith("<RESTART>\n") for value in values)
    assert all(
        execute_transaction(verified, value).endswith(verified) for value in values
    )


def test_prefix_draft_allows_exact_continue_and_restart() -> None:
    source = _math()
    verified = source["response"]
    draft = "Reasoning. "
    values = verified_equivalence_set(source, draft, verified)
    assert execute_transaction(draft, values[0]) == verified
    assert {value.partition("\n")[0] for value in values} == {
        "<CONTINUE>",
        "<RESTART>",
    }


def test_wrong_natural_math_allows_verified_correction_append() -> None:
    source = _math()
    verified = source["response"]
    draft = r"Wrong. \boxed{4}"
    values = verified_equivalence_set(source, draft, verified)
    assert {value.partition("\n")[0] for value in values} == {
        "<CONTINUE>",
        "<RESTART>",
    }


def test_code_excludes_append_correction() -> None:
    verified = "def f():\n    return 5\n"
    source = {
        "training_group": "code",
        "verification": "execution_verified_source_tests",
        "response": verified,
    }
    values = verified_equivalence_set(source, "def f():\n    return 4\n", verified)
    assert len(values) == 1
    assert values[0].startswith("<RESTART>\n")


def test_source_receipt_covers_choices_outside_narrow_benchmark_parser() -> None:
    verified = r"Reasoning. \boxed{\text{G}}"
    source = {
        "training_group": "science",
        "verification": "expected_answer_match_v1",
        "expected_answer_normalized": r"\text{g}",
        "response": verified,
    }
    values = verified_equivalence_set(source, verified, verified)
    assert values[0] == "<KEEP>"
    wrong_values = verified_equivalence_set(source, r"Wrong. \boxed{A}", verified)
    assert len(wrong_values) == 1
    assert wrong_values[0].startswith("<RESTART>\n")


def test_unverified_source_fails_closed() -> None:
    source = _math()
    source["verification"] = "unverified"
    with pytest.raises(VTE1DataError, match="verified target differs"):
        verified_equivalence_set(source, source["response"], source["response"])


def test_legacy_reasoning_gym_requires_exact_source_and_semantic_match() -> None:
    verified = "Reasoning. The answer is 42."
    source = {
        "training_group": "procedural",
        "source": "reasoning_gym_trace",
        "verification": None,
        "answer": "42",
        "response": verified,
    }
    assert verified_equivalence_set(source, verified, verified)[0] == "<KEEP>"
    source["source"] = "unknown"
    with pytest.raises(VTE1DataError, match="verified target differs"):
        verified_equivalence_set(source, verified, verified)
