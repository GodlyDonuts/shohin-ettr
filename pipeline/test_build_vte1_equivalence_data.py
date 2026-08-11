"""Tests for VTE1's verified transaction equivalence sets."""

from __future__ import annotations

from build_vte1_equivalence_data import verified_equivalence_set
from kcr1_branch_transducer import execute_transaction


def _math() -> dict[str, str]:
    return {
        "training_group": "math",
        "expected_answer_normalized": "5",
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
    source = {"training_group": "code", "response": verified}
    values = verified_equivalence_set(source, "def f():\n    return 4\n", verified)
    assert len(values) == 1
    assert values[0].startswith("<RESTART>\n")
