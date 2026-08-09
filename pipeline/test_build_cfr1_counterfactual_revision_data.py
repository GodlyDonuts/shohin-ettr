from build_cfr1_counterfactual_revision_data import (
    counterfactual_draft,
    donor_map,
    verification_admitted,
    wrong_answer,
)


def test_wrong_answer_changes_common_answer_types() -> None:
    assert wrong_answer("17", "00" * 32) == "18"
    assert wrong_answer("17", "01" * 32) == "16"
    assert wrong_answer("3/7", "00" * 32) == "4/7"
    assert wrong_answer("B", "00" * 32) == "C"
    assert wrong_answer("true", "00" * 32) == "false"


def test_counterfactual_drafts_are_decisively_wrong() -> None:
    math, math_kind = counterfactual_draft(
        {"response": "work\\n\\boxed{17}", "training_group": "math", "answer": "17"},
        "00" * 32,
    )
    assert math.endswith(r"\boxed{18}.")
    assert math_kind == "contradictory_final_answer"
    code, code_kind = counterfactual_draft(
        {"response": "print(1)", "training_group": "code"}, "00" * 32
    )
    assert code.startswith("raise RuntimeError")
    assert code_kind == "guaranteed_runtime_failure"


def test_verification_admission_is_domain_specific() -> None:
    assert verification_admitted(
        {"training_group": "math", "verification": "expected_answer_match_v1"}
    )
    assert verification_admitted(
        {"training_group": "code", "verification": "execution_verified"}
    )
    assert verification_admitted(
        {
            "training_group": "procedural",
            "verification": "reasoning_gym_answer_verified",
        }
    )
    assert not verification_admitted({"training_group": "code"})
    assert not verification_admitted(
        {"training_group": "science", "verification": "execution_verified"}
    )


def test_donor_map_never_assigns_same_source() -> None:
    rows = [
        {
            "source_identity_sha256": str(index) * 64,
            "training_group": "math",
            "counterfactual_draft": "x" * (10 + index),
        }
        for index in range(3)
    ]
    donors = donor_map(rows)
    assert set(donors) == {row["source_identity_sha256"] for row in rows}
    assert all(source != donor for source, donor in donors.items())
