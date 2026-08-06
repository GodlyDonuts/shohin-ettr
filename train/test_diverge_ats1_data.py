#!/usr/bin/env python3
"""Focused standalone checks for ATS1 byte-role supervision."""

from diverge_ats1_data import OPERATION_TO_ID, ROLE_TO_ID, segment_target


def _row(family: str, step: str, operation: object) -> dict[str, object]:
    return {
        "family": family,
        "depth": 1,
        "program": [operation],
        "wrong_steps": [step],
        "correct_steps": [step],
        "identity_sha256": "a" * 64,
    }


def test_scalar_roles() -> None:
    target = segment_target(
        _row("scalar", "Step 1: -12 + 9 = -3.", ["add", 9]),
        0,
        trace_kind="wrong",
    )
    assert target.operation_id == OPERATION_TO_ID["SCALAR_ADD"]
    assert target.lhs_a == "-12" and target.rhs_a == "-3"
    assert target.arguments == ("9",)
    assert target.role_ids.count(ROLE_TO_ID["LHS_A"]) == 3


def test_register_roles() -> None:
    target = segment_target(
        _row(
            "register",
            "Step 1: add B to A: (A=-2, B=7) -> (A=5, B=7).",
            "A+=B",
        ),
        0,
        trace_kind="correct",
    )
    assert target.operation_id == OPERATION_TO_ID["REGISTER_A_ADD_B"]
    assert (target.lhs_a, target.lhs_b, target.rhs_a, target.rhs_b) == (
        "-2",
        "7",
        "5",
        "7",
    )


def test_symbol_roles() -> None:
    target = segment_target(
        _row(
            "symbolic",
            "Step 1: swap positions 2 and 5: abcde -> aecdb.",
            ["swap", 2, 5],
        ),
        0,
        trace_kind="wrong",
    )
    assert target.operation_id == OPERATION_TO_ID["SYMBOL_SWAP"]
    assert target.lhs_symbol == "abcde" and target.rhs_symbol == "aecdb"
    assert target.arguments == ("2", "5")


def main() -> None:
    test_scalar_roles()
    test_register_roles()
    test_symbol_roles()
    print("diverge ATS1 data tests passed")


if __name__ == "__main__":
    main()
