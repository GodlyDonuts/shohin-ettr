#!/usr/bin/env python3
"""Focused source extraction tests for DIVERGE-NFE1."""

from __future__ import annotations

from diverge_nfe1_data import (
    apply_scalar,
    corrupt_visible_operator,
    extract_verified_equations,
    scan_signed_integer_spans,
)


def main() -> None:
    text = "-12 + 7 = -5"
    spans = scan_signed_integer_spans(text)
    assert tuple(text[start:end] for start, end in spans) == ("-12", "7", "-5")
    adjacent = "12-5=7"
    assert tuple(
        adjacent[start:end] for start, end in scan_signed_integer_spans(adjacent)
    ) == (
        "12",
        "5",
        "7",
    )
    negative_argument = "12--5=17"
    assert tuple(
        negative_argument[start:end]
        for start, end in scan_signed_integer_spans(negative_argument)
    ) == ("12", "-5", "17")

    equations = extract_verified_equations(
        "First 12 + 5 = 17. Then 17 * 3 = 51; ignore 5 + 2 = 9."
    )
    assert len(equations) == 2
    assert [equation.operation for equation in equations] == ["add", "multiply"]
    corrupted = [corrupt_visible_operator(equation) for equation in equations]
    assert [equation.operator for equation in corrupted] == ["-", "+"]
    assert [
        tuple(int(equation.text[start:end]) for start, end in equation.mention_spans)
        for equation in corrupted
    ] == [(12, 5, 17), (17, 3, 51)]
    assert apply_scalar("subtract", 12, 5) == 7
    print("diverge NFE1 data tests passed")


if __name__ == "__main__":
    main()
