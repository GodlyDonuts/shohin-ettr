#!/usr/bin/env python3
"""Deterministic board checks for DIVERGE-TOL1."""

from diverge_tol1_data import (
    ROLE_NAMES,
    clause_from_record,
    generate_split,
    source_candidates,
    split_report,
)


def main() -> None:
    train = generate_split("train", 24, 2026080501)
    repeat = generate_split("train", 24, 2026080501)
    ood = generate_split("ood", 24, 2026080503)
    assert train == repeat
    assert {row["identity_sha256"] for row in train}.isdisjoint(
        row["identity_sha256"] for row in ood
    )
    assert all(4 <= row["body_depth"] <= 8 for row in train)
    assert all(9 <= row["body_depth"] <= 14 for row in ood)
    report = split_report(ood)
    assert report["feature_counts"] == {
        "guard": 24,
        "swap": 24,
        "register_operand": 24,
        "rational": 24,
    }
    for row in (*train, *ood):
        for record in row["clauses"]:
            clause = clause_from_record(record)
            proposed = source_candidates(clause.text)
            assert [value.text for value in proposed] == [
                value.text for value in clause.candidates
            ]
            assert all(ROLE_NAMES[value.role_id] in ROLE_NAMES for value in clause.candidates)
    print("diverge TOL1 data tests passed")


if __name__ == "__main__":
    main()
