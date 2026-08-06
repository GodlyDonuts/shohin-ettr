#!/usr/bin/env python3
"""Focused checks for the frozen DIVERGE-TOL3 confirmation renderer."""

import random

from diverge_tol1_data import OOD_NAMES, TRAIN_NAMES, clause_from_record, validate_row
from diverge_tol1_ir import DIRECT_OPS
from diverge_tol2_anchor_decoder import split_guard
from diverge_tol3_confirmation_data import (
    CONFIRMATION_NAMES,
    generate_confirmation_row,
)


def main() -> None:
    assert not set(CONFIRMATION_NAMES) & set(TRAIN_NAMES)
    assert not set(CONFIRMATION_NAMES) & set(OOD_NAMES)
    rows = [
        generate_confirmation_row(random.Random(2026080506 + index), index=index)
        for index in range(24)
    ]
    for row in rows:
        validate_row(row, "ood")
        assert 15 <= int(row["body_depth"]) <= 20
        for record in row["clauses"]:
            clause = clause_from_record(record)
            instruction = clause.instruction
            if instruction.operation == "GUARD":
                regions = split_guard(clause.text)
                assert clause.text.startswith("otherwise ")
                assert regions.true_action and regions.false_action and regions.predicate
            elif instruction.operation in DIRECT_OPS:
                assert instruction.action is not None
            elif instruction.operation == "SWAP":
                assert clause.text.startswith("with ") and ", exchange " in clause.text
            else:
                assert clause.text.startswith("with ") and clause.text.endswith(", return.")
    print("diverge TOL3 confirmation data tests passed")


if __name__ == "__main__":
    main()
