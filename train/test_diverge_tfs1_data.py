#!/usr/bin/env python3
"""Focused exact checks for the frozen DIVERGE-TFS1 board."""

from __future__ import annotations

from copy import deepcopy
import random

from diverge_tfs1_data import (
    FAULT_LINES,
    TFS1DataError,
    TFS1_NAMES,
    WORLDS,
    generate_row,
    steps_from_record,
    validate_row,
)
from diverge_tol1_data import OOD_NAMES, TRAIN_NAMES
from diverge_tol3_confirmation_data import CONFIRMATION_NAMES


def _must_reject(row: dict[str, object]) -> None:
    try:
        validate_row(row)
    except TFS1DataError:
        return
    raise AssertionError("corrupted TFS1 row was accepted")


def main() -> None:
    assert not set(TFS1_NAMES) & set(TRAIN_NAMES)
    assert not set(TFS1_NAMES) & set(OOD_NAMES)
    assert not set(TFS1_NAMES) & set(CONFIRMATION_NAMES)

    seed = 2026080607
    row = generate_row(random.Random(seed), index=0)
    duplicate = generate_row(random.Random(seed), index=0)
    assert row == duplicate
    validate_row(row)
    assert int(row["represented_worlds"]) == WORLDS
    assert int(row["partial_survivors"]) >= 2

    steps = steps_from_record(row["steps"])  # type: ignore[arg-type]
    ambiguous = [step for step in steps if step.options is not None]
    assert len(ambiguous) == FAULT_LINES
    assert [step.fault_index for step in ambiguous] == list(range(FAULT_LINES))
    assert all(step.text.count(" / ") == 1 for step in ambiguous)

    corrupted_evidence = deepcopy(row)
    corrupted_evidence["evidence"][0]["value"] = "999"  # type: ignore[index]
    _must_reject(corrupted_evidence)

    corrupted_enumeration = deepcopy(row)
    corrupted_enumeration["enumeration_sha256"] = "0" * 64
    _must_reject(corrupted_enumeration)

    corrupted_assignment = deepcopy(row)
    corrupted_assignment["gold_assignment"] = [0] * FAULT_LINES
    _must_reject(corrupted_assignment)
    print("diverge TFS1 data tests passed")


if __name__ == "__main__":
    main()
