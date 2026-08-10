#!/usr/bin/env python3
"""Focused tests for prospectively frozen WGP1 source selection."""

from generate_wgp1_replacement_source import (
    ROWS_PER_FAMILY,
    SEED,
    question_sha256,
    select_family,
)


def _entries(count: int):
    return [
        {"question": f"question {index}", "answer": str(index)}
        for index in range(count)
    ]


def test_selection_is_disjoint_deterministic_and_complete() -> None:
    entries = _entries(ROWS_PER_FAMILY + 3)
    protected = {question_sha256("question 0")}
    first, counts = select_family("basic_arithmetic", entries, protected, set())
    second, _ = select_family("basic_arithmetic", entries, protected, set())
    assert first == second
    assert len(first) == ROWS_PER_FAMILY
    assert all(row["generator_seed"] == SEED for row in first)
    assert counts["protected_overlap"] == 1
    assert len({row["source_question_sha256"] for row in first}) == ROWS_PER_FAMILY


def test_duplicate_is_skipped() -> None:
    entries = _entries(ROWS_PER_FAMILY + 1)
    entries.insert(1, dict(entries[0]))
    selected, counts = select_family("products", entries, set(), set())
    assert len(selected) == ROWS_PER_FAMILY
    assert counts["duplicate"] == 1
