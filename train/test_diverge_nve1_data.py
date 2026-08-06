#!/usr/bin/env python3
"""Focused deterministic data tests for DIVERGE-NVE1."""

from __future__ import annotations

from collections import Counter

from diverge_nve1_data import (
    TRAIN_ROWS,
    generate_training_records,
    scan_rational_spans,
    symbol_occurrence_groups,
)


def main() -> None:
    text = (
        "Reject register beacon; value -12/5 is not there. Following step 17, "
        "ignore beacon and use verified register apricot."
    )
    assert tuple(text[start:end] for start, end in scan_rational_spans(text)) == (
        "-12/5",
        "17",
    )
    groups = symbol_occurrence_groups(
        text, ("apricot", "beacon", "canvas", "dahlia", "equinox")
    )
    assert [group[0] for group in groups] == ["beacon", "apricot"]
    assert len(groups[0][1]) == 2

    rows = generate_training_records()
    assert len(rows) == TRAIN_ROWS
    assert len({row["source_text"] for row in rows}) == TRAIN_ROWS
    renderers = Counter(int(row["renderer"]) for row in rows)
    assert max(renderers.values()) - min(renderers.values()) == 1
    assert sum(tuple(row["numeric_role_ids"]) == (0, 1) for row in rows) in (
        25_000,
        25_001,
    )
    assert sum(tuple(row["symbol_role_ids"]) == (0, 1) for row in rows) in (
        25_000,
        25_001,
    )
    print("diverge NVE1 data tests passed")


if __name__ == "__main__":
    main()
