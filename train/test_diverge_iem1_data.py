#!/usr/bin/env python3
"""Focused deterministic data tests for DIVERGE-IEM1."""

from __future__ import annotations

from collections import Counter

from diverge_iem1_data import QUERY_TRAIN_ROWS, generate_query_training_records


def main() -> None:
    rows = generate_query_training_records()
    assert len(rows) == QUERY_TRAIN_ROWS
    assert len({row["source_text"] for row in rows}) == QUERY_TRAIN_ROWS
    assert len({row["identity_sha256"] for row in rows}) == QUERY_TRAIN_ROWS
    renderers = Counter(int(row["renderer"]) for row in rows)
    assert max(renderers.values()) - min(renderers.values()) == 1
    orders = Counter(tuple(row["symbol_role_ids"]) for row in rows)
    assert set(orders) == {(0, 1), (1, 0)}
    assert abs(orders[(0, 1)] - orders[(1, 0)]) <= 1
    print("diverge IEM1 data tests passed")


if __name__ == "__main__":
    main()
