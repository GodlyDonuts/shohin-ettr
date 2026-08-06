#!/usr/bin/env python3
"""CPU mechanics tests for the frozen DIVERGE-SRP1 data contract."""

from __future__ import annotations

from collections import Counter

from diverge_npw1_data import CONFIRMATION_NAMES, TRAIN_NAMES
from diverge_srp1_data import SRP1_BOARD_ROWS, SRP1_NAMES, query_text
from diverge_tfs1_data import TFS1_NAMES


def main() -> None:
    assert len(SRP1_NAMES) == len(set(SRP1_NAMES)) == 32
    assert not set(SRP1_NAMES) & set(TFS1_NAMES)
    assert not set(SRP1_NAMES) & set(TRAIN_NAMES)
    assert not set(SRP1_NAMES) & set(CONFIRMATION_NAMES)
    counts = Counter(
        (row_index + 2 * query_offset) % 6
        for row_index in range(SRP1_BOARD_ROWS)
        for query_offset in range(3)
    )
    assert counts == Counter({renderer: 128 for renderer in range(6)})
    surfaces = {
        query_text(renderer, target="alpha", distractor="omega")
        for renderer in range(6)
    }
    assert len(surfaces) == 6
    assert all("alpha" in text and "omega" in text for text in surfaces)
    print("DIVERGE-SRP1 data tests passed")


if __name__ == "__main__":
    main()

