#!/usr/bin/env python3
"""Tests for DIVERGE-CCR1 confirmation surfaces."""

from __future__ import annotations

from diverge_ccr1_data import CCR1_NAMES, query_text
from diverge_nve1_data import symbol_occurrence_groups


def main() -> None:
    assert len(CCR1_NAMES) == 32
    assert len(set(CCR1_NAMES)) == 32
    for renderer in range(6):
        text = query_text(renderer, target="asteroid", distractor="birchwood")
        groups = symbol_occurrence_groups(text, CCR1_NAMES)
        assert len(groups) == 2
        target_first = groups[0][0] == "asteroid"
        assert target_first == (renderer % 2 == 0)
    print("DIVERGE-CCR1 data tests passed")


if __name__ == "__main__":
    main()
