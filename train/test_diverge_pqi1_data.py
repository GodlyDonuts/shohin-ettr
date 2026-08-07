#!/usr/bin/env python3
"""Focused tests for the frozen PQI1 confirmation surface."""

from diverge_nve1_data import symbol_occurrence_groups
from diverge_pqi1_data import PQI1_NAMES, query_text


def test_name_bank() -> None:
    assert len(PQI1_NAMES) == 32
    assert len(set(PQI1_NAMES)) == 32


def test_query_renderers_expose_two_mentions() -> None:
    for renderer in range(6):
        text = query_text(
            renderer, target=PQI1_NAMES[0], distractor=PQI1_NAMES[1]
        )
        groups = symbol_occurrence_groups(text, PQI1_NAMES)
        assert len(groups) == 2
        assert all(len(spans) == 1 for _, spans in groups)


if __name__ == "__main__":
    test_name_bank()
    test_query_renderers_expose_two_mentions()
    print("DIVERGE-PQI1 data tests passed")
