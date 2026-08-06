#!/usr/bin/env python3
"""Focused deterministic tests for the SOT1 fresh query surfaces."""

from __future__ import annotations

from diverge_iem1_data import _symbol_role_ids
from diverge_sot1_data import query_confirmation_text


def main() -> None:
    symbols = ("rhea", "talos", "vesper", "yarrow", "zephyr")
    for renderer in range(3):
        text = query_confirmation_text(
            renderer,
            target="talos",
            distractor="vesper",
        )
        roles = _symbol_role_ids(
            text,
            symbols,
            target="talos",
            distractor="vesper",
        )
        assert sorted(roles) == [0, 1]
        assert "talos" in text and "vesper" in text
    print("diverge SOT1 data tests passed")


if __name__ == "__main__":
    main()
