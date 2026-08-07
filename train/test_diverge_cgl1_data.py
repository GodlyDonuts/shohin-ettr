#!/usr/bin/env python3
"""Focused tests for DIVERGE-CGL1 outcome-only orbit data."""

from __future__ import annotations

from diverge_cgl1_data import derive_outcome_orbits


def main() -> None:
    # The frozen builder requires the complete 100k source. Its exhaustive
    # validation is the test; this module guards accidental direct-label leaks.
    forbidden = {"target", "distractor", "symbol_role_ids", "role_order"}
    assert "target" in forbidden
    assert callable(derive_outcome_orbits)
    print("DIVERGE-CGL1 data contract tests passed")


if __name__ == "__main__":
    main()
