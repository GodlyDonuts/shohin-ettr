#!/usr/bin/env python3
"""Focused structural tests for the QTE1 composite."""

from __future__ import annotations

from diverge_qte1_composite import SYSTEM, _canonical_sha256


def main() -> None:
    assert "YES or NO" in SYSTEM
    assert len(_canonical_sha256({"qte1": 1})) == 64
    print("DIVERGE-QTE1 composite structural tests passed")


if __name__ == "__main__":
    main()
