#!/usr/bin/env python3
"""Fixed typed source questions for the DIVERGE-QTG1 gate."""

from __future__ import annotations

from diverge_mei1_runtime import REGISTER_COUNT


_ORDINAL = ("0", "1", "2", "3", "4")
FIELD_QUERIES = tuple(
    tuple(f"retrieve the {phase_name} value for slot {_ORDINAL[address]}".split())
    for phase_name in ("initial", "final")
    for address in range(REGISTER_COUNT)
)

if len(FIELD_QUERIES) != 2 * REGISTER_COUNT or len(set(FIELD_QUERIES)) != len(
    FIELD_QUERIES
):
    raise RuntimeError("QTG1 field questions are not a complete typed set")
