#!/usr/bin/env python3
"""Tests for natural arithmetic source roles."""

from diverge_ats1_data import ROLE_TO_ID
from diverge_nta1_data import natural_segment_target


def main() -> None:
    row = {
        "identity_sha256": "c" * 64,
        "wrong_steps": ["  -12 + 7 = -4  "],
        "correct_steps": ["  -12 + 7 = -5  "],
    }
    segment = natural_segment_target(row, 0, trace_kind="wrong")
    roles = segment.role_ids
    text = segment.text
    for value, role in (("-12", "LHS_A"), ("7", "ARG1"), ("-4", "RHS_A")):
        start = text.index(value)
        assert all(roles[position + 1] == ROLE_TO_ID[role] for position in range(start, start + len(value)))
    assert segment.operation_id == 0
    print("diverge NTA1 data tests passed")


if __name__ == "__main__":
    main()
