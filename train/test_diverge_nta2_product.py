#!/usr/bin/env python3
"""Tests for NTA2 finite-state role projection."""

from diverge_ats1_data import ROLE_TO_ID, encode_bytes
from diverge_nta2_product import project_scalar_roles


def main() -> None:
    text = "-285 - 133 = -418"
    roles = project_scalar_roles(encode_bytes(text))
    for value, role in (("-285", "LHS_A"), ("133", "ARG1"), ("-418", "RHS_A")):
        start = text.index(value)
        assert all(
            roles[position + 1] == ROLE_TO_ID[role]
            for position in range(start, start + len(value))
        )
    assert roles[text.index(" - ") + 2] == ROLE_TO_ID["OTHER"]
    print("diverge NTA2 product tests passed")


if __name__ == "__main__":
    main()
