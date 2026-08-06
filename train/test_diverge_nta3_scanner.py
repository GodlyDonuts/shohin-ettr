#!/usr/bin/env python3
"""Tests for the NTA3 full-document scanner."""

from diverge_nta3_scanner import scan_transactions


def main() -> None:
    document = (
        "Problem:\nCompute 5 + 3 * 2.\n\n"
        "Candidate reasoning:\n5 + 3 = 9 ; 9 * 2 = 18"
        "\n\nCandidate final answer: 18"
    )
    assert scan_transactions(document) == ("5 + 3 = 9", "9 * 2 = 18")
    print("diverge NTA3 scanner tests passed")


if __name__ == "__main__":
    main()
