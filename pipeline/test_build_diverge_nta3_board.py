#!/usr/bin/env python3
"""Static contract checks for the NTA3 board builder."""

from diverge_nta3_scanner import scan_transactions


def main() -> None:
    document = (
        "Problem:\nThere are 2 unrelated numbers and 4 more.\n\n"
        "Candidate reasoning:\n7 + 3 = 11 ; 11 * 2 = 22"
        "\n\nCandidate final answer: 22"
    )
    assert scan_transactions(document) == ("7 + 3 = 11", "11 * 2 = 22")
    print("diverge NTA3 board tests passed")


if __name__ == "__main__":
    main()
