#!/usr/bin/env python3
"""Tests for corpus-derived NTA1 row construction."""

from build_diverge_nta1_board import derive_row


def main() -> None:
    source = {
        "source": "reasoning_gym_trace",
        "training_group": "procedural",
        "question": "Compute 5 + 3 * 2.",
        "response": "<think>5 + 3 = 8 ; 8 * 2 = 16</think>\nThe answer is 16.",
        "answer": "16",
    }
    row = derive_row(source, "a" * 64)
    assert row is not None
    assert row["depth"] == 2
    assert row["answer"] == "16"
    assert row["wrong_answer"] != row["answer"]
    assert row["correct_steps"] == ["5 + 3 = 8", "8 * 2 = 16"]
    print("diverge NTA1 board tests passed")


if __name__ == "__main__":
    main()
