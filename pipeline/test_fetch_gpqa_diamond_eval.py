"""Tests for exact GPQA-Diamond board normalization."""

from __future__ import annotations

import unittest

from fetch_gpqa_diamond_eval import GPQAFetchError, normalize_gpqa


def fixture(index: int) -> dict[str, str]:
    return {
        "Record ID": f"row-{index:03d}",
        "Question": f"Which scientific option is correct for case {index}?",
        "Correct Answer": f"correct-{index}",
        "Incorrect Answer 1": f"wrong-a-{index}",
        "Incorrect Answer 2": f"wrong-b-{index}",
        "Incorrect Answer 3": f"wrong-c-{index}",
        "High-level domain": "Physics",
        "Subdomain": "General",
    }


class GPQADiamondFetchTests(unittest.TestCase):
    def test_normalization_is_deterministic_and_hides_answer_text_from_label(self) -> None:
        rows = [fixture(index) for index in range(198)]
        first = normalize_gpqa(rows, 31)
        second = normalize_gpqa(list(reversed(rows)), 31)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 198)
        for row in first:
            self.assertIn(row["answer"], "ABCD")
            labels = [choice["label"] for choice in row["choices"]]
            self.assertEqual(labels, list("ABCD"))

    def test_correct_answer_duplicated_as_distractor_fails_closed(self) -> None:
        rows = [fixture(index) for index in range(198)]
        rows[0]["Incorrect Answer 1"] = rows[0]["Correct Answer"]
        with self.assertRaises(GPQAFetchError):
            normalize_gpqa(rows, 31)

    def test_duplicate_distractors_are_preserved(self) -> None:
        rows = [fixture(index) for index in range(198)]
        rows[0]["Incorrect Answer 2"] = rows[0]["Incorrect Answer 1"]
        normalized = normalize_gpqa(rows, 31)
        self.assertEqual(len(normalized), 198)


if __name__ == "__main__":
    unittest.main()
