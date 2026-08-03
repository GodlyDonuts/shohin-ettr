"""Non-network tests for the Hugging Face product evaluator."""

from __future__ import annotations

import unittest

from hf_product_reasoning_eval import (
    ProductEvalError,
    extract_boxed,
    extract_gsm8k,
    gold_gsm8k,
    match_math,
    select_rows,
)


class ProductReasoningEvalTests(unittest.TestCase):
    def test_extracts_explicit_and_boxed_answers(self) -> None:
        self.assertEqual(extract_gsm8k("Work. Final answer: 1,234."), "1234")
        self.assertEqual(extract_boxed(r"Thus the result is \boxed{\frac{3}{4}}."), r"\frac{3}{4}")
        self.assertEqual(gold_gsm8k({"answer": "work\n#### -42"}), "-42")

    def test_math_normalizes_fraction_command(self) -> None:
        self.assertTrue(match_math(r"\dfrac{1}{2}", r"\frac{1}{2}"))

    def test_subset_is_stable_and_unique(self) -> None:
        rows = [{"question": f"q{index}", "answer": "#### 1"} for index in range(10)]
        first = select_rows("gsm8k", rows, 4, 31)
        second = select_rows("gsm8k", list(reversed(rows)), 4, 31)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)

    def test_subset_rejects_oversized_request(self) -> None:
        with self.assertRaises(ProductEvalError):
            select_rows("gsm8k", [{"question": "q"}], 2, 31)


if __name__ == "__main__":
    unittest.main()
