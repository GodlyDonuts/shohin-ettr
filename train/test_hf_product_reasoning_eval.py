"""Non-network tests for the Hugging Face product evaluator."""

from __future__ import annotations

import unittest

from hf_product_reasoning_eval import (
    ProductEvalError,
    _bounded_program_result,
    _humaneval_program,
    _mbpp_program,
    _strip_reasoning_and_fences,
    _task_prompt,
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

    def test_later_empty_box_instruction_does_not_hide_answer(self) -> None:
        transcript = r"Therefore \boxed{7}. Remember to use \boxed{}."
        self.assertEqual(extract_gsm8k(transcript), "7")
        self.assertEqual(extract_boxed(transcript), "7")

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

    def test_code_prompts_and_fences_are_normalized(self) -> None:
        row = {
            "prompt": "def add(a, b):\n",
            "test": "def check(fn): assert fn(2, 3) == 5",
            "entry_point": "add",
        }
        completion = "<think>simple</think>\n```python\ndef add(a, b):\n    return a + b\n```"
        self.assertIn(
            "complete the python function", _task_prompt("humaneval", row).lower()
        )
        self.assertEqual(
            _strip_reasoning_and_fences(completion),
            "def add(a, b):\n    return a + b",
        )
        program = _humaneval_program(row, completion)
        self.assertTrue(_bounded_program_result(program, 2.0)["passed"])

    def test_mbpp_program_executes_official_tests(self) -> None:
        row = {
            "text": "add two integers",
            "test_list": ["assert add(4, 7) == 11"],
            "test_setup_code": "",
        }
        self.assertIn("passes every test", _task_prompt("mbpp", row))
        program = _mbpp_program(row, "def add(a, b):\n    return a + b")
        self.assertTrue(_bounded_program_result(program, 2.0)["passed"])


if __name__ == "__main__":
    unittest.main()
