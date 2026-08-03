"""Non-network tests for the Hugging Face product evaluator."""

from __future__ import annotations

import unittest
from unittest.mock import patch
import types

from hf_product_reasoning_eval import (
    _completion_usage,
    _generation_stop_token_ids,
    _finalization_question,
    ProductEvalError,
    _bounded_program_result,
    _humaneval_program,
    _mbpp_program,
    _strip_reasoning_and_fences,
    _task_prompt,
    extract_aime,
    extract_boxed,
    extract_gsm8k,
    extract_short_answer,
    gold_numeric_answer,
    gold_gsm8k,
    has_explicit_final_answer,
    match_aime,
    match_math,
    match_gsm8k,
    match_short_answer,
    select_rows,
)


class ProductReasoningEvalTests(unittest.TestCase):
    def test_completion_usage_distinguishes_eos_from_exhaustion(self) -> None:
        self.assertEqual(_completion_usage([4, 5, 2, 2], [2, 9], 4), (3, False))
        self.assertEqual(_completion_usage([4, 9, 2, 2], [2, 9], 4), (2, False))
        self.assertEqual(_completion_usage([4, 5, 6, 7], [2, 9], 4), (4, True))

    def test_generation_stops_at_eos_and_new_chat_turn(self) -> None:
        tokenizer = types.SimpleNamespace(
            eos_token_id=2,
            unk_token_id=0,
            all_special_tokens=["<|im_start|>", "<|im_end|>"],
            convert_tokens_to_ids=lambda token: {"<|im_start|>": 9}.get(token, 0),
            convert_ids_to_tokens=lambda token_id: {9: "<|im_start|>"}.get(
                token_id, "<unk>"
            ),
        )
        self.assertEqual(_generation_stop_token_ids(tokenizer), [2, 9])

    def test_extracts_explicit_and_boxed_answers(self) -> None:
        self.assertEqual(extract_gsm8k("Work. Final answer: 1,234."), "1234")
        self.assertEqual(extract_boxed(r"Thus the result is \boxed{\frac{3}{4}}."), r"\frac{3}{4}")
        self.assertEqual(gold_gsm8k({"answer": "work\n#### -42"}), "-42")
        self.assertEqual(gold_numeric_answer({"answer": "204"}), "204")

    def test_aime_requires_an_explicit_final_integer(self) -> None:
        self.assertEqual(extract_aime(r"Work. Therefore \boxed{025}."), "025")
        self.assertEqual(extract_aime("Work. The answer is 25."), "25")
        self.assertIsNone(extract_aime(r"Work. The answer is 25^{9/5}."))
        self.assertIsNone(extract_aime("Intermediate value 25, still working."))
        self.assertTrue(match_aime("25", "025"))
        self.assertFalse(match_aime(r"25^{9/5}", "025"))

    def test_explicit_final_answer_markers(self) -> None:
        self.assertTrue(has_explicit_final_answer(r"Therefore \boxed{4}."))
        self.assertTrue(has_explicit_final_answer("The final answer: B"))
        self.assertFalse(
            has_explicit_final_answer("A capped partial thought mentions option B")
        )

    def test_finalization_question_preserves_problem_and_draft(self) -> None:
        rendered = _finalization_question("What is 6 times 7?", "6*7 = 42")
        self.assertIn("What is 6 times 7?", rendered)
        self.assertIn("6*7 = 42", rendered)
        self.assertIn(r"\boxed{}", rendered)
        self.assertIn("Do not redo or extend", rendered)

    def test_gsm8k_normalizes_numeric_equivalence_and_currency_phrases(self) -> None:
        transcript = "Work. The answer is 1 dollar 40 cents."
        self.assertEqual(extract_gsm8k(transcript), "1.4")
        self.assertTrue(match_gsm8k("2.00", "2"))
        self.assertTrue(match_gsm8k("3/4", "0.75"))
        self.assertFalse(match_gsm8k(extract_gsm8k(transcript), "1"))

    def test_later_empty_box_instruction_does_not_hide_answer(self) -> None:
        transcript = r"Therefore \boxed{7}. Remember to use \boxed{}."
        self.assertEqual(extract_gsm8k(transcript), "7")
        self.assertEqual(extract_boxed(transcript), "7")

    def test_math_normalizes_fraction_command(self) -> None:
        self.assertTrue(match_math(r"\dfrac{1}{2}", r"\frac{1}{2}"))

    def test_math_verify_backend_handles_equivalent_latex(self) -> None:
        fake = types.SimpleNamespace(
            LatexExtractionConfig=lambda: object(),
            parse=lambda value, extraction_config: [value.replace("{", "").replace("}", "")],
            verify=lambda gold, prediction: gold == prediction,
        )
        with patch.dict("sys.modules", {"math_verify": fake}):
            self.assertTrue(match_math(r"\frac9{19}", r"\frac{9}{19}"))

    def test_short_answer_scoring_handles_bbh_labels(self) -> None:
        self.assertEqual(extract_short_answer(r"Therefore \boxed{(B)}."), "(B)")
        self.assertTrue(match_short_answer("(b)", "(B)"))
        self.assertTrue(match_short_answer("TRUE.", "True"))
        self.assertTrue(match_short_answer(r"\text{False}", "False"))
        self.assertTrue(match_short_answer(r"\mathrm{D}", "(D)"))

    def test_gpqa_prompt_contains_only_labeled_choices(self) -> None:
        row = {
            "question": "Which option follows?",
            "choices": [
                {"label": "A", "text": "alpha"},
                {"label": "B", "text": "beta"},
                {"label": "C", "text": "gamma"},
                {"label": "D", "text": "delta"},
            ],
            "answer": "C",
        }
        prompt = _task_prompt("gpqa", row)
        alternate = _task_prompt("gpqa", {**row, "answer": "A"})
        self.assertIn("(C) gamma", prompt)
        self.assertEqual(prompt, alternate)

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
