#!/usr/bin/env python3

import unittest

from eval_dense_public_livecodebench import extract_fenced_code, task_passed


class LiveCodeBenchAdapterTests(unittest.TestCase):
    def test_extracts_last_fenced_block(self):
        self.assertEqual(
            extract_fenced_code("thinking\n```python\nprint(1)\n```\n```\nprint(2)\n```"),
            "print(2)",
        )

    def test_plain_completion_is_retained(self):
        self.assertEqual(extract_fenced_code("print(1)\n"), "print(1)")

    def test_pass_requires_nonempty_literal_true_tests(self):
        self.assertTrue(task_passed([True, True]))
        self.assertFalse(task_passed([]))
        self.assertFalse(task_passed([True, False]))
        self.assertFalse(task_passed([True, 1]))


if __name__ == "__main__":
    unittest.main()
