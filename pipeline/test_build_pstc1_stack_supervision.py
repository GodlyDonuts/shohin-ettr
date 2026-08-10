import unittest

from build_pstc1_stack_supervision import compile_row, execute_actions


class PSTC1SupervisionTest(unittest.TestCase):
    def test_nested_unary_expression(self):
        row = {
            "identity_sha256": "a" * 64,
            "source_question_sha256": "b" * 64,
            "split": "train",
            "family": "basic_arithmetic",
            "question": "Calculate -( 2 + 3 ) * 4.",
            "records": [
                {
                    "operation": "ADD",
                    "result": {"numerator": 5, "denominator": 1},
                },
                {
                    "operation": "MUL",
                    "result": {"numerator": -20, "denominator": 1},
                },
            ],
            "terminal_value": {"numerator": -20, "denominator": 1},
        }
        compiled = compile_row(row)
        self.assertEqual(
            [action["action"] for action in compiled["actions"]],
            ["PUSH", "PUSH", "APPLY_ADD", "NEGATE", "PUSH", "APPLY_MUL", "STOP"],
        )
        records, terminal, maximum = execute_actions(
            compiled["actions"], compiled["number_spans"]
        )
        self.assertEqual(str(terminal), "-20")
        self.assertEqual(maximum, 2)
        self.assertEqual([record["operation"] for record in records], ["ADD", "MUL"])


if __name__ == "__main__":
    unittest.main()
