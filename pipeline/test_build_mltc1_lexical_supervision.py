import unittest

from build_mltc1_lexical_supervision import compile_row


class MLTC1SupervisionTest(unittest.TestCase):
    def test_nested_unary_precedence(self):
        question = "Calculate -(2 + 3) * 4."
        row = {
            "identity_sha256": "a" * 64,
            "source_question_sha256": "b" * 64,
            "split": "development",
            "family": "basic_arithmetic",
            "question": question,
            "number_spans": [
                {"start": 12, "end": 13, "surface": "2", "magnitude": "2"},
                {"start": 16, "end": 17, "surface": "3", "magnitude": "3"},
                {"start": 21, "end": 22, "surface": "4", "magnitude": "4"},
            ],
            "actions": [
                {"action": "PUSH", "source_index": 0},
                {"action": "PUSH", "source_index": 1},
                {"action": "APPLY_ADD"},
                {"action": "NEGATE"},
                {"action": "PUSH", "source_index": 2},
                {"action": "APPLY_MUL"},
                {"action": "STOP"},
            ],
        }
        compiled = compile_row(row)
        selected = [candidate["role"] for candidate in compiled["candidates"] if candidate["role"] != "IGNORE"]
        self.assertEqual(
            selected,
            ["NEGATE", "LPAREN", "NUMBER", "ADD", "NUMBER", "RPAREN", "MUL", "NUMBER"],
        )


if __name__ == "__main__":
    unittest.main()
