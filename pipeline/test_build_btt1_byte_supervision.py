import unittest

from build_btt1_byte_supervision import compile_byte_roles, execute_byte_roles


class BTT1SupervisionTest(unittest.TestCase):
    def test_raw_byte_round_trip(self):
        question = "Calculate -(2 + 3) * 4."
        candidates = [
            {"role": "NEGATE", "start": 10, "end": 11},
            {"role": "LPAREN", "start": 11, "end": 12},
            {"role": "NUMBER", "start": 12, "end": 13},
            {"role": "ADD", "start": 14, "end": 15},
            {"role": "NUMBER", "start": 16, "end": 17},
            {"role": "RPAREN", "start": 17, "end": 18},
            {"role": "MUL", "start": 19, "end": 20},
            {"role": "NUMBER", "start": 21, "end": 22},
        ]
        roles = compile_byte_roles(question, candidates)
        actions, valid = execute_byte_roles(question, roles)
        self.assertTrue(valid)
        self.assertEqual(
            [action["action"] for action in actions],
            ["PUSH", "PUSH", "APPLY_ADD", "NEGATE", "PUSH", "APPLY_MUL", "STOP"],
        )


if __name__ == "__main__":
    unittest.main()
