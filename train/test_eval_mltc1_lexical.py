import unittest

from eval_mltc1_lexical import execute, flat_compile_selected, normalized_actions
from monotonic_lexical_compiler import load_lexical_program
from test_monotonic_lexical_compiler import row


class MLTC1EvaluationTest(unittest.TestCase):
    def test_flat_executor_is_valid_for_simple_sum(self):
        program = load_lexical_program(row())
        candidates = [
            {"role": "NUMBER", "source_index": 0},
            {"role": "ADD", "source_index": -1},
            {"role": "NUMBER", "source_index": 1},
        ]
        compiled, valid = flat_compile_selected(candidates)
        actions = normalized_actions(compiled, candidates)
        _, execution_valid = execute(actions, program)
        self.assertTrue(valid)
        self.assertTrue(execution_valid)


if __name__ == "__main__":
    unittest.main()
