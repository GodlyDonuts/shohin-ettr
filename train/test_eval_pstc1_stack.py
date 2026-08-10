import unittest

from eval_pstc1_stack import execute_symbolic
from pushdown_stack_typed_compiler import ACTION_TO_ID, load_stack_program
from test_pushdown_stack_typed_compiler import row


class PSTC1EvaluationTest(unittest.TestCase):
    def test_gold_symbolic_execution(self):
        program = load_stack_program(row())
        actions = [item.action for item in program.actions] + [ACTION_TO_ID["STOP"]] * 15
        pointers = [max(0, item.source_index) for item in program.actions] + [0] * 15
        tree, valid, length, invalid = execute_symbolic(actions[:22], pointers[:22], program)
        self.assertTrue(valid)
        self.assertEqual(length, len(program.actions))
        self.assertEqual(invalid, 0)
        self.assertEqual(tree[0], "APPLY_MUL")


if __name__ == "__main__":
    unittest.main()
