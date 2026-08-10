import unittest

import torch

from eval_fstc1_skeleton import evaluate_batch, source_shuffle_indices
from fixed_slot_typed_compiler import compile_typed_program


def make_row(identity, left, right):
    return {
        "identity_sha256": identity * 64,
        "family": "basic_arithmetic",
        "question": f"Calculate {left} + {right}.",
        "records": [
            {
                "operation": "ADD",
                "operands": [
                    {"numerator": left, "denominator": 1},
                    {"numerator": right, "denominator": 1},
                ],
                "result": {"numerator": left + right, "denominator": 1},
                "dependencies": [],
            }
        ],
    }


class Output:
    active_logits = torch.tensor([[[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]])
    operation_logits = torch.zeros(1, 5, 4)
    left_reference_logits = torch.zeros(1, 5, 12)
    right_reference_logits = torch.zeros(1, 5, 12)
    left_polarity_logits = torch.zeros(1, 5, 2)
    right_polarity_logits = torch.zeros(1, 5, 2)


class FSTC1EvaluationTest(unittest.TestCase):
    def test_exact_program_scores_complete(self):
        program = compile_typed_program(make_row("a", 2, 3))
        output = Output()
        output.right_reference_logits[0, 0, 1] = 1.0
        detail = evaluate_batch(output, [program], [program])[0]
        self.assertTrue(detail["complete_skeleton_exact"])
        self.assertTrue(detail["reference_kind_exact"])

    def test_shuffle_has_no_fixed_points(self):
        rows = [(make_row(str(index), index + 1, index + 11), None) for index in range(4)]
        rows = [(row, compile_typed_program(row)) for row, _ in rows]
        mapping = source_shuffle_indices(rows, 7)
        self.assertTrue(all(index != source for index, source in enumerate(mapping)))


if __name__ == "__main__":
    unittest.main()
