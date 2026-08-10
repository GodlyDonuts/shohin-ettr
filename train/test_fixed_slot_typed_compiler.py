import unittest

import torch

from fixed_slot_typed_compiler import (
    DIGIT_PAD,
    MAX_SOURCE_NUMBERS,
    FixedSlotSkeletonCompiler,
    compile_typed_program,
    decode_fraction,
    encode_fraction,
    skeleton_labels,
    skeleton_loss,
)
from fractions import Fraction


def row():
    return {
        "identity_sha256": "a" * 64,
        "question": "Calculate -(2 + 3) * 4.",
        "records": [
            {
                "operation": "ADD",
                "operands": [
                    {"numerator": 2, "denominator": 1},
                    {"numerator": 3, "denominator": 1},
                ],
                "result": {"numerator": 5, "denominator": 1},
                "dependencies": [],
            },
            {
                "operation": "MUL",
                "operands": [
                    {"numerator": -5, "denominator": 1},
                    {"numerator": 4, "denominator": 1},
                ],
                "result": {"numerator": -20, "denominator": 1},
                "dependencies": [],
            },
        ],
    }


class FixedSlotCompilerTest(unittest.TestCase):
    def test_fraction_round_trip(self):
        for value in (Fraction(0), Fraction(-17, 5), Fraction(123456789, 1000)):
            state = encode_fraction(value)
            self.assertEqual(decode_fraction(state), value)
            self.assertEqual(len(state.numerator), 23)
            self.assertEqual(len(state.denominator), 11)
        self.assertEqual(encode_fraction(Fraction(1)).numerator[1], DIGIT_PAD)

    def test_typed_program_recovers_negated_dependency(self):
        program = compile_typed_program(row())
        self.assertEqual(len(program.number_spans), 3)
        self.assertEqual(program.slots[1].left.index, 0)
        self.assertEqual(program.slots[1].left.polarity, 1)

    def test_skeleton_forward_and_loss(self):
        torch.manual_seed(7)
        program = compile_typed_program(row())
        labels = skeleton_labels([program], torch.device("cpu"))
        model = FixedSlotSkeletonCompiler(
            32, width=64, encoder_layers=1, heads=4
        )
        source = torch.randn(1, 12, 32)
        source_mask = torch.ones(1, 12, dtype=torch.bool)
        candidates = torch.zeros(1, MAX_SOURCE_NUMBERS, 12, dtype=torch.bool)
        candidates[0, 0, 1:2] = True
        candidates[0, 1, 3:4] = True
        candidates[0, 2, 7:8] = True
        output = model(
            source,
            source_mask,
            candidates,
            labels["candidate_count"],
            gold=labels,
            feedback="gold",
        )
        loss, components = skeleton_loss(output, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(output.operation_logits.shape, (1, 5, 4))
        self.assertEqual(
            output.left_reference_logits.shape,
            (1, 5, MAX_SOURCE_NUMBERS + 5),
        )
        self.assertEqual(set(components), {
            "active", "operation", "left_reference", "right_reference",
            "left_polarity", "right_polarity",
        })


if __name__ == "__main__":
    unittest.main()
