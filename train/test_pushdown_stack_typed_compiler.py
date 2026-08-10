import unittest

import torch

from pushdown_stack_typed_compiler import (
    MAX_SOURCE_NUMBERS,
    PushdownStackCompiler,
    load_stack_program,
    stack_labels,
    stack_loss,
)


def row():
    return {
        "identity_sha256": "a" * 64,
        "family": "basic_arithmetic",
        "question": "Calculate -(2 + 3) * 4.",
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
        "maximum_stack": 2,
    }


class PushdownStackCompilerTest(unittest.TestCase):
    def test_program_and_forward(self):
        torch.manual_seed(3)
        program = load_stack_program(row())
        labels = stack_labels([program], torch.device("cpu"))
        model = PushdownStackCompiler(32, width=64, encoder_layers=1, heads=4)
        source = torch.randn(1, 24, 32)
        source_mask = torch.ones(1, 24, dtype=torch.bool)
        candidates = torch.zeros(1, MAX_SOURCE_NUMBERS, 24, dtype=torch.bool)
        candidates[0, 0, 12] = True
        candidates[0, 1, 16] = True
        candidates[0, 2, 21] = True
        output = model(
            source,
            source_mask,
            candidates,
            labels["candidate_count"],
            gold=labels,
            feedback="gold",
        )
        loss, components = stack_loss(output, labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(output.action_logits.shape, (1, 22, 7))
        self.assertEqual(output.pointer_logits.shape, (1, 22, 7))
        self.assertEqual(int(output.invalid_action_count), 0)
        self.assertEqual(set(components), {"action", "pointer"})


if __name__ == "__main__":
    unittest.main()
