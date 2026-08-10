import unittest

import torch

from monotonic_lexical_compiler import (
    MAX_CANDIDATES,
    MonotonicLexicalCompiler,
    ROLE_TO_ID,
    lexical_labels,
    lexical_loss,
    load_lexical_program,
)


def row():
    return {
        "schema": "shohin-mltc1-lexical-supervision-v1",
        "identity_sha256": "a" * 64,
        "family": "basic_arithmetic",
        "question": "Calculate 2 + 3.",
        "number_spans": [
            {"start": 10, "end": 11, "surface": "2", "magnitude": "2"},
            {"start": 14, "end": 15, "surface": "3", "magnitude": "3"},
        ],
        "gold_actions": [],
        "candidates": [
            {"start": 10, "end": 11, "surface": "2", "surface_type": "NUMBER", "role": "NUMBER", "source_index": 0},
            {"start": 12, "end": 13, "surface": "+", "surface_type": "PLUS", "role": "ADD", "source_index": -1},
            {"start": 14, "end": 15, "surface": "3", "surface_type": "NUMBER", "role": "NUMBER", "source_index": 1},
        ],
    }


class MonotonicLexicalCompilerTest(unittest.TestCase):
    def test_forward_and_surface_mask(self):
        program = load_lexical_program(row())
        labels = lexical_labels([program], torch.device("cpu"))
        model = MonotonicLexicalCompiler(16, width=32, encoder_layers=1, heads=4)
        source = torch.randn(1, 5, 16)
        mask = torch.zeros(1, MAX_CANDIDATES, 5, dtype=torch.bool)
        mask[0, 0, 0] = mask[0, 1, 2] = mask[0, 2, 4] = True
        output = model(source, mask, labels["surface"], labels["candidate_count"])
        self.assertEqual(tuple(output.role_logits.shape), (1, MAX_CANDIDATES, 9))
        self.assertTrue(torch.isfinite(lexical_loss(output, labels)))
        self.assertIn(int(output.chosen_roles[0, 0]), {ROLE_TO_ID["IGNORE"], ROLE_TO_ID["NUMBER"]})


if __name__ == "__main__":
    unittest.main()
