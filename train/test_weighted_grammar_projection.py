import unittest

import torch

from byte_tape_compiler import ROLE_TO_ID
from weighted_grammar_projection import project_role_logits


class WeightedGrammarProjectionTest(unittest.TestCase):
    def test_repairs_unmatched_parenthesis_top1(self):
        source = list(b"(2+3)")
        gold = ["LPAREN", "NUM_BEGIN", "ADD", "NUM_BEGIN", "RPAREN"]
        logits = torch.full((len(source), len(ROLE_TO_ID)), -8.0)
        for position, role in enumerate(gold):
            logits[position, ROLE_TO_ID[role]] = 8.0
        logits[-1, ROLE_TO_ID["IGNORE"]] = 9.0
        projected = project_role_logits(logits, source, beam_width=16)
        self.assertEqual(projected, [ROLE_TO_ID[role] for role in gold])


if __name__ == "__main__":
    unittest.main()
