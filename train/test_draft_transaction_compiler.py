from fractions import Fraction
import unittest

import torch

from draft_transaction_compiler import (
    DraftTransactionError,
    compile_draft_transactions,
    normalize_expression,
    reset_state_reads,
)
from learned_arithmetic_microcode import LearnedDigitMicrocode
from train_lam1_microcode import candidate_fraction
from typed_microcode_graph import execute_fraction, execute_learned


class DraftTransactionCompilerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.microcode = LearnedDigitMicrocode()
        with torch.no_grad():
            self.microcode.add_logits.fill_(-100)
            self.microcode.sub_logits.fill_(-100)
            self.microcode.mul_logits.fill_(-100)
            for a in range(10):
                for b in range(10):
                    for carry in range(2):
                        total = a + b + carry
                        self.microcode.add_logits[a, b, carry, total].fill_(100)
                        difference = a - b - carry
                        target = difference % 10 + 10 * int(difference < 0)
                        self.microcode.sub_logits[a, b, carry, target].fill_(100)
                    for carry in range(10):
                        total = a * b + carry
                        self.microcode.mul_logits[a, b, carry, total].fill_(100)
        self.microcode.freeze_discrete()

    def test_claims_only_link_computed_states(self) -> None:
        graph, receipt = compile_draft_transactions(
            "A worker makes 3 items twice weekly.",
            "First <<3*2=7>> then annually <<7*52=364>>.",
        )
        self.assertEqual(execute_fraction(graph), Fraction(312))
        self.assertEqual(candidate_fraction(execute_learned(self.microcode, graph)), 312)
        self.assertEqual(receipt.accepted, 2)
        self.assertEqual(receipt.state_reads, 1)
        self.assertEqual(execute_fraction(reset_state_reads(graph)), 0)

    def test_source_precedes_literal_and_percent_is_exact(self) -> None:
        graph, receipt = compile_draft_transactions(
            "The price is $1,200 and the discount is 25%.",
            "The discount is <<$1,200*25%=300>>.",
        )
        self.assertEqual(normalize_expression("$1,200*25%"), "1200*0.25")
        self.assertEqual(execute_fraction(graph), 300)
        self.assertEqual(receipt.source_reads, 2)

    def test_unsupported_annotation_rejects_without_repair(self) -> None:
        graph, receipt = compile_draft_transactions(
            "There are 3 and 2 items.",
            "Bad <<pow(3,2)=9>> then good <<3+2=5>>.",
        )
        self.assertEqual(execute_fraction(graph), 5)
        self.assertEqual(receipt.accepted, 1)
        self.assertEqual(receipt.rejected, ("expression AST differs",))

    def test_no_accepted_transaction_fails_closed(self) -> None:
        with self.assertRaisesRegex(DraftTransactionError, "no accepted"):
            compile_draft_transactions("There are 3 items.", "Therefore 3.")


if __name__ == "__main__":
    unittest.main()
