import unittest

from build_ocet1_on_policy_data import derive_row


class OCET1DerivationTest(unittest.TestCase):
    def setUp(self):
        draft = "Reasoning. The answer is B."
        self.row = {
            "identity_sha256": "identity",
            "question": f"Draft:\n{draft}\n\nInstruction",
            "draft": draft,
            "final_response": "Reasoning. The answer is C.",
            "script": "<REPLACE_LAST>\nB\nC\n",
            "gold_answer": "C",
        }

    def test_wrong_proposal_becomes_repair(self):
        proposal = {
            "executed_trajectory": "Reasoning. The answer is D.",
            "new_surface": "D",
            "completion": "proposal",
        }
        row, mode = derive_row(self.row, proposal)
        self.assertEqual(mode, "on_policy")
        self.assertEqual(row["script"], "<REPLACE_LAST>\nD\nC\n")

    def test_correct_proposal_becomes_keep(self):
        proposal = {
            "executed_trajectory": self.row["final_response"],
            "new_surface": "C",
            "completion": "proposal",
        }
        row, mode = derive_row(self.row, proposal)
        self.assertEqual(mode, "on_policy")
        self.assertEqual(row["script"], "<KEEP>\n")


if __name__ == "__main__":
    unittest.main()
