import unittest

from eval_fret1_always_rewrite import score_rewrite


class FRET1ScoreTest(unittest.TestCase):
    def setUp(self):
        self.row = {
            "draft": "Work. The answer is B.",
            "final_response": "Work. The answer is C.",
            "changed_character_span": [20, 21],
            "gold_answer": "C",
        }

    def test_exact_rewrite(self):
        score = score_rewrite(self.row, "<REPLACE_LAST>\nB\nC\n")
        self.assertTrue(score["program_exact"])
        self.assertTrue(score["execution_correct"])
        self.assertEqual(score["copy_characters"], len(self.row["draft"]) - 1)

    def test_wrong_pointer_fails_program_and_execution(self):
        score = score_rewrite(self.row, "<REPLACE_LAST>\nanswer\nC\n")
        self.assertFalse(score["pointer_exact"])
        self.assertFalse(score["execution_correct"])

    def test_idempotent_clean_rewrite(self):
        row = dict(self.row)
        row["draft"] = row["final_response"]
        score = score_rewrite(row, "<REPLACE_LAST>\nC\nC\n")
        self.assertTrue(score["program_exact"])
        self.assertTrue(score["execution_correct"])


if __name__ == "__main__":
    unittest.main()
