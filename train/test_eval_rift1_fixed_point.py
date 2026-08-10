import unittest

from eval_rift1_fixed_point import execute_commit, replace_draft


class RIFT1MechanicsTest(unittest.TestCase):
    def test_replace_draft_once(self):
        question = "Draft:\nold\n\nInstruction"
        self.assertEqual(replace_draft(question, "old", "new"), "Draft:\nnew\n\nInstruction")

    def test_keep_fixed_point(self):
        final, action, error = execute_commit("answer C", "<KEEP>\n")
        self.assertEqual(final, "answer C")
        self.assertEqual(action, "<KEEP>")
        self.assertIsNone(error)

    def test_second_repair(self):
        final, action, error = execute_commit(
            "The answer is B.", "<REPLACE_LAST>\nB\nC\n"
        )
        self.assertEqual(final, "The answer is C.")
        self.assertEqual(action, "<REPLACE_LAST>")
        self.assertIsNone(error)

    def test_malformed_commit_fails_closed(self):
        final, action, error = execute_commit("answer C", "not a script")
        self.assertEqual(final, "answer C")
        self.assertIsNone(action)
        self.assertIsNotNone(error)


if __name__ == "__main__":
    unittest.main()
