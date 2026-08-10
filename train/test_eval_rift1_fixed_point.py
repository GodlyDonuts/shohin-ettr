import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from eval_rift1_fixed_point import ISET_SCHEMA, execute_commit, load_iset, replace_draft


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

    def test_iset_loader_binds_data_and_unique_trajectories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.jsonl"
            data_report = root / "report.json"
            data.write_text("data\n")
            data_report.write_text("{}\n")
            sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            results = [
                {
                    "identity_sha256": f"{index + 2000:064x}",
                    "source_dseo1_identity_sha256": f"{index:064x}",
                    "executed_trajectory": "answer",
                }
                for index in range(1908)
            ]
            merged = root / "merged.json"
            merged.write_text(
                json.dumps(
                    {
                        "schema": ISET_SCHEMA,
                        "status": "complete",
                        "holdout_used": False,
                        "arm": "aligned",
                        "shard_count": 8,
                        "row_count": 1908,
                        "data_sha256": sha(data),
                        "data_report_sha256": sha(data_report),
                        "results": results,
                    }
                )
            )
            rows, receipts = load_iset(merged, sha(data), sha(data_report))
            self.assertEqual(len(rows), 1908)
            self.assertEqual(receipts[0]["sha256"], sha(merged))

            payload = json.loads(merged.read_text())
            payload["data_sha256"] = "0" * 64
            merged.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "ISET report differs"):
                load_iset(merged, sha(data), sha(data_report))


if __name__ == "__main__":
    unittest.main()
