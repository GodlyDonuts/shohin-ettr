import argparse
import json
from pathlib import Path
import tempfile
import unittest

from compare_ocet1 import EVAL_SCHEMA, run


class OCET1ComparisonTest(unittest.TestCase):
    def test_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {arm: [] for arm in ("aligned", "swapped", "hidden")}
            for arm in paths:
                for shard in range(8):
                    rows = []
                    for index in range(shard, 954, 8):
                        for member in ("clean", "fault"):
                            correct = arm == "aligned"
                            rows.append(
                                {
                                    "identity_sha256": f"{index:04d}-{member}",
                                    "pair_identity_sha256": f"{index:04d}",
                                    "pair_member": member,
                                    "corruption_family": "choice_final" if index < 128 else "numeric_final",
                                    "proposal_correct": correct,
                                    "commit_valid": True,
                                    "final_correct": correct,
                                    "commit_error": None,
                                    "max_token_exhausted": False,
                                    "commit_action": "<KEEP>",
                                }
                            )
                    report = {
                        "schema": EVAL_SCHEMA,
                        "status": "complete",
                        "holdout_used": False,
                        "arm": arm,
                        "shard_index": shard,
                        "shard_count": 8,
                        "checkpoint_sha256": arm,
                        "data_sha256": "data",
                        "data_report_sha256": "report",
                        "row_count": len(rows),
                        "results": rows,
                    }
                    path = root / f"{arm}_{shard}.json"
                    path.write_text(json.dumps(report))
                    paths[arm].append(path)
            result = run(argparse.Namespace(**paths, output=root / "comparison.json"))
            self.assertEqual(result["status"], "pass")
            self.assertTrue(result["holdout_authorized"])


if __name__ == "__main__":
    unittest.main()
