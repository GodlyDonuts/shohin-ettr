import argparse
import json
from pathlib import Path
import tempfile
import unittest

from compare_rift1 import EVAL_SCHEMA, run


class RIFT1ComparisonTest(unittest.TestCase):
    def test_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {"aligned": [], "hidden": []}
            for arm in paths:
                for shard in range(8):
                    rows = []
                    for index in range(shard, 954, 8):
                        for member in ("clean", "fault"):
                            aligned = arm == "aligned"
                            rows.append(
                                {
                                    "identity_sha256": f"{index:04d}-{member}",
                                    "pair_identity_sha256": f"{index:04d}",
                                    "pair_member": member,
                                    "corruption_family": "choice_final" if index < 128 else "numeric_final",
                                    "proposal_correct": aligned,
                                    "commit_valid": True,
                                    "final_correct": aligned,
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
            output = root / "comparison.json"
            result = run(argparse.Namespace(aligned=paths["aligned"], hidden=paths["hidden"], output=output))
            self.assertEqual(result["status"], "pass")
            self.assertTrue(result["holdout_authorized"])


if __name__ == "__main__":
    unittest.main()
