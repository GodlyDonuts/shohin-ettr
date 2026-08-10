import argparse
import json
from pathlib import Path
import tempfile
import unittest

from compare_fret1 import EVAL_SCHEMA, run


class FRET1ComparisonTest(unittest.TestCase):
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
                                    "pointer_exact": aligned,
                                    "replacement_exact": aligned,
                                    "program_exact": aligned,
                                    "execution_correct": aligned,
                                    "execution_error": None,
                                    "max_token_exhausted": False,
                                    "copy_characters": 99,
                                    "draft_characters": 100,
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
            self.assertEqual(result["aligned"]["execution_correct"], 1908)
            self.assertEqual(result["hidden"]["execution_correct"], 0)


if __name__ == "__main__":
    unittest.main()
