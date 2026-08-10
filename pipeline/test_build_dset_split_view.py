import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from build_dset_split_view import REPORT_SCHEMA, run


class DSETSplitViewTest(unittest.TestCase):
    def test_train_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "train.jsonl"
            data.write_text("{}\n")
            digest = hashlib.sha256(data.read_bytes()).hexdigest()
            report = root / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "schema": REPORT_SCHEMA,
                        "status": "complete",
                        "holdout_used": False,
                        "outputs": {
                            "train": {
                                "path": str(data.resolve()),
                                "sha256": digest,
                                "rows": 1,
                                "sources": 1,
                            }
                        },
                    }
                )
            )
            output = root / "view.json"
            view = run(argparse.Namespace(report=report, split="train", output=output))
            self.assertEqual(view["outputs"]["diagnostic"]["sha256"], digest)
            self.assertFalse(view["split_view"]["holdout_used"])


if __name__ == "__main__":
    unittest.main()
