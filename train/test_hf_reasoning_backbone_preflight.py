"""Small non-network tests for the external-backbone preflight."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from hf_reasoning_backbone_preflight import _atomic_json, _package_version


class BackbonePreflightTests(unittest.TestCase):
    def test_atomic_json_replaces_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            _atomic_json(output, {"status": "pass", "count": 3})
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"status": "pass", "count": 3},
            )
            self.assertFalse(output.with_suffix(".json.tmp").exists())

    def test_package_version_handles_missing_distribution(self) -> None:
        self.assertIsNone(_package_version("definitely-not-a-real-package-shohin"))


if __name__ == "__main__":
    unittest.main()
