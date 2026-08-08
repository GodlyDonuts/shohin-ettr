"""Tests for the frozen TTR1 development comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

from compare_ttr1_development import CONTROLS, compare


def metrics(correct: int, math: int, logic: int, code: int) -> dict:
    return {
        "overall": {"generated_correct": correct, "total": 1289},
        "math500": {"generated_correct": math, "total": 480},
        "bbh_logic": {"generated_correct": logic, "total": 709},
        "mbpp": {"generated_correct": code, "total": 100},
    }


class TTR1ComparisonTests(unittest.TestCase):
    def test_conjunctive_gate_rejects_domain_regression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = {
                "status": "complete",
                "split": "development",
                "model_revision": "revision",
                "data_sha256": "d" * 64,
                "data_report_sha256": "r" * 64,
                "full_row_count": 1289,
                "shard_count": 1,
            }
            treatment = root / "treatment.json"
            treatment.write_text(
                json.dumps(
                    {
                        **shared,
                        "schema": "shohin-idr1-revision-evaluation-v1",
                        "metrics": metrics(400, 100, 295, 5),
                    }
                )
            )
            control_paths = []
            for name in CONTROLS:
                path = root / f"{name}.json"
                path.write_text(
                    json.dumps(
                        {
                            **shared,
                            "schema": "shohin-ttr1-control-evaluation-v1",
                            "control": name,
                            "metrics": metrics(300, 90, 200, 10),
                        }
                    )
                )
                control_paths.append(path)
            result = compare(
                argparse.Namespace(
                    treatment=treatment,
                    control_report=control_paths,
                    output=root / "comparison.json",
                )
            )
            self.assertFalse(result["gate_pass"])
            self.assertEqual(
                result["domain_correct_count_deltas_vs_unchanged"]["mbpp"], -5
            )
            self.assertFalse(result["holdout_authorized"])

    def test_accepts_complete_merged_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = {
                "status": "complete",
                "split": "development",
                "model_revision": "revision",
                "data_sha256": "d" * 64,
                "data_report_sha256": "r" * 64,
                "full_row_count": 1289,
                "shard_count": 1,
            }
            treatment = root / "treatment.json"
            treatment.write_text(
                json.dumps(
                    {
                        **shared,
                        "schema": "shohin-idr1-revision-evaluation-v1",
                        "metrics": metrics(500, 100, 390, 10),
                    }
                )
            )
            control_paths = []
            for name in CONTROLS:
                path = root / f"{name}.json"
                path.write_text(
                    json.dumps(
                        {
                            **shared,
                            "schema": "shohin-ttr1-control-evaluation-v1",
                            "control": name,
                            "merged_from_shards": True,
                            "shard_count": 8,
                            "metrics": metrics(400, 90, 301, 9),
                        }
                    )
                )
                control_paths.append(path)
            result = compare(
                argparse.Namespace(
                    treatment=treatment,
                    control_report=control_paths,
                    output=root / "comparison.json",
                )
            )
            self.assertTrue(result["gate_pass"])


if __name__ == "__main__":
    unittest.main()
