"""Tests for the frozen TTR1 holdout comparison."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from pipeline.compare_ttr1_holdout import (
    CONTROLS,
    TTR1HoldoutComparisonError,
    compare,
)


def metrics(correct: int, total: int) -> dict[str, int]:
    return {"generated_correct": correct, "total": total}


class TTR1HoldoutComparisonTests(unittest.TestCase):
    def write(self, root: Path, name: str, value: dict) -> Path:
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def report(self, *, control: str | None, overall: int) -> dict:
        value = {
            "schema": (
                "shohin-idr1-revision-evaluation-v1"
                if control is None
                else "shohin-ttr1-control-evaluation-v1"
            ),
            "status": "complete",
            "split": "holdout",
            "model_revision": "pinned",
            "data_sha256": "data",
            "data_report_sha256": "receipt",
            "full_row_count": 1279,
            "shard_count": 1,
            "metrics": {
                "overall": metrics(overall, 1279),
                "math500": metrics(80, 621),
                "bbh_logic": metrics(250, 625),
                "mbpp": metrics(10, 33),
            },
        }
        if control is not None:
            value["control"] = control
        return value

    def test_passes_frozen_holdout_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            development = self.write(
                root,
                "development.json",
                {
                    "schema": "shohin-ttr1-development-comparison-v1",
                    "status": "complete",
                    "gate_pass": True,
                    "holdout_authorized": True,
                },
            )
            treatment = self.write(root, "treatment.json", self.report(control=None, overall=500))
            controls = []
            for index, name in enumerate(CONTROLS):
                report = self.report(control=name, overall=400 + index)
                report["metrics"]["math500"]["generated_correct"] = 70
                report["metrics"]["bbh_logic"]["generated_correct"] = 240
                report["metrics"]["mbpp"]["generated_correct"] = 9
                controls.append(self.write(root, f"{name}.json", report))
            output = root / "comparison.json"
            result = compare(
                argparse.Namespace(
                    development_comparison=development,
                    treatment=treatment,
                    control_report=controls,
                    output=output,
                )
            )
            self.assertTrue(result["gate_pass"])
            self.assertTrue(result["product_authorized"])
            self.assertTrue(output.exists())

    def test_rejects_unqualified_development(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            development = self.write(
                root,
                "development.json",
                {
                    "schema": "shohin-ttr1-development-comparison-v1",
                    "status": "complete",
                    "gate_pass": False,
                    "holdout_authorized": False,
                },
            )
            with self.assertRaisesRegex(
                TTR1HoldoutComparisonError, "did not authorize"
            ):
                compare(
                    argparse.Namespace(
                        development_comparison=development,
                        treatment=root / "missing.json",
                        control_report=[],
                        output=root / "comparison.json",
                    )
                )


if __name__ == "__main__":
    unittest.main()
