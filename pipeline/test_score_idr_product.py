from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from score_idr_product import compare


def _write_bound(path: Path, report: Path, rows: list[dict], stage: str) -> None:
    encoded = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(encoded, encoding="utf-8")
    report.write_text(
        json.dumps(
            {
                "status": "complete",
                "stage": stage,
                "output": str(path.resolve()),
                "output_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )


def test_product_comparison_passes_matched_gain(tmp_path: Path) -> None:
    tasks = ["gsm8k"] * 100 + ["math500"] * 100 + ["gpqa"] * 198
    tasks += ["bbh_logic"] * 100 + ["humaneval"] * 20 + ["mbpp"] * 20 + ["aime"] * 30
    source = [
        {"identity_sha256": f"{index:064x}", "task": task}
        for index, task in enumerate(tasks)
    ]
    control = [
        {
            "schema": "shohin-aqc1-product-candidates-v1",
            "identity_sha256": row["identity_sha256"],
            "task": row["task"],
            "correct": index % 2 == 0,
        }
        for index, row in enumerate(source)
    ]
    treatment = [dict(row, correct=True) for row in control]
    paths = {}
    for name, rows, stage in (
        ("source", source, "source"),
        ("treatment", treatment, "merged-idr4"),
        ("control", control, "merged-control"),
    ):
        path, report = tmp_path / f"{name}.jsonl", tmp_path / f"{name}.report.json"
        _write_bound(path, report, rows, stage)
        paths[name] = (path, report)
    output = tmp_path / "comparison.json"
    result = compare(
        argparse.Namespace(
            source=paths["source"][0],
            source_report=paths["source"][1],
            treatment=paths["treatment"][0],
            treatment_report=paths["treatment"][1],
            treatment_stage="merged-idr4",
            control=paths["control"][0],
            control_report=paths["control"][1],
            control_stage="merged-control",
            report=output,
        )
    )
    assert result["gate_pass"] is True
    assert result["deltas"]["solved"] == 269
    assert result["treatment_summary"]["total"] == 538
    assert result["treatment_summary"]["aime"]["total"] == 30
