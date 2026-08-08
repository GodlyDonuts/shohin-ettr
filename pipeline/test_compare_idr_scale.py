from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from compare_idr_scale import IDRScaleComparisonError, compare


def write_report(
    path: Path,
    split: str,
    adapter: str,
    *,
    overall: int,
    math: int,
    logic: int,
    code: int,
) -> None:
    totals = {
        "development": (1_289, 621, 635, 33),
        "holdout": (1_279, 621, 625, 33),
    }[split]
    values = (overall, math, logic, code)
    metrics = {
        task: {"total": total, "generated_correct": correct}
        for task, total, correct in zip(
            ("overall", "math500", "bbh_logic", "mbpp"), totals, values
        )
    }
    path.write_text(
        json.dumps(
            {
                "schema": "shohin-idr1-revision-evaluation-v1",
                "status": "complete",
                "split": split,
                "merged_from_shards": True,
                "full_row_count": totals[0],
                "model_root": "/model",
                "model_revision": "revision-1",
                "adapter_checkpoint_sha256": adapter,
                "data_sha256": f"data-{split}",
                "data_report_sha256": "report",
                "generation_mode": "greedy",
                "max_new_tokens": 768,
                "batch_size": 2,
                "seed": 7,
                "metrics": metrics,
            }
        )
        + "\n"
    )


def arguments(tmp_path: Path, *, passing: bool = True) -> argparse.Namespace:
    paths = {}
    for split in ("development", "holdout"):
        control = tmp_path / f"{split}.control.json"
        trained = tmp_path / f"{split}.trained.json"
        baseline = (300, 120, 170, 10)
        treatment = (380, 150, 215, 15) if passing else (340, 155, 175, 10)
        write_report(control, split, "c" * 64, overall=baseline[0], math=baseline[1], logic=baseline[2], code=baseline[3])
        write_report(trained, split, "t" * 64, overall=treatment[0], math=treatment[1], logic=treatment[2], code=treatment[3])
        paths[split] = (trained, control)
    return argparse.Namespace(
        trained_development=paths["development"][0],
        control_development=paths["development"][1],
        trained_holdout=paths["holdout"][0],
        control_holdout=paths["holdout"][1],
        output=tmp_path / "comparison.json",
    )


def test_passing_boundary_selects_390m(tmp_path: Path):
    report = compare(arguments(tmp_path))
    assert report["gate_pass"] is True
    assert report["scratch_scale_decision"] == "shohin_390m"
    assert report["splits"]["development"]["deltas"]["overall"]["correct"] == 80


def test_sub_five_point_boundary_selects_920m(tmp_path: Path):
    report = compare(arguments(tmp_path, passing=False))
    assert report["gate_pass"] is False
    assert report["scratch_scale_decision"] == "shohin_920m"


def test_identical_treatment_and_control_adapters_fail_closed(tmp_path: Path):
    args = arguments(tmp_path)
    trained = json.loads(args.trained_development.read_text())
    control = json.loads(args.control_development.read_text())
    trained["adapter_checkpoint_sha256"] = control["adapter_checkpoint_sha256"]
    args.trained_development.write_text(json.dumps(trained) + "\n")
    with pytest.raises(IDRScaleComparisonError, match="identical"):
        compare(args)
