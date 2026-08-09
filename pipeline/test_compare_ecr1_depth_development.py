#!/usr/bin/env python3
"""Focused tests for the frozen ECR1 all-layer depth gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compare_ecr1_depth_development import compare


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _evaluation(correct: int, *, math: int, logic: int, code: int) -> dict:
    return {
        "schema": "shohin-idr1-revision-evaluation-v1",
        "status": "complete",
        "split": "development",
        "full_row_count": 1289,
        "merged_from_shards": True,
        "shard_count": 8,
        "ecr_code_intervention": "normal",
        "model_revision": "revision",
        "data_sha256": "data",
        "data_report_sha256": "report",
        "metrics": {
            "overall": {"generated_correct": correct, "total": 1289},
            "math500": {"generated_correct": math, "total": 623},
            "bbh_logic": {"generated_correct": logic, "total": 637},
            "mbpp": {"generated_correct": code, "total": 29},
        },
    }


def _fit(parameters: int, mode: str) -> dict:
    return {
        "schema": "shohin-ecr1-product-training-v1",
        "status": "complete",
        "updates": 256,
        "selected_rows": 9651,
        "trainable_parameters": parameters,
        "ecr1_config": {
            "mode": mode,
            "controlled_layers": 16,
            "rank": 8,
            "alpha": 8.0,
        },
        "ecr1_draft_control": "normal",
        "protected_router_expert_trainables": 0,
        "sequence_custody": {
            "overflow_rows": 0,
            "source_retention": 1.0,
            "draft_retention": 1.0,
            "target_retention": 1.0,
        },
        "charged_tokens": 338620,
    }


def _args(root: Path) -> argparse.Namespace:
    paths = {
        name: root / f"{name}.json"
        for name in (
            "treatment_report",
            "shared_report",
            "treatment_fit",
            "shared_fit",
            "final_four_comparison",
            "unchanged_report",
            "mtr_report",
            "output",
        )
    }
    _write(paths["treatment_report"], _evaluation(300, math=80, logic=210, code=10))
    _write(paths["shared_report"], _evaluation(250, math=70, logic=172, code=8))
    _write(paths["treatment_fit"], _fit(532480, "expert_conditioned"))
    _write(paths["shared_fit"], _fit(524288, "shared"))
    _write(
        paths["final_four_comparison"],
        {
            "schema": "shohin-ecr1-development-comparison-v1",
            "status": "complete",
            "arms": {"treatment": {"correct": 221}, "shared": {"correct": 223}},
            "holdout_authorized": False,
        },
    )
    _write(paths["unchanged_report"], _evaluation(191, math=40, logic=145, code=6))
    _write(paths["mtr_report"], _evaluation(204, math=42, logic=156, code=6))
    return argparse.Namespace(**paths)


def test_pass(tmp_path: Path) -> None:
    result = compare(_args(tmp_path))
    assert result["stage_two_causal_controls_authorized"] is True
    assert result["holdout_authorized"] is False


def test_kill_on_shared_margin(tmp_path: Path) -> None:
    args = _args(tmp_path)
    _write(args.shared_report, _evaluation(270, math=70, logic=192, code=8))
    result = compare(args)
    assert result["stage_two_causal_controls_authorized"] is False
    assert result["close_ecr1_if_false"] is True
