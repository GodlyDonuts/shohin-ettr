"""Tests for one-shot PCF1 Slurm accounting custody."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pytest

from capture_pcf1_slurm_accounting import PCF1AccountingError, capture


def _dispatch(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "shohin-pcf1-dispatch-v1",
                "status": "submitted",
                "run_id": "pcf1-test",
                "partition": "normal",
                "excluded_nodes": [
                    "evc26",
                    "evc29",
                    "evc31",
                    "evc32",
                    "evc33",
                    "evc38",
                    "evc46",
                ],
                "retry_authorized": False,
                "successor_authorized": False,
                "stop_after_gate": True,
                "accounting_predecessors": ["prepare_inputs", "draft_generate"],
                "job_ids": {"prepare_inputs": "101", "draft_generate": "202"},
                "stage_resources": {
                    "prepare_inputs": {
                        "gpus": 0,
                        "is_array": False,
                        "array_tasks": 1,
                    },
                    "draft_generate": {
                        "gpus": 1,
                        "is_array": True,
                        "array_tasks": 2,
                    },
                },
            }
        )
    )


def _args(root: Path) -> argparse.Namespace:
    dispatch = root / "dispatch.json"
    _dispatch(dispatch)
    return argparse.Namespace(
        run_id="pcf1-test",
        dispatch_receipt=dispatch,
        output=root / "accounting.json",
    )


def _runner(text: str):
    def run(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, stdout=text, stderr="")

    return run


def test_captures_array_tasks_without_double_counting_root(tmp_path: Path) -> None:
    text = "\n".join(
        (
            "101|COMPLETED|normal|10|cpu=2|evc10|0:0|0|",
            "202|COMPLETED|normal|99|cpu=4,gres/gpu=1|evc11|0:0|0|",
            "202_0|COMPLETED|normal|20|cpu=4,gres/gpu=1,gres/gpu:nvidia_h100_pcie=1|evc11|0:0|0|",
            "202_0.batch|COMPLETED||20|cpu=4,gres/gpu=1|evc11|0:0|0|",
            "202_1|COMPLETED|normal|30|cpu=4,gres/gpu=1,gres/gpu:nvidia_h100_pcie=1|evc12|0:0|0|",
        )
    )
    report = capture(_args(tmp_path), runner=_runner(text))
    assert report["charged_gpu_seconds"] == 50
    assert [
        row["job_id_raw"] for row in report["jobs"]["draft_generate"]["records"]
    ] == ["202_0", "202_1"]
    assert report["jobs"]["draft_generate"]["records"][0]["allocated_gpu_types"] == {
        "nvidia_h100_pcie": 1
    }


@pytest.mark.parametrize(
    "record, message",
    (
        ("101|FAILED|normal|10|cpu=2|evc10|1:0|0|", "not complete"),
        ("101|COMPLETED|debug|10|cpu=2|evc10|0:0|0|", "wrong partition"),
        ("101|COMPLETED|normal|10|cpu=2|evc26|0:0|0|", "excluded node"),
    ),
)
def test_rejects_inadmissible_allocation(
    tmp_path: Path, record: str, message: str
) -> None:
    text = "\n".join(
        (
            record,
            "202_0|COMPLETED|normal|20|cpu=4,gres/gpu:nvidia_h100_pcie=1|evc11|0:0|0|",
        )
    )
    with pytest.raises(PCF1AccountingError, match=message):
        capture(_args(tmp_path), runner=_runner(text))


def test_rejects_missing_required_job(tmp_path: Path) -> None:
    text = "101|COMPLETED|normal|10|cpu=2|evc10|0:0|0|"
    with pytest.raises(PCF1AccountingError, match="array allocation geometry"):
        capture(_args(tmp_path), runner=_runner(text))


@pytest.mark.parametrize(
    "array_rows",
    (
        ("202_0|COMPLETED|normal|20|cpu=4,gres/gpu=1|evc11|0:0|0|",),
        (
            "202_0|COMPLETED|normal|20|cpu=4,gres/gpu=1|evc11|0:0|0|",
            "202_1|COMPLETED|normal|20|cpu=4,gres/gpu=1|evc11|0:0|0|",
            "202_2|COMPLETED|normal|20|cpu=4,gres/gpu=1|evc11|0:0|0|",
        ),
        (
            "202_0|COMPLETED|normal|20|cpu=4,gres/gpu=1|evc11|0:0|0|",
            "202_0|COMPLETED|normal|20|cpu=4,gres/gpu=1|evc11|0:0|0|",
            "202_1|COMPLETED|normal|20|cpu=4,gres/gpu=1|evc11|0:0|0|",
        ),
    ),
)
def test_rejects_missing_or_extra_array_index(
    tmp_path: Path, array_rows: tuple[str, ...]
) -> None:
    text = "\n".join(("101|COMPLETED|normal|10|cpu=2|evc10|0:0|0|", *array_rows))
    with pytest.raises(PCF1AccountingError, match="array allocation geometry"):
        capture(_args(tmp_path), runner=_runner(text))


def test_rejects_missing_gpu_tres_for_gpu_stage(tmp_path: Path) -> None:
    text = "\n".join(
        (
            "101|COMPLETED|normal|10|cpu=2|evc10|0:0|0|",
            "202_0|COMPLETED|normal|20|cpu=4|evc11|0:0|0|",
            "202_1|COMPLETED|normal|20|cpu=4|evc12|0:0|0|",
        )
    )
    with pytest.raises(PCF1AccountingError, match="GPU allocation differs"):
        capture(_args(tmp_path), runner=_runner(text))


def test_rejects_generic_only_gpu_without_h100_type(tmp_path: Path) -> None:
    text = "\n".join(
        (
            "101|COMPLETED|normal|10|cpu=2|evc10|0:0|0|",
            "202_0|COMPLETED|normal|20|cpu=4,gres/gpu=1|evc11|0:0|0|",
            "202_1|COMPLETED|normal|20|cpu=4,gres/gpu=1|evc12|0:0|0|",
        )
    )
    with pytest.raises(PCF1AccountingError, match="GPU allocation differs"):
        capture(_args(tmp_path), runner=_runner(text))


def test_rejects_restarted_allocation(tmp_path: Path) -> None:
    text = "\n".join(
        (
            "101|COMPLETED|normal|10|cpu=2|evc10|0:0|1|",
            "202_0|COMPLETED|normal|20|cpu=4,gres/gpu:nvidia_h100_pcie=1|evc11|0:0|0|",
            "202_1|COMPLETED|normal|20|cpu=4,gres/gpu:nvidia_h100_pcie=1|evc12|0:0|0|",
        )
    )
    with pytest.raises(PCF1AccountingError, match="restarted"):
        capture(_args(tmp_path), runner=_runner(text))
