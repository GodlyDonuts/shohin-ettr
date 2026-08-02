from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from eval_parallel_schedule_ensemble import (
    ParallelScheduleEnsembleError,
    _validate_args,
)


def _args(tmp_path: Path, **overrides: object) -> Namespace:
    values: dict[str, object] = {
        "release_root": tmp_path,
        "release_sha256": "0" * 64,
        "data_root": tmp_path,
        "tokenizer": tmp_path / "tokenizer.json",
        "protected_checkpoint": tmp_path / "protected.pt",
        "joint_model": tmp_path / "joint.pt",
        "joint_model_sha256": "1" * 64,
        "joint_run_contract": tmp_path / "joint.json",
        "joint_run_contract_sha256": "2" * 64,
        "compiler": tmp_path / "compiler.safetensors",
        "compiler_sha256": "3" * 64,
        "compiler_contract": tmp_path / "compiler.json",
        "compiler_contract_sha256": "4" * 64,
        "schedule_run_dir": [tmp_path / "schedule-a", tmp_path / "schedule-b"],
        "schedule_run_sha256s_sha256": ["5" * 64, "6" * 64],
        "output": tmp_path / "output",
        "source_commit": "7" * 40,
        "data_seed": 8,
        "max_batches": 32,
    }
    values.update(overrides)
    return Namespace(**values)


def test_ensemble_accepts_two_distinct_hash_bound_members(tmp_path: Path) -> None:
    _validate_args(_args(tmp_path))


@pytest.mark.parametrize("member_count", [0, 1, 7])
def test_ensemble_rejects_member_count_outside_cap(
    tmp_path: Path,
    member_count: int,
) -> None:
    with pytest.raises(ParallelScheduleEnsembleError, match="arguments differ"):
        _validate_args(
            _args(
                tmp_path,
                schedule_run_dir=[
                    tmp_path / f"schedule-{index}"
                    for index in range(member_count)
                ],
                schedule_run_sha256s_sha256=["5" * 64] * member_count,
            )
        )


def test_ensemble_rejects_mismatched_member_receipts(tmp_path: Path) -> None:
    with pytest.raises(ParallelScheduleEnsembleError, match="arguments differ"):
        _validate_args(
            _args(
                tmp_path,
                schedule_run_sha256s_sha256=["5" * 64],
            )
        )


def test_ensemble_rejects_duplicate_member_paths(tmp_path: Path) -> None:
    duplicate = tmp_path / "same-schedule"
    with pytest.raises(ParallelScheduleEnsembleError, match="arguments differ"):
        _validate_args(
            _args(
                tmp_path,
                schedule_run_dir=[duplicate, duplicate],
            )
        )
