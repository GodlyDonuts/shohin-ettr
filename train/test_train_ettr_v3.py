from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ettr_distributed import ETTRDistributedCursor
from train_ettr_v3 import (
    ETTRV3TrainerError,
    _data_state,
    _validate_resume_cursor,
)


class _Manifest:
    dataset_sha256 = "b" * 64

    @staticmethod
    def sha256() -> str:
        return "a" * 64


def _stream():
    return SimpleNamespace(
        manifest=_Manifest(),
        records={"train": tuple(range(16))},
    )


def test_cluster_entrypoint_imports_without_external_pythonpath(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("train_ettr_v3.py")),
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    assert process.returncode == 0, process.stderr


def test_compiled_resume_cursor_round_trip() -> None:
    stream = _stream()
    cursor = ETTRDistributedCursor(epoch=2, position=8)
    state = _data_state(
        stream=stream,
        release_sha256="c" * 64,
        cursor=cursor,
        data_seed=7,
        world_size=2,
        accumulation=2,
        optimizer_step=2,
        compile_backend="inductor",
        compile_mode="default",
    )
    assert (
        _validate_resume_cursor(
            state,
            stream=stream,
            release_sha256="c" * 64,
            data_seed=7,
            world_size=2,
            accumulation=2,
            optimizer_step=2,
            compile_backend="inductor",
            compile_mode="default",
        )
        == cursor
    )


@pytest.mark.parametrize(
    ("compile_backend", "compile_mode"),
    (
        (None, None),
        ("inductor", "reduce-overhead"),
    ),
)
def test_resume_rejects_execution_mode_change(
    compile_backend,
    compile_mode,
) -> None:
    stream = _stream()
    state = _data_state(
        stream=stream,
        release_sha256="c" * 64,
        cursor=ETTRDistributedCursor(epoch=0, position=8),
        data_seed=7,
        world_size=2,
        accumulation=2,
        optimizer_step=2,
        compile_backend="inductor",
        compile_mode="default",
    )
    with pytest.raises(
        ETTRV3TrainerError,
        match="resume data stream differs",
    ):
        _validate_resume_cursor(
            state,
            stream=stream,
            release_sha256="c" * 64,
            data_seed=7,
            world_size=2,
            accumulation=2,
            optimizer_step=2,
            compile_backend=compile_backend,
            compile_mode=compile_mode,
        )
