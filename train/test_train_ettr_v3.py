from __future__ import annotations

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
