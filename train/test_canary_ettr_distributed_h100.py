from __future__ import annotations

import os

import pytest

from canary_ettr_distributed_h100 import (
    ETTRDistributedCanaryError,
    _environment,
    _settings,
)


def test_distributed_canary_uses_validated_b16_architecture_shape() -> None:
    settings = _settings(2026072902, "default")
    assert settings.batch_size == 16
    assert settings.microsteps == 1
    assert settings.train_scope == "architecture"
    assert settings.reactor_steps == 4


def test_distributed_canary_rejects_missing_torchrun_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(
        ETTRDistributedCanaryError,
        match="distributed environment differs",
    ):
        _environment(2)


def test_distributed_canary_source_contains_rank_identity_gate() -> None:
    source = open(
        os.path.join(
            os.path.dirname(__file__),
            "canary_ettr_distributed_h100.py",
        ),
        encoding="utf-8",
    ).read()
    assert "ETTRDistributedGradientAverager" in source
    assert "final_parameter_sha256" in source
    assert "dist.ReduceOp.MAX" in source
    assert "checkpoint_after != checkpoint_sha256" in source
