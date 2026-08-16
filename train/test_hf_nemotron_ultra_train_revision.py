"""CPU contract tests for direct Nemotron Ultra Shohin training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hf_nemotron_ultra_train_revision as training
from hf_nemotron_ultra_mechanics import SCHEMA as MECHANICS_SCHEMA
from q36_upward_moe_ultra_host import (
    MINIMUM_H100S,
    MODEL_REVISION,
    TRAINABLE_PARAMETERS_PER_ROLE,
)


def _mechanics(manifest_sha256: str) -> dict[str, object]:
    return {
        "schema": MECHANICS_SCHEMA,
        "status": "pass",
        "score_rows_read": 0,
        "benchmark_rows_read": 0,
        "model_revision": MODEL_REVISION,
        "trainable_parameters": TRAINABLE_PARAMETERS_PER_ROLE,
        "native_router_expert_trainables": 0,
        "serialization_restore_exact": True,
        "devices": [{} for _ in range(MINIMUM_H100S)],
        "model_receipt": {
            "manifest_sha256": manifest_sha256,
            "exact_membership": True,
        },
    }


def test_direct_ultra_training_reuses_exact_super_recipe() -> None:
    assert training.UPDATES == 256
    assert training.GRADIENT_ACCUMULATION == 8
    assert training.CONSUMED_PRESENTATIONS == 2048
    assert training.MAX_SEQUENCE_LENGTH == 4096
    assert training.LEARNING_RATE == 2e-5
    assert training.DATA_PRESENTATIONS == 9655
    assert training.TRAINABLE_PARAMETERS_PER_ROLE == 4_718_592


def test_direct_ultra_training_requires_exact_mechanics(tmp_path: Path) -> None:
    manifest = "a" * 64
    path = tmp_path / "mechanics.json"
    path.write_text(json.dumps(_mechanics(manifest)), encoding="utf-8")
    assert training.validate_mechanics_report(path, manifest)["status"] == "pass"
    payload = _mechanics(manifest)
    payload["benchmark_rows_read"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(training.NemotronUltraTrainingError, match="authorization"):
        training.validate_mechanics_report(path, manifest)


def test_direct_ultra_training_report_is_write_once(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    training._atomic_json(output, {"status": "complete"})
    with pytest.raises(training.NemotronUltraTrainingError, match="existing"):
        training._atomic_json(output, {"status": "changed"})
