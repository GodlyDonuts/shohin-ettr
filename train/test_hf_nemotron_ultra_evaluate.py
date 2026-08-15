"""CPU contract tests for the 550B-A55B matched evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import hf_nemotron_ultra_evaluate as ultra_evaluate
from hf_nemotron_ultra_evaluate import (
    NemotronUltraEvaluationError,
    _atomic_json,
    load_transferred_checkpoint,
    validate_mechanics_report,
)
from hf_nemotron_ultra_mechanics import SCHEMA as MECHANICS_SCHEMA
from lift_nemotron_super_adapter_to_ultra import (
    CHECKPOINT_SCHEMA,
    FACTOR_SHA256,
    SCHEMA as TRANSFER_SCHEMA,
    _state_sha256,
)
from q36_upward_moe_ultra_host import (
    MINIMUM_H100S,
    MODEL_REVISION,
    TRAINABLE_PARAMETERS_PER_ROLE,
)


def _mechanics_payload(manifest_sha256: str) -> dict[str, object]:
    return {
        "schema": MECHANICS_SCHEMA,
        "status": "pass",
        "model_revision": MODEL_REVISION,
        "score_rows_read": 0,
        "benchmark_rows_read": 0,
        "trainable_parameters": TRAINABLE_PARAMETERS_PER_ROLE,
        "native_router_expert_trainables": 0,
        "serialization_restore_exact": True,
        "devices": [{} for _ in range(MINIMUM_H100S)],
        "model_receipt": {
            "manifest_sha256": manifest_sha256,
            "exact_membership": True,
        },
    }


def test_ultra_evaluation_requires_exact_score_free_mechanics(tmp_path: Path) -> None:
    expected = "a" * 64
    report = tmp_path / "mechanics.json"
    report.write_text(json.dumps(_mechanics_payload(expected)), encoding="utf-8")
    assert validate_mechanics_report(report, expected)["status"] == "pass"
    payload = _mechanics_payload(expected)
    payload["benchmark_rows_read"] = 1
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(NemotronUltraEvaluationError, match="authorization"):
        validate_mechanics_report(report, expected)


def test_ultra_evaluation_rejects_manifest_substitution(tmp_path: Path) -> None:
    report = tmp_path / "mechanics.json"
    report.write_text(json.dumps(_mechanics_payload("a" * 64)), encoding="utf-8")
    with pytest.raises(NemotronUltraEvaluationError, match="authorization"):
        validate_mechanics_report(report, "b" * 64)


def test_ultra_evaluation_report_is_write_once(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    _atomic_json(path, {"status": "complete"})
    with pytest.raises(NemotronUltraEvaluationError, match="existing"):
        _atomic_json(path, {"status": "changed"})


class _TinyRevision:
    def __init__(self) -> None:
        self.adapter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def named_parameters(self):
        return [("adapter", self.adapter)]

    def trainable_state_sha256(self) -> str:
        return _state_sha256({"adapter": self.adapter.detach().cpu()})


def test_ultra_transfer_restore_is_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ultra_evaluate, "TRAINABLE_PARAMETERS_PER_ROLE", 1)
    state = {"adapter": torch.tensor([3.0], dtype=torch.float32)}
    metadata = {
        "schema": TRANSFER_SCHEMA,
        "target_model_revision": MODEL_REVISION,
        "ultra_config_sha256": ultra_evaluate.MODEL_CONFIG_SHA256,
        "factor_sha256": FACTOR_SHA256,
        "trainable_parameters": 1,
        "target_trainable_state_sha256": _state_sha256(state),
        "label_rows_read": 0,
        "benchmark_rows_read": 0,
        "optimizer_updates": 0,
        "native_router_expert_trainables": 0,
    }
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "trainable_state": state,
            "metadata": metadata,
        },
        checkpoint,
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                **metadata,
                "status": "complete",
                "checkpoint_sha256": ultra_evaluate.sha256_file(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    model = _TinyRevision()
    assert load_transferred_checkpoint(checkpoint, report, model) == metadata
    assert model.adapter.item() == 3.0
    altered = json.loads(report.read_text(encoding="utf-8"))
    altered["checkpoint_sha256"] = "0" * 64
    report.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(NemotronUltraEvaluationError, match="checkpoint differs"):
        load_transferred_checkpoint(checkpoint, report, _TinyRevision())
    torch.save(["not", "a", "checkpoint"], checkpoint)
    with pytest.raises(NemotronUltraEvaluationError, match="checkpoint differs"):
        load_transferred_checkpoint(checkpoint, report, _TinyRevision())
