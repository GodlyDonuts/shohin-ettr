from __future__ import annotations

from pathlib import Path

import pytest

import hf_q36_mtr_evaluate_temporal_gate as module


class _TemporalModel:
    def __init__(self, *args, **kwargs) -> None:
        self.evaluating = False
        self.reset = False

    def trainable_parameter_count(self) -> int:
        return module.GATE_PARAMETERS

    def eval(self) -> None:
        self.evaluating = True

    def reset_routing_receipt(self) -> None:
        self.reset = True


class _MultiModel(_TemporalModel):
    def trainable_parameter_count(self) -> int:
        return module.MULTI_GATE_PARAMETERS


def _metadata(role_receipt: dict[str, str]) -> dict[str, object]:
    return {
        "architecture": "q36-tokenwise-temporal-residual-gate-v1",
        "model_revision": module.MODEL_REVISION,
        "model_config_sha256": module.MODEL_CONFIG_SHA256,
        "controlled_layer_indices": list(module.CONTROLLED_LAYER_INDICES),
        "gate_parameters": module.GATE_PARAMETERS,
        "initial_revision_weight": module.GATE_INITIAL_REVISION_WEIGHT,
        "trainable_master_dtype": module.TRAINABLE_MASTER_DTYPE,
        "role_receipt": role_receipt,
    }


def test_loader_binds_role_pair_gate_and_native_moe(monkeypatch) -> None:
    role_receipt = {
        "owner_checkpoint_sha256": "1" * 64,
        "revision_checkpoint_sha256": "2" * 64,
        "owner_state_sha256": "3" * 64,
        "revision_state_sha256": "4" * 64,
    }
    monkeypatch.setattr(
        module, "_role_pair", lambda owner, revision: ({}, {}, role_receipt)
    )
    monkeypatch.setattr(
        module, "load_product_backbone", lambda *args, **kwargs: (object(), "causal")
    )
    monkeypatch.setattr(
        module,
        "validate_backbone_geometry",
        lambda backbone: list(module.CONTROLLED_LAYER_INDICES),
    )
    monkeypatch.setattr(
        module, "validate_backbone_moe_surface", lambda backbone: {"experts": 256}
    )
    monkeypatch.setattr(
        module,
        "resolve_product_backbone_layout",
        lambda backbone: (object(), object(), module.HIDDEN_SIZE, "causal"),
    )
    monkeypatch.setattr(module, "TemporalGatedProductModel", _TemporalModel)
    monkeypatch.setattr(
        module,
        "restore_gate_checkpoint",
        lambda checkpoint, model: (256, _metadata(role_receipt)),
    )
    model, metadata, loader, receipt = module.load_temporal_gate_model(
        Path("model"), Path("owner"), Path("revision"), Path("gate")
    )
    assert isinstance(model, _TemporalModel)
    assert model.evaluating and model.reset
    assert metadata == _metadata(role_receipt)
    assert loader == "causal"
    assert receipt == {
        "backbone_layout": "causal",
        "native_moe_surface": {"experts": 256},
        "role_receipt": role_receipt,
    }


def test_loader_rejects_gate_role_mismatch(monkeypatch) -> None:
    role_receipt = {"owner_state_sha256": "1" * 64}
    monkeypatch.setattr(
        module, "_role_pair", lambda owner, revision: ({}, {}, role_receipt)
    )
    monkeypatch.setattr(
        module, "load_product_backbone", lambda *args, **kwargs: (object(), "causal")
    )
    monkeypatch.setattr(
        module,
        "validate_backbone_geometry",
        lambda backbone: list(module.CONTROLLED_LAYER_INDICES),
    )
    monkeypatch.setattr(module, "validate_backbone_moe_surface", lambda backbone: {})
    monkeypatch.setattr(
        module,
        "resolve_product_backbone_layout",
        lambda backbone: (object(), object(), module.HIDDEN_SIZE, "causal"),
    )
    monkeypatch.setattr(module, "TemporalGatedProductModel", _TemporalModel)
    metadata = _metadata({"owner_state_sha256": "2" * 64})
    monkeypatch.setattr(
        module, "restore_gate_checkpoint", lambda checkpoint, model: (256, metadata)
    )
    with pytest.raises(module.Q36MTRTemporalGateEvaluationError):
        module.load_temporal_gate_model(
            Path("model"), Path("owner"), Path("revision"), Path("gate")
        )


def test_multi_loader_binds_two_sibling_trajectories(monkeypatch) -> None:
    role_receipt = {
        "owner_checkpoint_sha256": "1" * 64,
        "revision_checkpoint_sha256": "2" * 64,
        "draft_hidden_checkpoint_sha256": "3" * 64,
    }
    monkeypatch.setattr(
        module, "_role_bank", lambda owner, revision, hidden: ({}, role_receipt)
    )
    monkeypatch.setattr(
        module, "load_product_backbone", lambda *args, **kwargs: (object(), "causal")
    )
    monkeypatch.setattr(
        module,
        "validate_backbone_geometry",
        lambda backbone: list(module.CONTROLLED_LAYER_INDICES),
    )
    monkeypatch.setattr(
        module, "validate_backbone_moe_surface", lambda backbone: {"experts": 256}
    )
    monkeypatch.setattr(
        module,
        "resolve_product_backbone_layout",
        lambda backbone: (object(), object(), module.HIDDEN_SIZE, "causal"),
    )
    monkeypatch.setattr(module, "MultiTrajectoryGatedProductModel", _MultiModel)
    metadata = {
        "architecture": "q36-tokenwise-multi-trajectory-residual-gate-v1",
        "model_revision": module.MODEL_REVISION,
        "model_config_sha256": module.MODEL_CONFIG_SHA256,
        "controlled_layer_indices": list(module.CONTROLLED_LAYER_INDICES),
        "gate_parameters": module.MULTI_GATE_PARAMETERS,
        "branch_names": list(module.MULTI_BRANCHES),
        "initial_branch_weights": list(module.MULTI_INITIAL_WEIGHTS),
        "trainable_master_dtype": module.TRAINABLE_MASTER_DTYPE,
        "role_receipt": role_receipt,
    }
    monkeypatch.setattr(
        module, "restore_gate_checkpoint", lambda *args, **kwargs: (256, metadata)
    )
    model, observed, loader, receipt = module.load_multi_trajectory_gate_model(
        Path("model"),
        Path("owner"),
        Path("revision"),
        Path("hidden"),
        Path("gate"),
    )
    assert isinstance(model, _MultiModel)
    assert model.evaluating and model.reset
    assert observed == metadata
    assert loader == "causal"
    assert receipt["role_receipt"] == role_receipt
