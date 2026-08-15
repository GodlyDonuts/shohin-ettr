from __future__ import annotations

import argparse
import json

import pytest
import torch
import torch.nn as nn

import hf_q36_mtr_train_temporal_gate as module
from shared_post_mlp_revision import trainable_state, trainable_state_sha256


class _GateLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_weight = nn.Parameter(torch.randn(1, module.HIDDEN_SIZE))
        self.gate_bias = nn.Parameter(torch.randn(1))


class _GateModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_GateLayer() for _ in module.CONTROLLED_LAYER_INDICES]
        )


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        model_revision=module.MODEL_REVISION,
        model_config_sha256=module.MODEL_CONFIG_SHA256,
        updates=module.REVISION_UPDATES,
        max_rows=module.REVISION_PRESENTATIONS,
        max_sequence_length=module.REVISION_MAX_SEQUENCE_LENGTH,
        learning_rate=module.GATE_LEARNING_RATE,
        gradient_accumulation=module.GATE_GRADIENT_ACCUMULATION,
        batch_size=module.GATE_BATCH_SIZE,
        seed=module.GATE_SEED,
        data_seed=module.REVISION_DATA_SEED,
        initial_revision_weight=module.GATE_INITIAL_REVISION_WEIGHT,
        routing_supervision_weight=0.0,
        causal_loss_weight=1.0,
        loss_chunk_size=module.LOSS_CHUNK_SIZE,
    )


def test_temporal_gate_settings_are_exact() -> None:
    module._validate_args(_args())
    changed = _args()
    changed.learning_rate *= 2
    with pytest.raises(module.Q36MTRTemporalGateTrainingError):
        module._validate_args(changed)
    supervised = _args()
    supervised.routing_supervision_weight = module.GATE_ROUTING_SUPERVISION_WEIGHT
    module._validate_args(supervised)
    supervised.routing_supervision_weight = 0.2
    with pytest.raises(module.Q36MTRTemporalGateTrainingError):
        module._validate_args(supervised)


def test_routing_supervision_masks_prompt_tokens() -> None:
    labels = torch.tensor([[-100, -100, 17, 23]], dtype=torch.long)
    assert module._response_routing_mask(labels).tolist() == [
        [False, False, True, True]
    ]
    with pytest.raises(module.Q36MTRTemporalGateTrainingError):
        module._response_routing_mask(torch.full((1, 4), -100, dtype=torch.long))


def test_gate_checkpoint_is_trainable_only_and_restores(tmp_path) -> None:
    model = _GateModel()
    state = trainable_state(model)
    module._validate_gate_state(state)
    expected_sha256 = trainable_state_sha256(state)
    path = tmp_path / "checkpoint_0000256.pt"
    metadata = {"status": "test", "optimizer_state_serialized": False}
    module.save_gate_checkpoint(
        path, model, module.REVISION_UPDATES, metadata  # type: ignore[arg-type]
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert set(payload) == {"schema", "update", "trainable_state", "metadata"}
    assert payload["schema"] == module.CHECKPOINT_SCHEMA
    assert len(payload["trainable_state"]) == 32
    assert sum(t.numel() for t in payload["trainable_state"].values()) == 32_784
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    update, restored = module.restore_gate_checkpoint(
        path, model  # type: ignore[arg-type]
    )
    assert update == module.REVISION_UPDATES
    assert restored == metadata
    assert trainable_state_sha256(trainable_state(model)) == expected_sha256


def test_role_pair_binds_owner_and_aligned_states(monkeypatch, tmp_path) -> None:
    owner_path = tmp_path / "owner.pt"
    revision_path = tmp_path / "revision.pt"
    owner_path.write_bytes(b"owner")
    revision_path.write_bytes(b"revision")
    shared = {
        "model_revision": module.MODEL_REVISION,
        "model_config_sha256": module.MODEL_CONFIG_SHA256,
        "controlled_layer_indices": list(module.CONTROLLED_LAYER_INDICES),
        "trainable_parameter_name_sha256": "1" * 64,
        "trainable_parameters": 1_179_648,
        "trainable_master_dtype": module.TRAINABLE_MASTER_DTYPE,
    }
    owner_state = {"state": torch.tensor([1.0])}
    revision_state = {"state": torch.tensor([2.0])}
    owner_state_sha256 = trainable_state_sha256(owner_state)
    payloads = {
        owner_path: {
            "metadata": {
                **shared,
                "role": "owner",
                "final_trainable_state_sha256": owner_state_sha256,
                "warm_start_checkpoint": None,
                "warm_start_checkpoint_sha256": None,
            },
            "trainable_state": owner_state,
        },
        revision_path: {
            "metadata": {
                **shared,
                "role": "aligned",
                "warm_start_checkpoint_sha256": owner_path.name * 4,
                "warm_start_update": module.REVISION_UPDATES,
                "initial_trainable_state_sha256": owner_state_sha256,
            },
            "trainable_state": revision_state,
        },
    }
    monkeypatch.setattr(module, "load_role_checkpoint_payload", payloads.__getitem__)
    monkeypatch.setattr(module, "sha256_file", lambda path: path.name * 4)
    owner, revision, receipt = module._role_pair(owner_path, revision_path)
    assert float(owner["state"][0]) == 1.0
    assert float(revision["state"][0]) == 2.0
    assert receipt["owner_state_sha256"] != receipt["revision_state_sha256"]
    payloads[revision_path]["metadata"]["warm_start_checkpoint_sha256"] = "0" * 64
    with pytest.raises(module.Q36MTRTemporalGateTrainingError):
        module._role_pair(owner_path, revision_path)
    payloads[revision_path]["metadata"]["warm_start_checkpoint_sha256"] = (
        owner_path.name * 4
    )
    payloads[revision_path]["metadata"]["role"] = "draft_hidden"
    with pytest.raises(module.Q36MTRTemporalGateTrainingError):
        module._role_pair(owner_path, revision_path)


def test_gate_state_rejects_wrong_count_or_nonfinite_value() -> None:
    model = _GateModel()
    state = trainable_state(model)
    state.pop(next(iter(state)))
    with pytest.raises(module.Q36MTRTemporalGateTrainingError):
        module._validate_gate_state(state)
    state = trainable_state(model)
    state[next(name for name in state if name.endswith("gate_bias"))][0] = float("nan")
    with pytest.raises(module.Q36MTRTemporalGateTrainingError):
        module._validate_gate_state(state)


def test_routing_loader_preserves_temporal_supervision(tmp_path) -> None:
    path = tmp_path / "revision.jsonl"
    rows = [
        {
            "schema": "shohin-q36-mtr-revision-train-v1",
            "question": f"question {index}",
            "response": f"response {index}",
            "outcome_class": outcome,
        }
        for index, outcome in enumerate(("both_wrong", "expert_only"))
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    loaded, digest = module._routing_rows_with_sha256(
        path, 2, module.REVISION_DATA_SEED, architecture="temporal"
    )
    assert {row["outcome_class"] for row in loaded} == {
        "both_wrong",
        "expert_only",
    }
    assert digest == module.sha256_file(path)


def test_routing_loader_preserves_multi_soft_targets(tmp_path) -> None:
    path = tmp_path / "multi.jsonl"
    rows = [
        {
            "schema": module.MULTI_ROW_SCHEMA,
            "question": f"question {index}",
            "response": f"response {index}",
            "outcome_class": outcome,
            "branch_names": list(module.MULTI_BRANCHES),
            "routing_target": target,
        }
        for index, (outcome, target) in enumerate(
            (("both_correct", [0.5, 0.5]), ("draft_hidden_only", [0.0, 1.0]))
        )
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    loaded, _ = module._routing_rows_with_sha256(
        path, 2, module.MULTI_DATA_SEED, architecture="multi_trajectory"
    )
    assert sorted(tuple(row["routing_target"]) for row in loaded) == [
        (0.0, 1.0),
        (0.5, 0.5),
    ]
    rows[0]["routing_target"] = [0.6, 0.6]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(module.Q36MTRTemporalGateTrainingError):
        module._routing_rows_with_sha256(
            path, 2, module.MULTI_DATA_SEED, architecture="multi_trajectory"
        )


def test_routing_loader_accepts_tri_trajectory_targets(tmp_path) -> None:
    path = tmp_path / "tri.jsonl"
    rows = [
        {
            "schema": module.TRI_ROW_SCHEMA,
            "question": "question",
            "response": "response",
            "outcome_class": "owner_revision_correct",
            "branch_names": list(module.TRI_BRANCHES),
            "routing_target": [0.5, 0.5, 0.0],
        }
    ]
    path.write_text(json.dumps(rows[0]) + "\n")
    loaded, _ = module._routing_rows_with_sha256(
        path, 1, module.TRI_DATA_SEED, architecture="tri_trajectory"
    )
    assert loaded[0]["routing_target"] == [0.5, 0.5, 0.0]
    geometry, _ = module._routing_rows_with_sha256(
        path, 1, module.TRI_DATA_SEED, architecture="tri_geometry"
    )
    assert geometry == loaded
    hierarchical, _ = module._routing_rows_with_sha256(
        path, 1, module.TRI_DATA_SEED, architecture="tri_hierarchical"
    )
    assert hierarchical == loaded


def test_multi_settings_require_soft_supervision() -> None:
    args = _args()
    args.architecture = "multi_trajectory"
    args.max_rows = module.MULTI_PRESENTATIONS
    args.data_seed = module.MULTI_DATA_SEED
    args.initial_branch_weights = module.MULTI_INITIAL_WEIGHTS
    args.routing_supervision_weight = module.GATE_ROUTING_SUPERVISION_WEIGHT
    module._validate_args(args)
    args.routing_supervision_weight = 0.0
    with pytest.raises(module.Q36MTRTemporalGateTrainingError):
        module._validate_args(args)
    args.routing_supervision_weight = module.GATE_ROUTING_SUPERVISION_WEIGHT
    args.causal_loss_weight = 0.0
    module._validate_args(args)
    args.architecture = "temporal"
    args.max_rows = module.REVISION_PRESENTATIONS
    args.data_seed = module.REVISION_DATA_SEED
    with pytest.raises(module.Q36MTRTemporalGateTrainingError):
        module._validate_args(args)


def test_tri_settings_bind_retention_aware_geometry() -> None:
    args = _args()
    args.architecture = "tri_trajectory"
    args.max_rows = module.TRI_PRESENTATIONS
    args.data_seed = module.TRI_DATA_SEED
    args.initial_branch_weights = module.TRI_INITIAL_WEIGHTS
    args.routing_supervision_weight = module.GATE_ROUTING_SUPERVISION_WEIGHT
    args.causal_loss_weight = 0.0
    module._validate_args(args)
    args.architecture = "tri_geometry"
    module._validate_args(args)
    args.architecture = "tri_hierarchical"
    module._validate_args(args)
    args.architecture = "tri_hierarchical_set_mass"
    module._validate_args(args)
    args.initial_branch_weights = module.MULTI_INITIAL_WEIGHTS
    with pytest.raises(module.Q36MTRTemporalGateTrainingError):
        module._validate_args(args)
