from pathlib import Path

import pytest
import torch

import hf_upward_moe_train_temporal_gate as module
from upward_moe_role_lineage import trainable_state_sha256
from upward_moe_temporal_gate import UpwardMoETemporalGateSpec

SPEC = UpwardMoETemporalGateSpec(
    host="test-large-moe",
    model_revision="1" * 40,
    model_config_sha256="2" * 64,
    architecture="test-causal-temporal",
    attachment_surface="post-mixer-residual",
    module_attribute="mixer",
    hidden_size=2,
    rank=1,
    alpha=1.0,
    controlled_layer_indices=(1, 3),
    require_final_contiguous=False,
)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gates = torch.nn.ModuleList(
            [
                torch.nn.Linear(SPEC.hidden_size, 1, bias=True, dtype=torch.float32)
                for _ in SPEC.controlled_layer_indices
            ]
        )

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def trainable_state(self):
        return {
            name.replace("weight", "gate_weight")
            .replace("bias", "gate_bias"): value.detach()
            .cpu()
            .clone()
            for name, value in super().named_parameters()
        }

    def trainable_state_sha256(self) -> str:
        return trainable_state_sha256(self.trainable_state())

    def named_parameters(self, *args, **kwargs):
        for name, value in super().named_parameters(*args, **kwargs):
            yield name.replace("weight", "gate_weight").replace(
                "bias", "gate_bias"
            ), value


def test_static_gate_contract_is_causal_only_and_upward() -> None:
    contract = module.static_gate_contract()
    assert contract["updates"] == 256
    assert contract["consumed_presentations"] == 2048
    assert contract["causal_loss_weight"] == 1.0
    assert contract["routing_supervision_weight"] == 0.0
    assert contract["native_router_expert_trainables"] == 0
    assert [host["host"] for host in contract["hosts"]] == [
        "Nemotron-Super-120B-A12B",
        "Mixtral-8x22B-141B-A39B",
        "Nemotron-Ultra-550B-A55B",
    ]


def test_gate_checkpoint_round_trip_is_trainable_only(tmp_path: Path) -> None:
    model = _Model()
    final = model.trainable_state_sha256()
    metadata = {
        "schema": module.SCHEMA,
        "final_trainable_state_sha256": final,
        "role_receipt": {"warm_start_exact": True},
    }
    checkpoint = tmp_path / "gate.pt"
    digest = module.save_gate_checkpoint(checkpoint, model, metadata, SPEC)
    assert len(digest) == 64
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert set(payload) == {"schema", "update", "trainable_state", "metadata"}
    assert payload["schema"] == module.CHECKPOINT_SCHEMA
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    assert module.restore_gate_checkpoint(checkpoint, model, SPEC) == metadata
    assert model.trainable_state_sha256() == final


def test_gate_state_rejects_native_or_nonfinite_tensors() -> None:
    state = _Model().trainable_state()
    assert module._validate_gate_state(state, SPEC) is state
    state["native_router.weight"] = torch.ones(1)
    with pytest.raises(module.UpwardMoETemporalTrainingError):
        module._validate_gate_state(state, SPEC)


def test_host_aliases_exclude_small_moes() -> None:
    assert module.host_spec("nemotron-super").host == "Nemotron-Super-120B-A12B"
    assert module.host_spec("mixtral-8x22b").host == "Mixtral-8x22B-141B-A39B"
    with pytest.raises(module.UpwardMoETemporalTrainingError):
        module.host_spec("olmoe-small")
