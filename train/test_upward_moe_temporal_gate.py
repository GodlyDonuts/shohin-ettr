from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import upward_moe_temporal_gate as module


class _Native(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.scale


class _Layer(nn.Module):
    def __init__(self, hidden_size: int, surface: str) -> None:
        super().__init__()
        setattr(self, surface, _Native(hidden_size))


class _Backbone(nn.Module):
    def __init__(self, layers: int, hidden_size: int, surface: str) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_Layer(hidden_size, surface) for _ in range(layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            native = getattr(layer, "mlp", getattr(layer, "mixer", None))
            hidden_states = native(hidden_states)
        return hidden_states


def _states(spec: module.UpwardMoETemporalGateSpec, offset: float):
    a = torch.full((spec.rank, spec.hidden_size), 0.01 + offset)
    b = torch.full((spec.hidden_size, spec.rank), 0.02 + offset)
    state = {}
    for index in spec.controlled_layer_indices:
        prefix = f"backbone.model.layers.{index}.{spec.module_attribute}"
        state[f"{prefix}.adapter_a.weight"] = a
        state[f"{prefix}.adapter_b.weight"] = b
    return state


@pytest.mark.parametrize(
    ("spec", "model_class", "validator_name"),
    [
        (
            module.NEMOTRON_SPEC,
            module.NemotronSuperTemporalGateModel,
            "validate_nemotron_surface",
        ),
        (
            module.MIXTRAL_SPEC,
            module.MixtralTemporalGateModel,
            "validate_mixtral_surface",
        ),
    ],
)
def test_exact_upward_host_installs_only_temporal_gate(
    monkeypatch, spec, model_class, validator_name
) -> None:
    layers = max(spec.controlled_layer_indices) + 1
    backbone = _Backbone(layers, spec.hidden_size, spec.module_attribute)
    monkeypatch.setattr(
        module,
        validator_name,
        lambda _: {"attachment_surface": spec.attachment_surface},
    )
    model = model_class(backbone, _states(spec, 0.0), _states(spec, 0.01))
    assert model.trainable_parameter_count() == spec.gate_trainable_parameters
    assert len(model.blocks) == len(spec.controlled_layer_indices)
    assert all(
        not parameter.requires_grad
        for block in model.blocks
        for parameter in block.base.parameters()
    )
    assert len(model.trainable_state()) == len(spec.controlled_layer_indices) * 2
    assert len(model.trainable_state_sha256()) == 64
    receipt = model.receipt()
    assert receipt["architecture"] == spec.architecture
    assert receipt["attachment_surface"] == spec.attachment_surface
    assert receipt["native_router_expert_trainables"] == 0


def test_transfer_contract_pins_causal_only_lineage() -> None:
    contract = module.static_transfer_contract()
    assert contract["causal_loss_weight"] == 1.0
    assert contract["routing_supervision_weight"] == 0.0
    assert contract["frozen_trajectories"] == ["owner", "aligned_revision"]
    assert [host["host"] for host in contract["hosts"]] == [
        "Nemotron-Super-120B-A12B",
        "Mixtral-8x22B-141B-A39B",
    ]
    assert [host["gate_trainable_parameters"] for host in contract["hosts"]] == [
        65552,
        98320,
    ]


def test_wrong_host_surface_or_state_fails_closed(monkeypatch) -> None:
    spec = module.NEMOTRON_SPEC
    layers = max(spec.controlled_layer_indices) + 1
    backbone = _Backbone(layers, spec.hidden_size, spec.module_attribute)
    monkeypatch.setattr(
        module,
        "validate_nemotron_surface",
        lambda _: {"attachment_surface": "post-mlp-residual"},
    )
    with pytest.raises(module.UpwardMoETemporalGateError):
        module.NemotronSuperTemporalGateModel(
            backbone, _states(spec, 0.0), _states(spec, 0.01)
        )

    monkeypatch.setattr(
        module,
        "validate_nemotron_surface",
        lambda _: {"attachment_surface": spec.attachment_surface},
    )
    cross_surface = {
        name.replace(".mixer.", ".mlp."): value
        for name, value in _states(spec, 0.0).items()
    }
    with pytest.raises(module.UpwardMoETemporalGateError):
        module.NemotronSuperTemporalGateModel(
            backbone, cross_surface, _states(spec, 0.01)
        )
