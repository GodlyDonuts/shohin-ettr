"""Tests for the pinned Nemotron Super role-owned residual surface."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from nemotron_super_post_mixer_revision import (
    ACTIVATION_GRADIENT_BOUNDARY,
    ACTIVATION_GRADIENT_MAX_L2_NORM,
    NemotronSuperRevisionError,
    NemotronSuperRevisionModel,
    _bounded_activation_gradient,
)
from q36_upward_moe_host import (
    CONTROLLED_LAYER_INDICES,
    HIDDEN_SIZE,
    LAYER_TYPES,
    MODEL_LAYERS,
    MODEL_TYPE,
    MOE_INTERMEDIATE_SIZE,
    MOE_LATENT_SIZE,
    NUM_EXPERTS,
    ROUTER_GROUPS,
    ROUTER_TOP_K,
    ROUTER_TOP_K_GROUPS,
    SHARED_EXPERT_INTERMEDIATE_SIZE,
    TRAINABLE_PARAMETERS_PER_ROLE,
)


def _named(name: str, **values: object) -> object:
    value = type(name, (), {})()
    for key, member in values.items():
        setattr(value, key, member)
    return value


class _NativeMixer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.weight


def _moe_mixer(*, modelopt_quantized: bool = False) -> nn.Module:
    mixer = type(
        "QuantNemotronHMoE" if modelopt_quantized else "NemotronHMoE",
        (_NativeMixer,),
        {},
    )()
    mixer.gate = _named(
        "NemotronHTopkRouter",
        top_k=ROUTER_TOP_K,
        n_routed_experts=NUM_EXPERTS,
        n_group=ROUTER_GROUPS,
        topk_group=ROUTER_TOP_K_GROUPS,
    )
    expert = _named(
        "NemotronHMLP",
        hidden_size=HIDDEN_SIZE,
        intermediate_size=MOE_INTERMEDIATE_SIZE,
    )
    mixer.experts = [expert] * NUM_EXPERTS
    mixer.shared_experts = _named(
        "NemotronHMLP",
        hidden_size=HIDDEN_SIZE,
        intermediate_size=SHARED_EXPERT_INTERMEDIATE_SIZE,
    )
    mixer.fc1_latent_proj = SimpleNamespace(
        in_features=HIDDEN_SIZE, out_features=MOE_LATENT_SIZE
    )
    mixer.fc2_latent_proj = SimpleNamespace(
        in_features=MOE_LATENT_SIZE, out_features=HIDDEN_SIZE
    )
    return mixer


def _backbone(*, modelopt_quantized: bool = False) -> nn.Module:
    layers = []
    for block_type in LAYER_TYPES:
        if block_type == "moe":
            mixer = _moe_mixer(modelopt_quantized=modelopt_quantized)
        elif block_type == "mamba":
            mixer = type("NemotronHMamba2Mixer", (_NativeMixer,), {})()
        else:
            mixer = type(
                (
                    "QuantNemotronHAttention"
                    if modelopt_quantized
                    else "NemotronHAttention"
                ),
                (_NativeMixer,),
                {},
            )()
        layers.append(SimpleNamespace(block_type=block_type, mixer=mixer))

    class _Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                model_type=MODEL_TYPE,
                hidden_size=HIDDEN_SIZE,
                num_hidden_layers=MODEL_LAYERS,
                layers_block_type=LAYER_TYPES,
            )
            self.model = SimpleNamespace(layers=layers)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            for layer in self.model.layers:
                hidden_states = layer.mixer(hidden_states)
            return hidden_states

    _Backbone.__name__ = "NemotronHForCausalLM"
    return _Backbone()


def test_exact_final_moe_surface_is_attached_and_trainable_only() -> None:
    model = NemotronSuperRevisionModel(_backbone())
    assert model.trainable_parameter_count() == TRAINABLE_PARAMETERS_PER_ROLE
    assert len(model.blocks) == 16
    assert len(model.trainable_state()) == 32
    assert all(
        type(model.backbone.model.layers[index].mixer).__name__ == "PostMixerResidual"
        for index in CONTROLLED_LAYER_INDICES
    )
    assert not any(
        parameter.requires_grad
        for block in model.blocks
        for parameter in block.base.parameters()
    )


def test_exact_modelopt_moe_surface_is_attached_only_when_declared() -> None:
    backbone = _backbone(modelopt_quantized=True)
    with pytest.raises(NemotronSuperRevisionError, match="layer differs"):
        NemotronSuperRevisionModel(backbone)

    model = NemotronSuperRevisionModel(backbone, modelopt_quantized=True)
    assert model.trainable_parameter_count() == TRAINABLE_PARAMETERS_PER_ROLE


def test_zero_initialized_residual_preserves_native_forward_then_updates() -> None:
    model = NemotronSuperRevisionModel(_backbone())
    value = torch.ones(1, 1, HIDDEN_SIZE)
    native = value.clone()
    observed = model(value)
    assert torch.equal(observed, native)
    loss = observed.float().sum()
    loss.backward()
    assert all(block.adapter_b.weight.grad is not None for block in model.blocks)
    assert model.receipt()["layers"][0]["tokens"] == 1
    assert model.receipt()["activation_gradient_boundary"] == {
        "mode": ACTIVATION_GRADIENT_BOUNDARY,
        "maximum_l2_norm": ACTIVATION_GRADIENT_MAX_L2_NORM,
        "training_only": True,
        "forward_values_unchanged": True,
    }


def test_activation_gradient_boundary_preserves_direction_and_caps_norm() -> None:
    gradient = torch.tensor([3.0, 4.0])
    bounded = _bounded_activation_gradient(gradient)
    assert torch.allclose(bounded, torch.tensor([0.6, 0.8]))
    assert float(bounded.norm()) == pytest.approx(ACTIVATION_GRADIENT_MAX_L2_NORM)
    small = torch.tensor([0.3, 0.4])
    assert torch.equal(_bounded_activation_gradient(small), small)
    with pytest.raises(NemotronSuperRevisionError, match="nonfinite"):
        _bounded_activation_gradient(torch.tensor([float("nan")]))


def test_non_moe_controlled_surface_fails_closed() -> None:
    backbone = _backbone()
    index = CONTROLLED_LAYER_INDICES[-1]
    backbone.model.layers[index].block_type = "attention"
    backbone.model.layers[index].mixer = type(
        "NemotronHAttention", (_NativeMixer,), {}
    )()
    with pytest.raises(NemotronSuperRevisionError):
        NemotronSuperRevisionModel(backbone)
