from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from nemotron_ultra_post_mixer_revision import (
    NemotronUltraRevisionError,
    NemotronUltraRevisionModel,
)
from q36_upward_moe_ultra_host import (
    CONTROLLED_LAYER_INDICES,
    HIDDEN_SIZE,
    LAYER_TYPES,
    MOE_INTERMEDIATE_SIZE,
    MOE_LATENT_SIZE,
    NUM_EXPERTS,
    ROUTER_TOP_K,
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


def _mixer() -> nn.Module:
    mixer = type("NemotronHMoE", (_NativeMixer,), {})()
    mixer.gate = _named(
        "NemotronHTopkRouter",
        top_k=ROUTER_TOP_K,
        n_routed_experts=NUM_EXPERTS,
        n_group=1,
        topk_group=1,
    )
    expert = _named(
        "NemotronHMLP", hidden_size=HIDDEN_SIZE, intermediate_size=MOE_INTERMEDIATE_SIZE
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


def _backbone() -> nn.Module:
    class _Layer(nn.Module):
        def __init__(self, block_type: str, mixer: nn.Module) -> None:
            super().__init__()
            self.block_type = block_type
            self.mixer = mixer

    layers: list[_Layer] = []
    for layer_type in LAYER_TYPES:
        if layer_type == "moe":
            mixer = _mixer()
        else:
            cls = (
                "NemotronHMamba2Mixer"
                if layer_type == "mamba"
                else "NemotronHAttention"
            )
            mixer = type(cls, (_NativeMixer,), {})()
        layers.append(_Layer(layer_type, mixer))

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(layers)

    class _Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                model_type="nemotron_h",
                hidden_size=HIDDEN_SIZE,
                layers_block_type=LAYER_TYPES,
            )
            self.model = _Model()

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            for layer in self.model.layers:
                hidden_states = layer.mixer(hidden_states)
            return hidden_states

    _Backbone.__name__ = "NemotronHForCausalLM"
    return _Backbone()


def test_ultra_attachment_matches_lifted_checkpoint_geometry() -> None:
    model = NemotronUltraRevisionModel(_backbone())
    assert model.trainable_parameter_count() == TRAINABLE_PARAMETERS_PER_ROLE
    assert len(model.blocks) == 16
    assert len(model.trainable_state()) == 32
    assert set(model.trainable_state()) == {
        f"backbone.model.layers.{layer}.mixer.adapter_{kind}.weight"
        for layer in CONTROLLED_LAYER_INDICES
        for kind in ("a", "b")
    }


def test_zero_initialized_ultra_residual_preserves_native_then_receipts() -> None:
    model = NemotronUltraRevisionModel(_backbone())
    value = torch.ones(1, 1, HIDDEN_SIZE)
    observed = model(value)
    assert torch.equal(observed, value)
    observed.float().sum().backward()
    assert all(block.adapter_b.weight.grad is not None for block in model.blocks)
    assert model.receipt()["layers"][0]["tokens"] == 1


def test_ultra_attachment_rejects_non_moe_controlled_layer() -> None:
    backbone = _backbone()
    index = CONTROLLED_LAYER_INDICES[-1]
    backbone.model.layers[index].block_type = "attention"
    backbone.model.layers[index].mixer = type(
        "NemotronHAttention", (_NativeMixer,), {}
    )()
    with pytest.raises(NemotronUltraRevisionError):
        NemotronUltraRevisionModel(backbone)
