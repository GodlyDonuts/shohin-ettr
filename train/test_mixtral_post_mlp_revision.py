from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mixtral_post_mlp_revision import MixtralRevisionError, MixtralRevisionModel
from q36_upward_moe_mixtral_host import (
    CONTROLLED_LAYER_INDICES,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    MODEL_LAYERS,
    NUM_EXPERTS,
    ROUTER_TOP_K,
    TRAINABLE_PARAMETERS_PER_ROLE,
)


def _named(name: str, **members: object) -> object:
    value = type(name, (), {})()
    for key, member in members.items():
        setattr(value, key, member)
    return value


class _NativeMoe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.top_k = ROUTER_TOP_K
        self.gate = _named(
            "MixtralTopKRouter",
            top_k=ROUTER_TOP_K,
            num_experts=NUM_EXPERTS,
            hidden_dim=HIDDEN_SIZE,
        )
        self.experts = _named(
            "MixtralExperts",
            num_experts=NUM_EXPERTS,
            hidden_dim=HIDDEN_SIZE,
            intermediate_dim=INTERMEDIATE_SIZE,
            gate_up_proj=SimpleNamespace(
                shape=(NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
            ),
            down_proj=SimpleNamespace(
                shape=(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.weight


_NativeMoe.__name__ = "MixtralSparseMoeBlock"


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = type("MixtralAttention", (), {})()
        self.mlp = _NativeMoe()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


_Layer.__name__ = "MixtralDecoderLayer"


class _TextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(MODEL_LAYERS)])


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="mixtral",
            hidden_size=HIDDEN_SIZE,
            num_hidden_layers=MODEL_LAYERS,
            num_local_experts=NUM_EXPERTS,
            num_experts_per_tok=ROUTER_TOP_K,
            intermediate_size=INTERMEDIATE_SIZE,
        )
        self.model = _TextModel()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


_Backbone.__name__ = "MixtralForCausalLM"


def test_exact_final_moe_surface_is_attached_and_trainable_only() -> None:
    model = MixtralRevisionModel(_Backbone())
    assert model.trainable_parameter_count() == TRAINABLE_PARAMETERS_PER_ROLE
    assert len(model.blocks) == 16
    assert len(model.trainable_state()) == 32
    assert all(
        type(model.backbone.model.layers[index].mlp).__name__ == "SharedPostMLPResidual"
        for index in CONTROLLED_LAYER_INDICES
    )
    assert not any(
        parameter.requires_grad
        for block in model.blocks
        for parameter in block.base.parameters()
    )


def test_zero_residual_preserves_native_forward_and_has_gradients() -> None:
    model = MixtralRevisionModel(_Backbone())
    value = torch.ones(1, 1, HIDDEN_SIZE)
    observed = model(value)
    assert torch.equal(observed, value)
    observed.float().sum().backward()
    assert all(block.adapter_b.weight.grad is not None for block in model.blocks)
    assert model.receipt()["layers"][0]["tokens"] == 1


def test_non_mixtral_controlled_surface_fails_closed() -> None:
    backbone = _Backbone()
    backbone.model.layers[CONTROLLED_LAYER_INDICES[-1]].mlp = nn.Linear(
        HIDDEN_SIZE, HIDDEN_SIZE, bias=False
    )
    with pytest.raises(MixtralRevisionError):
        MixtralRevisionModel(backbone)
