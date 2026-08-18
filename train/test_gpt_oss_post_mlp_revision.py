from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from gpt_oss_post_mlp_revision import GptOssRevisionError, GptOssRevisionModel
from q36_upward_moe_gpt_oss_host import (
    CONTROLLED_LAYER_INDICES,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    LAYER_TYPES,
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


class _NativeMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.router = _named(
            "GptOssTopKRouter",
            top_k=ROUTER_TOP_K,
            num_experts=NUM_EXPERTS,
            hidden_dim=HIDDEN_SIZE,
        )
        self.experts = _named(
            "Mxfp4GptOssExperts",
            num_experts=NUM_EXPERTS,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            gate_up_proj_bias=SimpleNamespace(
                shape=(NUM_EXPERTS, 2 * INTERMEDIATE_SIZE)
            ),
            down_proj_bias=SimpleNamespace(shape=(NUM_EXPERTS, HIDDEN_SIZE)),
            gate_up_proj_precision_config=object(),
            down_proj_precision_config=object(),
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Emulate a frozen inference kernel with no input autograd.  The decoder
        # residual path must still carry gradients to every Shohin block.
        native = hidden_states.detach() * self.anchor.detach()
        return native, torch.ones((*hidden_states.shape[:-1], ROUTER_TOP_K))


_NativeMlp.__name__ = "GptOssMLP"


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _named("GptOssAttention")
        self.mlp = _NativeMlp()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        native, _ = self.mlp(hidden_states)
        return hidden_states + native


_Layer.__name__ = "GptOssDecoderLayer"


class _TextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(MODEL_LAYERS)])


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="gpt_oss",
            hidden_size=HIDDEN_SIZE,
            num_hidden_layers=MODEL_LAYERS,
            num_local_experts=NUM_EXPERTS,
            num_experts_per_tok=ROUTER_TOP_K,
            intermediate_size=INTERMEDIATE_SIZE,
            layer_types=list(LAYER_TYPES),
        )
        self.model = _TextModel()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


_Backbone.__name__ = "GptOssForCausalLM"


def test_exact_final_moe_surface_is_attached_and_native_weights_are_frozen() -> None:
    model = GptOssRevisionModel(_Backbone())
    assert model.trainable_parameter_count() == TRAINABLE_PARAMETERS_PER_ROLE
    assert len(model.blocks) == 16
    assert len(model.trainable_state()) == 32
    assert all(
        type(model.backbone.model.layers[index].mlp).__name__ == "GptOssPostMLPResidual"
        for index in CONTROLLED_LAYER_INDICES
    )
    assert not any(
        parameter.requires_grad
        for block in model.blocks
        for parameter in block.base.parameters()
    )


def test_residual_path_propagates_to_earliest_and_latest_blocks() -> None:
    model = GptOssRevisionModel(_Backbone())
    value = torch.ones(1, 1, HIDDEN_SIZE)
    observed = model(value)
    observed.float().sum().backward()
    assert model.blocks[0].adapter_b.weight.grad is not None
    assert model.blocks[-1].adapter_b.weight.grad is not None
    assert float(model.blocks[0].adapter_b.weight.grad.norm()) > 0
    assert float(model.blocks[-1].adapter_b.weight.grad.norm()) > 0


def test_non_mxfp4_surface_fails_closed() -> None:
    backbone = _Backbone()
    backbone.model.layers[-1].mlp.experts = _named("GptOssExperts")
    with pytest.raises(GptOssRevisionError, match="layer differs"):
        GptOssRevisionModel(backbone)
