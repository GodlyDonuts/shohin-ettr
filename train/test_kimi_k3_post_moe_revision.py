from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from kimi_k3_post_moe_revision import KimiK3RevisionError, KimiK3RevisionModel
from q36_upward_moe_kimi_k3_host import (
    CONTROLLED_LAYER_INDICES,
    HIDDEN_SIZE,
    MODEL_LAYERS,
    NUM_EXPERTS,
    ROUTED_EXPERT_HIDDEN_SIZE,
    ROUTER_TOP_K,
    TRAINABLE_PARAMETERS_PER_ROLE,
)


def _named(name: str, **values: object) -> object:
    value = type(name, (), {})()
    for key, member in values.items():
        setattr(value, key, member)
    return value


class _NativeSparse(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.hidden_dim = HIDDEN_SIZE
        self.num_experts = NUM_EXPERTS
        self.top_k = ROUTER_TOP_K
        self.experts = [_named("KimiBlockSparseMLP")] * NUM_EXPERTS
        self.gate = _named("KimiMoEGate")
        self.shared_experts = _named("KimiMLP")
        self.routed_expert_down_proj = SimpleNamespace(
            in_features=HIDDEN_SIZE, out_features=ROUTED_EXPERT_HIDDEN_SIZE
        )
        self.routed_expert_up_proj = SimpleNamespace(
            in_features=ROUTED_EXPERT_HIDDEN_SIZE, out_features=HIDDEN_SIZE
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise NotImplementedError("native Kimi sparse training is forbidden")
        return hidden_states * self.weight


def _backbone() -> nn.Module:
    class _Layer(nn.Module):
        def __init__(self, index: int) -> None:
            super().__init__()
            if index == 0:
                self.mlp = type("KimiMLP", (nn.Identity,), {})()
            else:
                self.block_sparse_moe = type(
                    "KimiSparseMoeBlock", (_NativeSparse,), {}
                )()

    _Layer.__name__ = "KimiDecoderLayer"

    class _TextModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(_Layer(index) for index in range(MODEL_LAYERS))

    class _LanguageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _TextModel()

    _LanguageModel.__name__ = "KimiLinearForCausalLM"

    class _Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                model_type="kimi_k3",
                text_config=SimpleNamespace(
                    model_type="kimi_linear", hidden_size=HIDDEN_SIZE
                ),
            )
            self.language_model = _LanguageModel()

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            for index in CONTROLLED_LAYER_INDICES:
                hidden_states = self.language_model.model.layers[
                    index
                ].block_sparse_moe(hidden_states)
            return hidden_states

    _Backbone.__name__ = "KimiK3ForConditionalGeneration"
    return _Backbone()


def test_kimi_attachment_matches_frozen_post_moe_geometry() -> None:
    model = KimiK3RevisionModel(_backbone())
    assert model.trainable_parameter_count() == TRAINABLE_PARAMETERS_PER_ROLE
    assert len(model.blocks) == 16
    assert len(model.trainable_state()) == 32
    assert model.receipt()["native_router_expert_trainables"] == 0
    assert model.receipt()["native_execution_mode"] == "eval"


def test_model_train_keeps_every_native_sparse_block_in_eval() -> None:
    model = KimiK3RevisionModel(_backbone())
    model.train()
    assert model.training is True
    assert all(block.training is True for block in model.blocks)
    assert all(block.base.training is False for block in model.blocks)
    value = torch.ones(1, 1, HIDDEN_SIZE)
    observed = model(value)
    assert torch.equal(observed, value)
    observed.float().sum().backward()
    assert all(block.adapter_b.weight.grad is not None for block in model.blocks)


def test_kimi_attachment_fails_if_native_block_enters_training() -> None:
    model = KimiK3RevisionModel(_backbone())
    model.blocks[0].base.train()
    with pytest.raises(KimiK3RevisionError, match="entered training mode"):
        model.blocks[0](torch.ones(1, 1, HIDDEN_SIZE))
