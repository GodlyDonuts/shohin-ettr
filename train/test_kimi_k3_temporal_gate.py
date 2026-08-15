from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import kimi_k3_temporal_gate as module
from q36_upward_moe_kimi_k3_host import (
    CONTROLLED_LAYER_INDICES,
    HIDDEN_SIZE,
    MODEL_LAYERS,
    NUM_EXPERTS,
    RANK,
    ROUTED_EXPERT_HIDDEN_SIZE,
    ROUTER_TOP_K,
    TEMPORAL_GATE_PARAMETERS,
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


def _state(offset: float) -> dict[str, torch.Tensor]:
    state = {}
    for index in CONTROLLED_LAYER_INDICES:
        prefix = f"{module.ROLE_STATE_PREFIX}.{index}.{module.MODULE_ATTRIBUTE}"
        state[f"{prefix}.adapter_a.weight"] = torch.full(
            (RANK, HIDDEN_SIZE), 0.001 + offset, dtype=torch.float32
        )
        state[f"{prefix}.adapter_b.weight"] = torch.full(
            (HIDDEN_SIZE, RANK), 0.002 + offset, dtype=torch.float32
        )
    return state


def test_kimi_temporal_gate_is_causal_only_and_exactly_trainable() -> None:
    model = module.KimiK3TemporalGateModel(_backbone(), _state(0.0), _state(0.001))
    assert model.trainable_parameter_count() == TEMPORAL_GATE_PARAMETERS == 114_704
    assert len(model.blocks) == 16
    assert len(model.trainable_state()) == 32
    assert len(model.trainable_state_sha256()) == 64
    receipt = model.receipt()
    assert receipt["causal_loss_weight"] == 1.0
    assert receipt["routing_supervision_weight"] == 0.0
    assert receipt["frozen_trajectories"] == ["owner", "aligned_revision"]
    assert receipt["native_router_expert_trainables"] == 0


def test_kimi_temporal_training_keeps_all_native_sparse_blocks_in_eval() -> None:
    model = module.KimiK3TemporalGateModel(_backbone(), _state(0.0), _state(0.001))
    model.train()
    assert model.training is True
    assert all(block.training is True for block in model.blocks)
    assert all(block.base.training is False for block in model.blocks)
    value = torch.ones(1, 1, HIDDEN_SIZE)
    observed = model(value)
    observed.float().sum().backward()
    assert observed.shape == value.shape
    assert all(block.gate_weight.grad is not None for block in model.blocks)


def test_kimi_temporal_rejects_cross_surface_or_nonfinite_role_state() -> None:
    owner = _state(0.0)
    revision = _state(0.001)
    wrong_name = next(iter(owner))
    owner[wrong_name.replace("block_sparse_moe", "mlp")] = owner.pop(wrong_name)
    with pytest.raises(module.KimiK3TemporalGateError, match="state names differ"):
        module.KimiK3TemporalGateModel(_backbone(), owner, revision)

    owner = _state(0.0)
    owner[next(iter(owner))][0, 0] = float("nan")
    with pytest.raises(module.KimiK3TemporalGateError, match="tensor differs"):
        module.KimiK3TemporalGateModel(_backbone(), owner, revision)


def test_kimi_temporal_fails_if_native_moe_is_forced_to_train() -> None:
    model = module.KimiK3TemporalGateModel(_backbone(), _state(0.0), _state(0.001))
    model.blocks[0].base.train()
    with pytest.raises(module.KimiK3TemporalGateError, match="entered training mode"):
        model(torch.ones(1, 1, HIDDEN_SIZE))
