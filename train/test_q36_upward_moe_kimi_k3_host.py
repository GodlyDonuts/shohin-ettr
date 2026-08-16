from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from q36_upward_moe_kimi_k3_host import (
    CONTROLLED_LAYER_INDICES,
    HIDDEN_SIZE,
    MINIMUM_H100S,
    MINIMUM_NODES,
    MODEL_LAYERS,
    NUM_EXPERTS,
    Q36UpwardMoEKimiK3HostError,
    ROUTED_EXPERT_HIDDEN_SIZE,
    ROUTER_TOP_K,
    TEMPORAL_GATE_PARAMETERS,
    TRAINABLE_PARAMETERS_PER_ROLE,
    static_host_contract,
    validate_config_payload,
    validate_loaded_surface,
)


def _named(name: str, **values: object) -> object:
    value = type(name, (), {})()
    for key, member in values.items():
        setattr(value, key, member)
    return value


def _config() -> dict:
    return {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "model_type": "kimi_k3",
        "text_config": {
            "architectures": ["KimiLinearForCausalLM"],
            "model_type": "kimi_linear",
            "hidden_size": HIDDEN_SIZE,
            "num_hidden_layers": MODEL_LAYERS,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "num_experts": NUM_EXPERTS,
            "num_experts_per_token": ROUTER_TOP_K,
            "num_shared_experts": 2,
            "moe_intermediate_size": 3072,
            "routed_expert_hidden_size": ROUTED_EXPERT_HIDDEN_SIZE,
            "moe_renormalize": True,
            "moe_router_activation_func": "sigmoid",
            "routed_scaling_factor": 1.0,
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "mxfp4-pack-quantized",
            },
        },
    }


def _sparse() -> object:
    expert = _named("KimiBlockSparseMLP")
    return _named(
        "KimiSparseMoeBlock",
        hidden_dim=HIDDEN_SIZE,
        num_experts=NUM_EXPERTS,
        top_k=ROUTER_TOP_K,
        experts=[expert] * NUM_EXPERTS,
        gate=_named("KimiMoEGate"),
        shared_experts=_named("KimiMLP"),
        routed_expert_down_proj=SimpleNamespace(
            in_features=HIDDEN_SIZE, out_features=ROUTED_EXPERT_HIDDEN_SIZE
        ),
        routed_expert_up_proj=SimpleNamespace(
            in_features=ROUTED_EXPERT_HIDDEN_SIZE, out_features=HIDDEN_SIZE
        ),
    )


def _loaded_host() -> object:
    layers = [_named("KimiDecoderLayer", mlp=_named("KimiMLP"))]
    layers.extend(
        _named("KimiDecoderLayer", block_sparse_moe=_sparse())
        for _ in range(1, MODEL_LAYERS)
    )
    text_config = SimpleNamespace(model_type="kimi_linear", hidden_size=HIDDEN_SIZE)
    config = SimpleNamespace(model_type="kimi_k3", text_config=text_config)
    language_model = _named(
        "KimiLinearForCausalLM", model=SimpleNamespace(layers=layers)
    )
    return _named(
        "KimiK3ForConditionalGeneration",
        config=config,
        language_model=language_model,
    )


def test_kimi_contract_is_true_upward_total_and_active_scale() -> None:
    contract = static_host_contract()
    assert contract["total_parameters"] == 2_800_000_000_000
    assert contract["active_parameters"] == 104_000_000_000
    assert contract["minimum_h100s"] == MINIMUM_H100S == 24
    assert contract["minimum_nodes"] == MINIMUM_NODES == 3
    assert contract["controlled_layer_indices"] == list(range(77, 93))
    assert TRAINABLE_PARAMETERS_PER_ROLE == 4_128_768
    assert TEMPORAL_GATE_PARAMETERS == 114_704
    assert contract["native_router_expert_trainables"] == 0


def test_kimi_config_rejects_architecture_or_router_drift() -> None:
    payload = _config()
    validate_config_payload(payload)
    for path, changed in (
        (("text_config", "hidden_size"), 8192),
        (("text_config", "num_experts"), 512),
        (("text_config", "num_experts_per_token"), 8),
        (("text_config", "first_k_dense_replace"), 0),
        (("text_config", "quantization_config", "format"), "fp8"),
    ):
        drifted = deepcopy(payload)
        target = drifted
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = changed
        with pytest.raises(Q36UpwardMoEKimiK3HostError, match="config differs"):
            validate_config_payload(drifted)


def test_kimi_live_surface_binds_dense_prefix_and_all_sparse_layers() -> None:
    receipt = validate_loaded_surface(_loaded_host())
    assert receipt["model_layers"] == 93
    assert receipt["moe_layers"] == 92
    assert receipt["controlled_layer_indices"] == list(CONTROLLED_LAYER_INDICES)
    assert receipt["native_execution_mode"] == "eval"
    assert len(receipt["native_topology_sha256"]) == 64


def test_kimi_live_surface_rejects_expert_or_projection_drift() -> None:
    host = _loaded_host()
    sparse = host.language_model.model.layers[
        CONTROLLED_LAYER_INDICES[0]
    ].block_sparse_moe
    sparse.top_k -= 1
    with pytest.raises(Q36UpwardMoEKimiK3HostError, match="layer differs"):
        validate_loaded_surface(host)
