from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from q36_upward_moe_gpt_oss_host import (
    ACTIVE_PARAMETERS,
    CONTROLLED_LAYER_INDICES,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    LAYER_TYPES,
    MODEL_LAYERS,
    NUM_EXPERTS,
    ROUTER_TOP_K,
    TOTAL_PARAMETERS,
    TRAINABLE_PARAMETERS_PER_ROLE,
    Q36UpwardMoEGptOssHostError,
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
        "architectures": ["GptOssForCausalLM"],
        "model_type": "gpt_oss",
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": MODEL_LAYERS,
        "num_local_experts": NUM_EXPERTS,
        "num_experts_per_tok": ROUTER_TOP_K,
        "experts_per_token": ROUTER_TOP_K,
        "intermediate_size": INTERMEDIATE_SIZE,
        "vocab_size": 201_088,
        "sliding_window": 128,
        "max_position_embeddings": 131_072,
        "layer_types": list(LAYER_TYPES),
        "quantization_config": {
            "quant_method": "mxfp4",
            "modules_to_not_convert": [
                "model.layers.*.self_attn",
                "model.layers.*.mlp.router",
                "model.embed_tokens",
                "lm_head",
            ],
        },
    }


def _loaded_host() -> object:
    layers = []
    for _ in range(MODEL_LAYERS):
        router = _named(
            "GptOssTopKRouter",
            top_k=ROUTER_TOP_K,
            num_experts=NUM_EXPERTS,
            hidden_dim=HIDDEN_SIZE,
        )
        experts = _named(
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
        mlp = _named("GptOssMLP", router=router, experts=experts)
        layers.append(
            _named(
                "GptOssDecoderLayer",
                self_attn=_named("GptOssAttention"),
                mlp=mlp,
            )
        )
    config = SimpleNamespace(
        model_type="gpt_oss",
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=MODEL_LAYERS,
        num_local_experts=NUM_EXPERTS,
        num_experts_per_tok=ROUTER_TOP_K,
        intermediate_size=INTERMEDIATE_SIZE,
        layer_types=list(LAYER_TYPES),
    )
    return _named(
        "GptOssForCausalLM", config=config, model=SimpleNamespace(layers=layers)
    )


def test_contract_is_a_distinct_large_sparse_host() -> None:
    contract = static_host_contract()
    assert contract["total_parameters"] == TOTAL_PARAMETERS == 117_000_000_000
    assert contract["active_parameters"] == ACTIVE_PARAMETERS == 5_100_000_000
    assert contract["num_experts"] == 128
    assert contract["router_top_k"] == 4
    assert contract["controlled_layer_indices"] == list(range(20, 36))
    assert TRAINABLE_PARAMETERS_PER_ROLE == 1_658_880
    assert contract["native_router_expert_trainables"] == 0


def test_config_rejects_architecture_router_or_quantization_drift() -> None:
    validate_config_payload(_config())
    for path, value in (
        (("hidden_size",), 4096),
        (("num_local_experts",), 64),
        (("num_experts_per_tok",), 2),
        (("quantization_config", "quant_method"), "fp8"),
    ):
        payload = deepcopy(_config())
        target = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(Q36UpwardMoEGptOssHostError, match="config differs"):
            validate_config_payload(payload)


def test_live_surface_binds_every_native_mxfp4_moe_layer() -> None:
    receipt = validate_loaded_surface(_loaded_host())
    assert receipt["model_layers"] == MODEL_LAYERS
    assert receipt["moe_layers"] == MODEL_LAYERS
    assert receipt["controlled_layer_indices"] == list(CONTROLLED_LAYER_INDICES)
    assert receipt["quantization_method"] == "mxfp4"
    assert len(receipt["native_topology_sha256"]) == 64


def test_live_surface_rejects_missing_precision_metadata() -> None:
    host = _loaded_host()
    host.model.layers[-1].mlp.experts.down_proj_precision_config = None
    with pytest.raises(Q36UpwardMoEGptOssHostError, match="layer differs"):
        validate_loaded_surface(host)
