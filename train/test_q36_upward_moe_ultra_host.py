from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from q36_upward_moe_ultra_host import (
    CONTROLLED_LAYER_INDICES,
    HIDDEN_SIZE,
    LAYER_TYPES,
    MODEL_LAYERS,
    MOE_INTERMEDIATE_SIZE,
    MOE_LATENT_SIZE,
    NUM_EXPERTS,
    Q36UpwardMoEUltraHostError,
    ROUTER_TOP_K,
    SHARED_EXPERT_INTERMEDIATE_SIZE,
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
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "hidden_size": HIDDEN_SIZE,
        "layers_block_type": list(LAYER_TYPES),
        "n_routed_experts": NUM_EXPERTS,
        "num_experts_per_tok": ROUTER_TOP_K,
        "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
        "moe_shared_expert_intermediate_size": SHARED_EXPERT_INTERMEDIATE_SIZE,
        "moe_latent_size": MOE_LATENT_SIZE,
        "routed_scaling_factor": 5.0,
        "n_group": 1,
        "topk_group": 1,
        "mtp_layers_block_type": ["attention", "moe"],
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "producer": {"name": "modelopt", "version": "1.0.0"},
        },
    }


def _loaded_host() -> object:
    expert = _named(
        "NemotronHMLP", hidden_size=HIDDEN_SIZE, intermediate_size=MOE_INTERMEDIATE_SIZE
    )
    layers = []
    for layer_type in LAYER_TYPES:
        if layer_type == "moe":
            mixer = _named(
                "NemotronHMoE",
                gate=_named(
                    "NemotronHTopkRouter",
                    top_k=ROUTER_TOP_K,
                    n_routed_experts=NUM_EXPERTS,
                    n_group=1,
                    topk_group=1,
                ),
                experts=[expert] * NUM_EXPERTS,
                shared_experts=_named(
                    "NemotronHMLP",
                    hidden_size=HIDDEN_SIZE,
                    intermediate_size=SHARED_EXPERT_INTERMEDIATE_SIZE,
                ),
                fc1_latent_proj=SimpleNamespace(
                    in_features=HIDDEN_SIZE, out_features=MOE_LATENT_SIZE
                ),
                fc2_latent_proj=SimpleNamespace(
                    in_features=MOE_LATENT_SIZE, out_features=HIDDEN_SIZE
                ),
            )
        else:
            mixer = _named(
                "NemotronHMamba2Mixer"
                if layer_type == "mamba"
                else "NemotronHAttention"
            )
        layers.append(SimpleNamespace(block_type=layer_type, mixer=mixer))
    config = SimpleNamespace(
        model_type="nemotron_h", hidden_size=HIDDEN_SIZE, layers_block_type=LAYER_TYPES
    )
    return _named(
        "NemotronHForCausalLM", config=config, model=SimpleNamespace(layers=layers)
    )


def test_ultra_contract_is_an_upward_550b_active_parameter_point() -> None:
    contract = static_host_contract()
    assert contract["total_parameters"] == 550_000_000_000
    assert contract["active_parameters"] == 55_000_000_000
    assert contract["minimum_h100s"] == 8
    assert contract["model_layers"] == MODEL_LAYERS == 108
    assert contract["controlled_layer_indices"] == list(CONTROLLED_LAYER_INDICES)
    assert TRAINABLE_PARAMETERS_PER_ROLE == 4_718_592


def test_ultra_config_rejects_host_or_quantization_drift() -> None:
    payload = _config()
    validate_config_payload(payload)
    for path, changed in (
        (("hidden_size",), 4096),
        (("num_experts_per_tok",), 8),
        (("layers_block_type",), list(reversed(LAYER_TYPES))),
        (("quantization_config", "quant_algo"), "FP8"),
        (("quantization_config", "producer", "version"), "0.41.0"),
    ):
        drifted = deepcopy(payload)
        target = drifted
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = changed
        with pytest.raises(Q36UpwardMoEUltraHostError, match="config differs"):
            validate_config_payload(drifted)


def test_ultra_live_surface_binds_all_layers_and_final_sixteen_moe_mixers() -> None:
    receipt = validate_loaded_surface(_loaded_host())
    assert receipt["model_layers"] == 108
    assert receipt["moe_layers"] == 48
    assert receipt["controlled_layer_indices"] == list(CONTROLLED_LAYER_INDICES)
    assert len(receipt["native_topology_sha256"]) == 64


def test_ultra_live_surface_rejects_router_drift() -> None:
    host = _loaded_host()
    host.model.layers[CONTROLLED_LAYER_INDICES[0]].mixer.gate.top_k -= 1
    with pytest.raises(Q36UpwardMoEUltraHostError, match="layer differs"):
        validate_loaded_surface(host)
