from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from q36_upward_moe_host import (
    CONTROLLED_LAYER_INDICES,
    HIDDEN_SIZE,
    HYBRID_PATTERN,
    LAYER_TYPES,
    MODEL_CONFIG_SHA256,
    MODEL_LAYERS,
    MOE_INTERMEDIATE_SIZE,
    MOE_LATENT_SIZE,
    NUM_EXPERTS,
    Q36UpwardMoEHostError,
    ROUTER_TOP_K,
    SHARED_EXPERT_INTERMEDIATE_SIZE,
    TRAINABLE_PARAMETERS_PER_ROLE,
    load_pinned_config,
    static_host_contract,
    validate_config_payload,
    validate_loaded_surface,
)


def _config_payload() -> dict:
    return {
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": MODEL_LAYERS,
        "hybrid_override_pattern": HYBRID_PATTERN,
        "n_routed_experts": NUM_EXPERTS,
        "num_experts_per_tok": ROUTER_TOP_K,
        "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
        "moe_shared_expert_intermediate_size": SHARED_EXPERT_INTERMEDIATE_SIZE,
        "moe_latent_size": MOE_LATENT_SIZE,
        "routed_scaling_factor": 5.0,
        "n_group": 1,
        "topk_group": 1,
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "FP8",
            "producer": {"name": "modelopt", "version": "0.41.0"},
        },
    }


def _named(name: str, **values):
    value = type(name, (), {})()
    for key, item in values.items():
        setattr(value, key, item)
    return value


def _loaded_host():
    layers = []
    for layer_type in LAYER_TYPES:
        if layer_type == "moe":
            expert = lambda: _named(  # noqa: E731
                "NemotronHMLP",
                hidden_size=HIDDEN_SIZE,
                intermediate_size=MOE_INTERMEDIATE_SIZE,
            )
            mixer = _named(
                "NemotronHMoE",
                gate=_named(
                    "NemotronHTopkRouter",
                    top_k=ROUTER_TOP_K,
                    n_routed_experts=NUM_EXPERTS,
                    n_group=1,
                    topk_group=1,
                ),
                experts=[expert() for _ in range(NUM_EXPERTS)],
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
        elif layer_type == "mamba":
            mixer = _named("NemotronHMamba2Mixer")
        else:
            mixer = _named("NemotronHAttention")
        layers.append(SimpleNamespace(block_type=layer_type, mixer=mixer))
    config = SimpleNamespace(
        model_type="nemotron_h",
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=MODEL_LAYERS,
        layers_block_type=list(LAYER_TYPES),
    )
    return _named(
        "NemotronHForCausalLM", config=config, model=SimpleNamespace(layers=layers)
    )


def test_static_contract_scales_q36_to_final_sixteen_moe_layers() -> None:
    contract = static_host_contract()
    assert contract["total_parameters"] == 120_000_000_000
    assert contract["active_parameters"] == 12_000_000_000
    assert contract["controlled_layer_indices"] == [
        54,
        56,
        59,
        61,
        63,
        65,
        67,
        70,
        72,
        74,
        76,
        79,
        81,
        83,
        85,
        87,
    ]
    assert tuple(contract["controlled_layer_indices"]) == CONTROLLED_LAYER_INDICES
    assert TRAINABLE_PARAMETERS_PER_ROLE == 2_359_296
    assert contract["native_router_expert_trainables"] == 0


def test_config_validator_accepts_only_pinned_host() -> None:
    payload = _config_payload()
    validate_config_payload(payload)
    for path, changed in (
        (("hidden_size",), 8192),
        (("hybrid_override_pattern",), HYBRID_PATTERN[::-1]),
        (("num_experts_per_tok",), 8),
        (("quantization_config", "quant_algo"), "NVFP4"),
        (("quantization_config", "producer", "version"), "0.42.0"),
    ):
        drifted = deepcopy(payload)
        target = drifted
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = changed
        with pytest.raises(Q36UpwardMoEHostError, match="config differs"):
            validate_config_payload(drifted)


def test_config_loader_rejects_unpinned_bytes(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{}")
    assert MODEL_CONFIG_SHA256 != ""
    with pytest.raises(Q36UpwardMoEHostError, match="hash differs"):
        load_pinned_config(path)


def test_live_surface_binds_every_native_layer_and_final_moe_attachment() -> None:
    host = _loaded_host()
    receipt = validate_loaded_surface(host)
    assert receipt["model_layers"] == 88
    assert receipt["moe_layers"] == 40
    assert receipt["controlled_layer_indices"] == list(CONTROLLED_LAYER_INDICES)
    assert len(receipt["native_topology_sha256"]) == 64
    assert receipt["attachment_surface"] == "post-mixer-residual"
    assert receipt["native_router_expert_trainables"] == 0


def test_live_surface_accepts_only_exact_modelopt_moe_wrapper_when_declared() -> None:
    host = _loaded_host()
    for layer in host.model.layers:
        if layer.block_type == "moe":
            layer.mixer.__class__ = type("QuantNemotronHMoE", (), {})
        elif layer.block_type == "attention":
            layer.mixer.__class__ = type("QuantNemotronHAttention", (), {})
    receipt = validate_loaded_surface(host, modelopt_quantized=True)
    assert len(receipt["native_topology_sha256"]) == 64

    with pytest.raises(Q36UpwardMoEHostError, match="layer differs"):
        validate_loaded_surface(host)

    host.model.layers[CONTROLLED_LAYER_INDICES[0]].mixer.__class__ = type(
        "UnexpectedQuantNemotronHMoE", (), {}
    )
    with pytest.raises(Q36UpwardMoEHostError, match="layer differs"):
        validate_loaded_surface(host, modelopt_quantized=True)

    host = _loaded_host()
    for layer in host.model.layers:
        if layer.block_type == "moe":
            layer.mixer.__class__ = type("QuantNemotronHMoE", (), {})
        elif layer.block_type == "attention":
            layer.mixer.__class__ = type("UnexpectedQuantNemotronHAttention", (), {})
    with pytest.raises(Q36UpwardMoEHostError, match="layer differs"):
        validate_loaded_surface(host, modelopt_quantized=True)


def test_live_surface_rejects_router_or_attachment_drift() -> None:
    host = _loaded_host()
    controlled = host.model.layers[CONTROLLED_LAYER_INDICES[0]]
    controlled.mixer.gate.top_k = ROUTER_TOP_K - 1
    with pytest.raises(Q36UpwardMoEHostError, match="layer differs"):
        validate_loaded_surface(host)

    host = _loaded_host()
    host.model.layers[CONTROLLED_LAYER_INDICES[-1]].mixer = _named("NemotronHAttention")
    with pytest.raises(Q36UpwardMoEHostError, match="layer differs"):
        validate_loaded_surface(host)
