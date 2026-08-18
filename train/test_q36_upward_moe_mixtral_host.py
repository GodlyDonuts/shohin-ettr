from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from q36_upward_moe_mixtral_host import (
    CONTROLLED_LAYER_INDICES,
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    MODEL_CONFIG_SHA256,
    MODEL_LAYERS,
    NUM_EXPERTS,
    Q36UpwardMoEMixtralHostError,
    ROUTER_TOP_K,
    TWO_H100_LAYER_BOUNDARY,
    TRAINABLE_PARAMETERS_PER_ROLE,
    load_pinned_config,
    static_host_contract,
    two_h100_device_map,
    validate_config_payload,
    validate_loaded_surface,
)


def test_two_h100_device_map_is_exact_balanced_and_has_no_offload() -> None:
    mapping = two_h100_device_map()
    assert mapping["model.embed_tokens"] == 0
    assert mapping["model.norm"] == 1
    assert mapping["lm_head"] == 1
    assert mapping[f"model.layers.{TWO_H100_LAYER_BOUNDARY - 1}"] == 0
    assert mapping[f"model.layers.{TWO_H100_LAYER_BOUNDARY}"] == 1
    assert (
        len([key for key in mapping if key.startswith("model.layers.")]) == MODEL_LAYERS
    )
    assert set(mapping.values()) == {0, 1}


def _config_payload() -> dict:
    return {
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": MODEL_LAYERS,
        "num_local_experts": NUM_EXPERTS,
        "num_experts_per_tok": ROUTER_TOP_K,
        "intermediate_size": INTERMEDIATE_SIZE,
        "vocab_size": 32768,
        "sliding_window": None,
        "max_position_embeddings": 65536,
        "router_aux_loss_coef": 0.001,
    }


def _named(name: str, **members: object) -> object:
    value = type(name, (), {})()
    for key, member in members.items():
        setattr(value, key, member)
    return value


def _loaded_host() -> object:
    layers = []
    for _ in range(MODEL_LAYERS):
        gate = _named(
            "MixtralTopKRouter",
            top_k=ROUTER_TOP_K,
            num_experts=NUM_EXPERTS,
            hidden_dim=HIDDEN_SIZE,
        )
        experts = _named(
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
        mlp = _named(
            "MixtralSparseMoeBlock",
            top_k=ROUTER_TOP_K,
            gate=gate,
            experts=experts,
        )
        layers.append(
            _named(
                "MixtralDecoderLayer",
                self_attn=_named("MixtralAttention"),
                mlp=mlp,
            )
        )
    config = SimpleNamespace(
        model_type="mixtral",
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=MODEL_LAYERS,
        num_local_experts=NUM_EXPERTS,
        num_experts_per_tok=ROUTER_TOP_K,
        intermediate_size=INTERMEDIATE_SIZE,
    )
    return _named(
        "MixtralForCausalLM", config=config, model=SimpleNamespace(layers=layers)
    )


def test_static_contract_is_upward_active_parameter_cross_family_point() -> None:
    contract = static_host_contract()
    assert contract["total_parameters"] == 141_000_000_000
    assert contract["active_parameters"] == 39_000_000_000
    assert contract["controlled_layer_indices"] == list(range(40, 56))
    assert tuple(contract["controlled_layer_indices"]) == CONTROLLED_LAYER_INDICES
    assert TRAINABLE_PARAMETERS_PER_ROLE == 3_538_944
    assert contract["native_router_expert_trainables"] == 0


def test_config_validator_rejects_family_or_router_drift() -> None:
    payload = _config_payload()
    validate_config_payload(payload)
    for key, changed in (
        ("hidden_size", 4096),
        ("num_hidden_layers", 55),
        ("num_local_experts", 16),
        ("num_experts_per_tok", 4),
        ("sliding_window", 4096),
    ):
        drifted = deepcopy(payload)
        drifted[key] = changed
        with pytest.raises(Q36UpwardMoEMixtralHostError, match="config differs"):
            validate_config_payload(drifted)


def test_config_loader_rejects_unpinned_bytes(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{}")
    assert MODEL_CONFIG_SHA256
    with pytest.raises(Q36UpwardMoEMixtralHostError, match="hash differs"):
        load_pinned_config(path)


def test_loaded_surface_binds_all_native_experts_and_final_layers() -> None:
    receipt = validate_loaded_surface(_loaded_host())
    assert receipt["model_layers"] == MODEL_LAYERS
    assert receipt["moe_layers"] == MODEL_LAYERS
    assert receipt["controlled_layer_indices"] == list(CONTROLLED_LAYER_INDICES)
    assert len(receipt["native_topology_sha256"]) == 64
    assert receipt["attachment_surface"] == "post-mlp-residual"


def test_loaded_surface_rejects_expert_or_attachment_drift() -> None:
    host = _loaded_host()
    host.model.layers[0].mlp.experts.intermediate_dim -= 1
    with pytest.raises(Q36UpwardMoEMixtralHostError, match="layer differs"):
        validate_loaded_surface(host)

    host = _loaded_host()
    host.model.layers[CONTROLLED_LAYER_INDICES[-1]].mlp = _named("DenseMLP")
    with pytest.raises(Q36UpwardMoEMixtralHostError, match="layer differs"):
        validate_loaded_surface(host)
