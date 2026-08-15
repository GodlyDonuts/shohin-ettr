"""Exact 550B-A55B Nemotron Ultra host and Shohin attachment contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4"
MODEL_REVISION = "183968f87ae4cedce3039313cac1fd43d112c578"
MODEL_CONFIG_SHA256 = "0c939f324c8910f5ebdafbe2a56d7e4e074c50042a3b4f26326bf71a3fe33929"
MODEL_CLASS = "NemotronHForCausalLM"
MODEL_TYPE = "nemotron_h"
TOTAL_PARAMETERS = 550_000_000_000
ACTIVE_PARAMETERS = 55_000_000_000
MODEL_REPOSITORY_BYTES = 352_381_245_521
MODEL_WEIGHT_BYTES = 352_284_061_280
MINIMUM_H100S = 8
HIDDEN_SIZE = 8192
MODEL_LAYERS = 108
NUM_EXPERTS = 512
ROUTER_TOP_K = 22
MOE_INTERMEDIATE_SIZE = 5120
SHARED_EXPERT_INTERMEDIATE_SIZE = 10240
MOE_LATENT_SIZE = 2048
ROUTED_SCALING_FACTOR = 5.0
ROUTER_GROUPS = 1
ROUTER_TOP_K_GROUPS = 1
QUANTIZATION_METHOD = "modelopt"
QUANTIZATION_ALGORITHM = "MIXED_PRECISION"
MODELOPT_VERSION = "1.0.0"
LAYER_TYPES = (
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "attention",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "attention",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "attention",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "attention",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "attention",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "attention",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "attention",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "attention",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "attention",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "attention",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "attention",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "attention",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
    "mamba",
    "moe",
)
MOE_LAYER_INDICES = tuple(
    index for index, layer_type in enumerate(LAYER_TYPES) if layer_type == "moe"
)
CONTROLLED_LAYERS = 16
CONTROLLED_LAYER_INDICES = MOE_LAYER_INDICES[-CONTROLLED_LAYERS:]
RANK = 18
ALPHA = 18.0
TRAINABLE_PARAMETERS_PER_ROLE = CONTROLLED_LAYERS * 2 * HIDDEN_SIZE * RANK
ATTACHMENT_SURFACE = "post-mixer-residual"


class Q36UpwardMoEUltraHostError(RuntimeError):
    """The pinned Ultra host or its Shohin attachment surface differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def static_host_contract() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "model_class": MODEL_CLASS,
        "model_type": MODEL_TYPE,
        "total_parameters": TOTAL_PARAMETERS,
        "active_parameters": ACTIVE_PARAMETERS,
        "model_repository_bytes": MODEL_REPOSITORY_BYTES,
        "model_weight_bytes": MODEL_WEIGHT_BYTES,
        "minimum_h100s": MINIMUM_H100S,
        "hidden_size": HIDDEN_SIZE,
        "model_layers": MODEL_LAYERS,
        "layer_types": list(LAYER_TYPES),
        "moe_layer_indices": list(MOE_LAYER_INDICES),
        "controlled_layer_indices": list(CONTROLLED_LAYER_INDICES),
        "num_experts": NUM_EXPERTS,
        "router_top_k": ROUTER_TOP_K,
        "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
        "shared_expert_intermediate_size": SHARED_EXPERT_INTERMEDIATE_SIZE,
        "moe_latent_size": MOE_LATENT_SIZE,
        "quantization_method": QUANTIZATION_METHOD,
        "quantization_algorithm": QUANTIZATION_ALGORITHM,
        "modelopt_version": MODELOPT_VERSION,
        "attachment_surface": ATTACHMENT_SURFACE,
        "rank": RANK,
        "alpha": ALPHA,
        "trainable_parameters_per_role": TRAINABLE_PARAMETERS_PER_ROLE,
        "native_router_expert_trainables": 0,
    }


def validate_config_payload(payload: Mapping[str, Any]) -> None:
    quantization = payload.get("quantization_config")
    producer = (
        quantization.get("producer") if isinstance(quantization, Mapping) else None
    )
    observed = {
        "architectures": payload.get("architectures"),
        "model_type": payload.get("model_type"),
        "hidden_size": payload.get("hidden_size"),
        "layers_block_type": tuple(payload.get("layers_block_type", ())),
        "n_routed_experts": payload.get("n_routed_experts"),
        "num_experts_per_tok": payload.get("num_experts_per_tok"),
        "moe_intermediate_size": payload.get("moe_intermediate_size"),
        "moe_shared_expert_intermediate_size": payload.get(
            "moe_shared_expert_intermediate_size"
        ),
        "moe_latent_size": payload.get("moe_latent_size"),
        "routed_scaling_factor": payload.get("routed_scaling_factor"),
        "n_group": payload.get("n_group"),
        "topk_group": payload.get("topk_group"),
        "mtp_layers_block_type": payload.get("mtp_layers_block_type"),
        "quant_method": (
            quantization.get("quant_method")
            if isinstance(quantization, Mapping)
            else None
        ),
        "quant_algo": (
            quantization.get("quant_algo")
            if isinstance(quantization, Mapping)
            else None
        ),
        "producer_name": (
            producer.get("name") if isinstance(producer, Mapping) else None
        ),
        "producer_version": (
            producer.get("version") if isinstance(producer, Mapping) else None
        ),
    }
    expected = {
        "architectures": [MODEL_CLASS],
        "model_type": MODEL_TYPE,
        "hidden_size": HIDDEN_SIZE,
        "layers_block_type": LAYER_TYPES,
        "n_routed_experts": NUM_EXPERTS,
        "num_experts_per_tok": ROUTER_TOP_K,
        "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
        "moe_shared_expert_intermediate_size": SHARED_EXPERT_INTERMEDIATE_SIZE,
        "moe_latent_size": MOE_LATENT_SIZE,
        "routed_scaling_factor": ROUTED_SCALING_FACTOR,
        "n_group": ROUTER_GROUPS,
        "topk_group": ROUTER_TOP_K_GROUPS,
        "mtp_layers_block_type": ["attention", "moe"],
        "quant_method": QUANTIZATION_METHOD,
        "quant_algo": QUANTIZATION_ALGORITHM,
        "producer_name": QUANTIZATION_METHOD,
        "producer_version": MODELOPT_VERSION,
    }
    if observed != expected:
        raise Q36UpwardMoEUltraHostError(
            f"Nemotron Ultra config differs: expected={expected!r} observed={observed!r}"
        )


def load_pinned_config(path: Path) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != MODEL_CONFIG_SHA256
    ):
        raise Q36UpwardMoEUltraHostError("Nemotron Ultra config hash differs")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Q36UpwardMoEUltraHostError(
            "Nemotron Ultra config is unreadable"
        ) from error
    if not isinstance(payload, dict):
        raise Q36UpwardMoEUltraHostError("Nemotron Ultra config payload differs")
    validate_config_payload(payload)
    return payload


def _sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, (list, tuple)) or type(value).__name__ == "ModuleList":
        return value
    return None


def _linear_geometry(module: Any) -> tuple[Any, Any]:
    return getattr(module, "in_features", None), getattr(module, "out_features", None)


def validate_loaded_surface(backbone: Any) -> dict[str, Any]:
    config = getattr(backbone, "config", None)
    layers = _sequence(getattr(getattr(backbone, "model", None), "layers", None))
    if (
        type(backbone).__name__ != MODEL_CLASS
        or getattr(config, "model_type", None) != MODEL_TYPE
        or getattr(config, "hidden_size", None) != HIDDEN_SIZE
        or tuple(getattr(config, "layers_block_type", ())) != LAYER_TYPES
        or layers is None
        or len(layers) != MODEL_LAYERS
    ):
        raise Q36UpwardMoEUltraHostError("loaded Nemotron Ultra host geometry differs")

    rows: list[dict[str, Any]] = []
    for index, (layer, expected_type) in enumerate(
        zip(layers, LAYER_TYPES, strict=True)
    ):
        mixer = getattr(layer, "mixer", None)
        row: dict[str, Any] = {
            "layer": index,
            "block_type": getattr(layer, "block_type", None),
            "mixer_class": type(mixer).__name__,
        }
        if expected_type == "moe":
            gate = getattr(mixer, "gate", None)
            experts = _sequence(getattr(mixer, "experts", None))
            shared = getattr(mixer, "shared_experts", None)
            first_expert = experts[0] if experts else None
            last_expert = experts[-1] if experts else None
            row.update(
                {
                    "gate_class": type(gate).__name__,
                    "router_top_k": getattr(gate, "top_k", None),
                    "router_experts": getattr(gate, "n_routed_experts", None),
                    "router_groups": getattr(gate, "n_group", None),
                    "router_top_k_groups": getattr(gate, "topk_group", None),
                    "expert_count": len(experts) if experts is not None else None,
                    "first_expert_class": type(first_expert).__name__,
                    "first_expert_hidden": getattr(first_expert, "hidden_size", None),
                    "first_expert_intermediate": getattr(
                        first_expert, "intermediate_size", None
                    ),
                    "last_expert_class": type(last_expert).__name__,
                    "shared_expert_class": type(shared).__name__,
                    "shared_expert_hidden": getattr(shared, "hidden_size", None),
                    "shared_expert_intermediate": getattr(
                        shared, "intermediate_size", None
                    ),
                    "fc1_latent_geometry": _linear_geometry(
                        getattr(mixer, "fc1_latent_proj", None)
                    ),
                    "fc2_latent_geometry": _linear_geometry(
                        getattr(mixer, "fc2_latent_proj", None)
                    ),
                }
            )
            expected_row = {
                "layer": index,
                "block_type": "moe",
                "mixer_class": "NemotronHMoE",
                "gate_class": "NemotronHTopkRouter",
                "router_top_k": ROUTER_TOP_K,
                "router_experts": NUM_EXPERTS,
                "router_groups": ROUTER_GROUPS,
                "router_top_k_groups": ROUTER_TOP_K_GROUPS,
                "expert_count": NUM_EXPERTS,
                "first_expert_class": "NemotronHMLP",
                "first_expert_hidden": HIDDEN_SIZE,
                "first_expert_intermediate": MOE_INTERMEDIATE_SIZE,
                "last_expert_class": "NemotronHMLP",
                "shared_expert_class": "NemotronHMLP",
                "shared_expert_hidden": HIDDEN_SIZE,
                "shared_expert_intermediate": SHARED_EXPERT_INTERMEDIATE_SIZE,
                "fc1_latent_geometry": (HIDDEN_SIZE, MOE_LATENT_SIZE),
                "fc2_latent_geometry": (MOE_LATENT_SIZE, HIDDEN_SIZE),
            }
        else:
            expected_row = {
                "layer": index,
                "block_type": expected_type,
                "mixer_class": (
                    "NemotronHMamba2Mixer"
                    if expected_type == "mamba"
                    else "NemotronHAttention"
                ),
            }
        if row != expected_row:
            raise Q36UpwardMoEUltraHostError(
                f"loaded Nemotron Ultra layer differs: expected={expected_row!r} observed={row!r}"
            )
        rows.append(row)
    canonical = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    return {
        "model_layers": MODEL_LAYERS,
        "moe_layers": len(MOE_LAYER_INDICES),
        "controlled_layer_indices": list(CONTROLLED_LAYER_INDICES),
        "native_topology_sha256": hashlib.sha256(canonical).hexdigest(),
        "attachment_surface": ATTACHMENT_SURFACE,
        "native_router_expert_trainables": 0,
    }


if len(LAYER_TYPES) != MODEL_LAYERS or len(MOE_LAYER_INDICES) != 48:
    raise RuntimeError("pinned Nemotron Ultra layer pattern differs")
