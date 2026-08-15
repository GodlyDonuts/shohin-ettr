"""Exact 2.8T-A104B Kimi K3 host and distributed Shohin contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

MODEL_ID = "moonshotai/Kimi-K3"
MODEL_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
MODEL_CONFIG_SHA256 = "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213"
MODELING_KIMI_K3_SHA256 = (
    "b9171c96726eda55234c92ac8dfae7e24c512fda68968ae8f2c3782b42665ea2"
)
MODELING_KIMI_LINEAR_SHA256 = (
    "9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a"
)
MODEL_CLASS = "KimiK3ForConditionalGeneration"
MODEL_TYPE = "kimi_k3"
TEXT_MODEL_CLASS = "KimiLinearForCausalLM"
TEXT_MODEL_TYPE = "kimi_linear"
TOTAL_PARAMETERS = 2_800_000_000_000
ACTIVE_PARAMETERS = 104_000_000_000
MODEL_REPOSITORY_BYTES = 1_560_998_984_390
MODEL_WEIGHT_BYTES = 1_560_936_091_448
MINIMUM_H100S = 24
MINIMUM_NODES = 3
H100S_PER_NODE = 8
HIDDEN_SIZE = 7168
MODEL_LAYERS = 93
DENSE_PREFIX_LAYERS = 1
NUM_EXPERTS = 896
ROUTER_TOP_K = 16
NUM_SHARED_EXPERTS = 2
MOE_INTERMEDIATE_SIZE = 3072
ROUTED_EXPERT_HIDDEN_SIZE = 3584
ROUTED_SCALING_FACTOR = 1.0
MOE_RENORMALIZE = True
MOE_ROUTER_ACTIVATION = "sigmoid"
QUANTIZATION_METHOD = "compressed-tensors"
QUANTIZATION_FORMAT = "mxfp4-pack-quantized"
MOE_LAYER_INDICES = tuple(range(DENSE_PREFIX_LAYERS, MODEL_LAYERS))
CONTROLLED_LAYERS = 16
CONTROLLED_LAYER_INDICES = MOE_LAYER_INDICES[-CONTROLLED_LAYERS:]
RANK = 18
ALPHA = 18.0
TRAINABLE_PARAMETERS_PER_ROLE = CONTROLLED_LAYERS * 2 * HIDDEN_SIZE * RANK
TEMPORAL_GATE_PARAMETERS = CONTROLLED_LAYERS * (HIDDEN_SIZE + 1)
ATTACHMENT_SURFACE = "post-block-sparse-moe-residual"
NATIVE_EXECUTION_MODE = "eval"


class Q36UpwardMoEKimiK3HostError(RuntimeError):
    """The pinned Kimi K3 host or its frozen post-MoE surface differs."""


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
        "modeling_kimi_k3_sha256": MODELING_KIMI_K3_SHA256,
        "modeling_kimi_linear_sha256": MODELING_KIMI_LINEAR_SHA256,
        "model_class": MODEL_CLASS,
        "model_type": MODEL_TYPE,
        "text_model_class": TEXT_MODEL_CLASS,
        "text_model_type": TEXT_MODEL_TYPE,
        "total_parameters": TOTAL_PARAMETERS,
        "active_parameters": ACTIVE_PARAMETERS,
        "model_repository_bytes": MODEL_REPOSITORY_BYTES,
        "model_weight_bytes": MODEL_WEIGHT_BYTES,
        "minimum_h100s": MINIMUM_H100S,
        "minimum_nodes": MINIMUM_NODES,
        "h100s_per_node": H100S_PER_NODE,
        "hidden_size": HIDDEN_SIZE,
        "model_layers": MODEL_LAYERS,
        "dense_prefix_layers": DENSE_PREFIX_LAYERS,
        "moe_layer_indices": list(MOE_LAYER_INDICES),
        "controlled_layer_indices": list(CONTROLLED_LAYER_INDICES),
        "num_experts": NUM_EXPERTS,
        "router_top_k": ROUTER_TOP_K,
        "num_shared_experts": NUM_SHARED_EXPERTS,
        "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
        "routed_expert_hidden_size": ROUTED_EXPERT_HIDDEN_SIZE,
        "quantization_method": QUANTIZATION_METHOD,
        "quantization_format": QUANTIZATION_FORMAT,
        "attachment_surface": ATTACHMENT_SURFACE,
        "native_execution_mode": NATIVE_EXECUTION_MODE,
        "rank": RANK,
        "alpha": ALPHA,
        "trainable_parameters_per_role": TRAINABLE_PARAMETERS_PER_ROLE,
        "temporal_gate_parameters": TEMPORAL_GATE_PARAMETERS,
        "native_router_expert_trainables": 0,
    }


def validate_config_payload(payload: Mapping[str, Any]) -> None:
    text = payload.get("text_config")
    quantization = (
        text.get("quantization_config") if isinstance(text, Mapping) else None
    )
    observed = {
        "architectures": payload.get("architectures"),
        "model_type": payload.get("model_type"),
        "text_architectures": (
            text.get("architectures") if isinstance(text, Mapping) else None
        ),
        "text_model_type": (
            text.get("model_type") if isinstance(text, Mapping) else None
        ),
        "hidden_size": text.get("hidden_size") if isinstance(text, Mapping) else None,
        "num_hidden_layers": (
            text.get("num_hidden_layers") if isinstance(text, Mapping) else None
        ),
        "first_k_dense_replace": (
            text.get("first_k_dense_replace") if isinstance(text, Mapping) else None
        ),
        "moe_layer_freq": (
            text.get("moe_layer_freq") if isinstance(text, Mapping) else None
        ),
        "num_experts": text.get("num_experts") if isinstance(text, Mapping) else None,
        "num_experts_per_token": (
            text.get("num_experts_per_token") if isinstance(text, Mapping) else None
        ),
        "num_shared_experts": (
            text.get("num_shared_experts") if isinstance(text, Mapping) else None
        ),
        "moe_intermediate_size": (
            text.get("moe_intermediate_size") if isinstance(text, Mapping) else None
        ),
        "routed_expert_hidden_size": (
            text.get("routed_expert_hidden_size") if isinstance(text, Mapping) else None
        ),
        "moe_renormalize": (
            text.get("moe_renormalize") if isinstance(text, Mapping) else None
        ),
        "moe_router_activation_func": (
            text.get("moe_router_activation_func")
            if isinstance(text, Mapping)
            else None
        ),
        "routed_scaling_factor": (
            text.get("routed_scaling_factor") if isinstance(text, Mapping) else None
        ),
        "quant_method": (
            quantization.get("quant_method")
            if isinstance(quantization, Mapping)
            else None
        ),
        "quant_format": (
            quantization.get("format") if isinstance(quantization, Mapping) else None
        ),
    }
    expected = {
        "architectures": [MODEL_CLASS],
        "model_type": MODEL_TYPE,
        "text_architectures": [TEXT_MODEL_CLASS],
        "text_model_type": TEXT_MODEL_TYPE,
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": MODEL_LAYERS,
        "first_k_dense_replace": DENSE_PREFIX_LAYERS,
        "moe_layer_freq": 1,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_token": ROUTER_TOP_K,
        "num_shared_experts": NUM_SHARED_EXPERTS,
        "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
        "routed_expert_hidden_size": ROUTED_EXPERT_HIDDEN_SIZE,
        "moe_renormalize": MOE_RENORMALIZE,
        "moe_router_activation_func": MOE_ROUTER_ACTIVATION,
        "routed_scaling_factor": ROUTED_SCALING_FACTOR,
        "quant_method": QUANTIZATION_METHOD,
        "quant_format": QUANTIZATION_FORMAT,
    }
    if observed != expected:
        raise Q36UpwardMoEKimiK3HostError(
            f"Kimi K3 config differs: expected={expected!r} observed={observed!r}"
        )


def load_pinned_config(path: Path) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != MODEL_CONFIG_SHA256
    ):
        raise Q36UpwardMoEKimiK3HostError("Kimi K3 config hash differs")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Q36UpwardMoEKimiK3HostError("Kimi K3 config is unreadable") from error
    if not isinstance(payload, dict):
        raise Q36UpwardMoEKimiK3HostError("Kimi K3 config payload differs")
    validate_config_payload(payload)
    return payload


def _sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, (list, tuple)) or type(value).__name__ == "ModuleList":
        return value
    return None


def validate_loaded_surface(backbone: Any) -> dict[str, Any]:
    config = getattr(backbone, "config", None)
    text_config = getattr(config, "text_config", None)
    language_model = getattr(backbone, "language_model", None)
    text_model = getattr(language_model, "model", None)
    layers = _sequence(getattr(text_model, "layers", None))
    if (
        type(backbone).__name__ != MODEL_CLASS
        or getattr(config, "model_type", None) != MODEL_TYPE
        or type(language_model).__name__ != TEXT_MODEL_CLASS
        or getattr(text_config, "model_type", None) != TEXT_MODEL_TYPE
        or getattr(text_config, "hidden_size", None) != HIDDEN_SIZE
        or layers is None
        or len(layers) != MODEL_LAYERS
    ):
        raise Q36UpwardMoEKimiK3HostError("loaded Kimi K3 host geometry differs")

    rows: list[dict[str, Any]] = []
    for index, layer in enumerate(layers):
        sparse = getattr(layer, "block_sparse_moe", None)
        dense = getattr(layer, "mlp", None)
        row = {
            "layer": index,
            "layer_class": type(layer).__name__,
            "sparse_class": type(sparse).__name__ if sparse is not None else None,
            "dense_class": type(dense).__name__ if dense is not None else None,
        }
        if index == 0:
            expected = {
                "layer": 0,
                "layer_class": "KimiDecoderLayer",
                "sparse_class": None,
                "dense_class": "KimiMLP",
            }
        else:
            gate = getattr(sparse, "gate", None)
            experts = _sequence(getattr(sparse, "experts", None))
            row.update(
                {
                    "hidden_dim": getattr(sparse, "hidden_dim", None),
                    "num_experts": getattr(sparse, "num_experts", None),
                    "top_k": getattr(sparse, "top_k", None),
                    "expert_count": len(experts) if experts is not None else None,
                    "gate_class": type(gate).__name__,
                    "shared_class": type(
                        getattr(sparse, "shared_experts", None)
                    ).__name__,
                    "down_geometry": (
                        getattr(
                            getattr(sparse, "routed_expert_down_proj", None),
                            "in_features",
                            None,
                        ),
                        getattr(
                            getattr(sparse, "routed_expert_down_proj", None),
                            "out_features",
                            None,
                        ),
                    ),
                    "up_geometry": (
                        getattr(
                            getattr(sparse, "routed_expert_up_proj", None),
                            "in_features",
                            None,
                        ),
                        getattr(
                            getattr(sparse, "routed_expert_up_proj", None),
                            "out_features",
                            None,
                        ),
                    ),
                }
            )
            expected = {
                "layer": index,
                "layer_class": "KimiDecoderLayer",
                "sparse_class": "KimiSparseMoeBlock",
                "dense_class": None,
                "hidden_dim": HIDDEN_SIZE,
                "num_experts": NUM_EXPERTS,
                "top_k": ROUTER_TOP_K,
                "expert_count": NUM_EXPERTS,
                "gate_class": "KimiMoEGate",
                "shared_class": "KimiMLP",
                "down_geometry": (HIDDEN_SIZE, ROUTED_EXPERT_HIDDEN_SIZE),
                "up_geometry": (ROUTED_EXPERT_HIDDEN_SIZE, HIDDEN_SIZE),
            }
        if row != expected:
            raise Q36UpwardMoEKimiK3HostError(
                f"loaded Kimi K3 layer differs: expected={expected!r} observed={row!r}"
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
        "native_execution_mode": NATIVE_EXECUTION_MODE,
        "native_router_expert_trainables": 0,
    }


if len(MOE_LAYER_INDICES) != 92 or CONTROLLED_LAYER_INDICES != tuple(range(77, 93)):
    raise RuntimeError("pinned Kimi K3 sparse layer geometry differs")
