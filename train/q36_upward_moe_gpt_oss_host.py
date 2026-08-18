"""Exact GPT-OSS-120B upward-MoE host and Shohin attachment contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

MODEL_ID = "openai/gpt-oss-120b"
MODEL_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
MODEL_CONFIG_SHA256 = "933aeb666a3fd851133ddd7686414f369bc564c4185fb5704416550879f10566"
MODEL_MANIFEST_SHA256 = (
    "a2a8026b3ab645285045dab439934416628e7bce392b351499b023298ff4a6d0"
)
MODEL_SOURCE_REVISION_SHA256 = (
    "1ce60ef7f313e9867434420ba27ffc75eeb92153be0be03cdef56ececa61c6f8"
)
MODEL_CLASS = "GptOssForCausalLM"
MODEL_TYPE = "gpt_oss"
TOTAL_PARAMETERS = 117_000_000_000
ACTIVE_PARAMETERS = 5_100_000_000
MODEL_REPOSITORY_BYTES = 65_276_859_410
HIDDEN_SIZE = 2880
MODEL_LAYERS = 36
NUM_EXPERTS = 128
ROUTER_TOP_K = 4
INTERMEDIATE_SIZE = 2880
VOCAB_SIZE = 201_088
MAX_POSITION_EMBEDDINGS = 131_072
SLIDING_WINDOW = 128
LAYER_TYPES = tuple(
    "sliding_attention" if index % 2 == 0 else "full_attention"
    for index in range(MODEL_LAYERS)
)
CONTROLLED_LAYERS = 16
CONTROLLED_LAYER_INDICES = tuple(range(MODEL_LAYERS - CONTROLLED_LAYERS, MODEL_LAYERS))
RANK = 18
ALPHA = 18.0
TRAINABLE_PARAMETERS_PER_ROLE = CONTROLLED_LAYERS * 2 * HIDDEN_SIZE * RANK
ATTACHMENT_SURFACE = "post-mlp-residual"
QUANTIZATION_METHOD = "mxfp4"
KERNEL_REPOSITORY = "kernels-community/gpt-oss-triton-kernels"
KERNEL_REVISION = "9655fcf7d0f638bec4a82f6f1a70014f0aa8cfb0"


class Q36UpwardMoEGptOssHostError(RuntimeError):
    """The pinned GPT-OSS host or its Shohin attachment surface differs."""


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
        "model_manifest_sha256": MODEL_MANIFEST_SHA256,
        "model_source_revision_sha256": MODEL_SOURCE_REVISION_SHA256,
        "model_class": MODEL_CLASS,
        "model_type": MODEL_TYPE,
        "total_parameters": TOTAL_PARAMETERS,
        "active_parameters": ACTIVE_PARAMETERS,
        "model_repository_bytes": MODEL_REPOSITORY_BYTES,
        "hidden_size": HIDDEN_SIZE,
        "model_layers": MODEL_LAYERS,
        "layer_types": list(LAYER_TYPES),
        "num_experts": NUM_EXPERTS,
        "router_top_k": ROUTER_TOP_K,
        "intermediate_size": INTERMEDIATE_SIZE,
        "vocab_size": VOCAB_SIZE,
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        "controlled_layer_indices": list(CONTROLLED_LAYER_INDICES),
        "rank": RANK,
        "alpha": ALPHA,
        "trainable_parameters_per_role": TRAINABLE_PARAMETERS_PER_ROLE,
        "attachment_surface": ATTACHMENT_SURFACE,
        "quantization_method": QUANTIZATION_METHOD,
        "kernel_repository": KERNEL_REPOSITORY,
        "kernel_revision": KERNEL_REVISION,
        "native_router_expert_trainables": 0,
    }


def validate_config_payload(payload: Mapping[str, Any]) -> None:
    quantization = payload.get("quantization_config")
    observed = {
        "architectures": payload.get("architectures"),
        "model_type": payload.get("model_type"),
        "hidden_size": payload.get("hidden_size"),
        "num_hidden_layers": payload.get("num_hidden_layers"),
        "num_local_experts": payload.get("num_local_experts"),
        "num_experts_per_tok": payload.get("num_experts_per_tok"),
        "experts_per_token": payload.get("experts_per_token"),
        "intermediate_size": payload.get("intermediate_size"),
        "vocab_size": payload.get("vocab_size"),
        "sliding_window": payload.get("sliding_window"),
        "max_position_embeddings": payload.get("max_position_embeddings"),
        "layer_types": payload.get("layer_types"),
        "quant_method": (
            quantization.get("quant_method")
            if isinstance(quantization, Mapping)
            else None
        ),
        "modules_to_not_convert": (
            quantization.get("modules_to_not_convert")
            if isinstance(quantization, Mapping)
            else None
        ),
    }
    expected = {
        "architectures": [MODEL_CLASS],
        "model_type": MODEL_TYPE,
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": MODEL_LAYERS,
        "num_local_experts": NUM_EXPERTS,
        "num_experts_per_tok": ROUTER_TOP_K,
        "experts_per_token": ROUTER_TOP_K,
        "intermediate_size": INTERMEDIATE_SIZE,
        "vocab_size": VOCAB_SIZE,
        "sliding_window": SLIDING_WINDOW,
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        "layer_types": list(LAYER_TYPES),
        "quant_method": QUANTIZATION_METHOD,
        "modules_to_not_convert": [
            "model.layers.*.self_attn",
            "model.layers.*.mlp.router",
            "model.embed_tokens",
            "lm_head",
        ],
    }
    if observed != expected:
        raise Q36UpwardMoEGptOssHostError(
            f"GPT-OSS config differs: expected={expected!r} observed={observed!r}"
        )


def load_pinned_config(path: Path) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != MODEL_CONFIG_SHA256
    ):
        raise Q36UpwardMoEGptOssHostError("GPT-OSS config hash differs")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Q36UpwardMoEGptOssHostError("GPT-OSS config is unreadable") from error
    if not isinstance(payload, dict):
        raise Q36UpwardMoEGptOssHostError("GPT-OSS config payload differs")
    validate_config_payload(payload)
    return payload


def _sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, (list, tuple)) or type(value).__name__ == "ModuleList":
        return value
    return None


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    try:
        return tuple(int(item) for item in shape) if shape is not None else None
    except (TypeError, ValueError):
        return None


def validate_loaded_surface(backbone: Any) -> dict[str, Any]:
    """Pin the live native MXFP4 topology before attaching Shohin residuals."""

    config = getattr(backbone, "config", None)
    model = getattr(backbone, "model", None)
    layers = _sequence(getattr(model, "layers", None))
    if (
        type(backbone).__name__ != MODEL_CLASS
        or getattr(config, "model_type", None) != MODEL_TYPE
        or getattr(config, "hidden_size", None) != HIDDEN_SIZE
        or getattr(config, "num_hidden_layers", None) != MODEL_LAYERS
        or getattr(config, "num_local_experts", None) != NUM_EXPERTS
        or getattr(config, "num_experts_per_tok", None) != ROUTER_TOP_K
        or getattr(config, "intermediate_size", None) != INTERMEDIATE_SIZE
        or tuple(getattr(config, "layer_types", ())) != LAYER_TYPES
        or layers is None
        or len(layers) != MODEL_LAYERS
    ):
        raise Q36UpwardMoEGptOssHostError("loaded GPT-OSS host geometry differs")

    rows: list[dict[str, Any]] = []
    for index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        router = getattr(mlp, "router", None)
        experts = getattr(mlp, "experts", None)
        row = {
            "layer": index,
            "layer_class": type(layer).__name__,
            "attention_class": type(getattr(layer, "self_attn", None)).__name__,
            "mlp_class": type(mlp).__name__,
            "router_class": type(router).__name__,
            "router_top_k": getattr(router, "top_k", None),
            "router_experts": getattr(router, "num_experts", None),
            "router_hidden": getattr(router, "hidden_dim", None),
            "experts_class": type(experts).__name__,
            "expert_count": getattr(experts, "num_experts", None),
            "expert_hidden": getattr(experts, "hidden_size", None),
            "expert_intermediate": getattr(experts, "intermediate_size", None),
            "gate_up_bias_shape": _shape(getattr(experts, "gate_up_proj_bias", None)),
            "down_bias_shape": _shape(getattr(experts, "down_proj_bias", None)),
            "gate_up_precision": getattr(experts, "gate_up_proj_precision_config", None)
            is not None,
            "down_precision": getattr(experts, "down_proj_precision_config", None)
            is not None,
        }
        expected = {
            "layer": index,
            "layer_class": "GptOssDecoderLayer",
            "attention_class": "GptOssAttention",
            "mlp_class": "GptOssMLP",
            "router_class": "GptOssTopKRouter",
            "router_top_k": ROUTER_TOP_K,
            "router_experts": NUM_EXPERTS,
            "router_hidden": HIDDEN_SIZE,
            "experts_class": "Mxfp4GptOssExperts",
            "expert_count": NUM_EXPERTS,
            "expert_hidden": HIDDEN_SIZE,
            "expert_intermediate": INTERMEDIATE_SIZE,
            "gate_up_bias_shape": (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE),
            "down_bias_shape": (NUM_EXPERTS, HIDDEN_SIZE),
            "gate_up_precision": True,
            "down_precision": True,
        }
        if row != expected:
            raise Q36UpwardMoEGptOssHostError(
                f"loaded GPT-OSS layer differs: expected={expected!r} observed={row!r}"
            )
        rows.append(row)
    encoded = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    return {
        "model_layers": MODEL_LAYERS,
        "moe_layers": MODEL_LAYERS,
        "controlled_layer_indices": list(CONTROLLED_LAYER_INDICES),
        "native_topology_sha256": hashlib.sha256(encoded).hexdigest(),
        "attachment_surface": ATTACHMENT_SURFACE,
        "quantization_method": QUANTIZATION_METHOD,
        "native_router_expert_trainables": 0,
    }
