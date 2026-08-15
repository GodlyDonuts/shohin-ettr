"""Exact Mixtral-8x22B upward-MoE host and Shohin attachment contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

MODEL_ID = "mistralai/Mixtral-8x22B-Instruct-v0.1"
MODEL_REVISION = "cc88a6cc19fbd17d9f1c0ee0b0d70a748dce698d"
MODEL_CONFIG_SHA256 = "9c4a6138d84029ab666943613e3d5844d2ea8fd6149f44f77188c62e2915e0f5"
MODEL_CLASS = "MixtralForCausalLM"
MODEL_TYPE = "mixtral"
TOTAL_PARAMETERS = 141_000_000_000
ACTIVE_PARAMETERS = 39_000_000_000
MODEL_REPOSITORY_BYTES = 281_260_955_124
HIDDEN_SIZE = 6144
MODEL_LAYERS = 56
NUM_EXPERTS = 8
ROUTER_TOP_K = 2
INTERMEDIATE_SIZE = 16384
VOCAB_SIZE = 32768
MAX_POSITION_EMBEDDINGS = 65536
CONTROLLED_LAYERS = 16
CONTROLLED_LAYER_INDICES = tuple(range(MODEL_LAYERS - CONTROLLED_LAYERS, MODEL_LAYERS))
RANK = 18
ALPHA = 18.0
TRAINABLE_PARAMETERS_PER_ROLE = CONTROLLED_LAYERS * 2 * HIDDEN_SIZE * RANK
ATTACHMENT_SURFACE = "post-mlp-residual"


class Q36UpwardMoEMixtralHostError(RuntimeError):
    """The pinned Mixtral host or its Shohin attachment surface differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
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
        "hidden_size": HIDDEN_SIZE,
        "model_layers": MODEL_LAYERS,
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
        "native_router_expert_trainables": 0,
    }


def validate_config_payload(payload: Mapping[str, Any]) -> None:
    observed = {
        "architectures": payload.get("architectures"),
        "model_type": payload.get("model_type"),
        "hidden_size": payload.get("hidden_size"),
        "num_hidden_layers": payload.get("num_hidden_layers"),
        "num_local_experts": payload.get("num_local_experts"),
        "num_experts_per_tok": payload.get("num_experts_per_tok"),
        "intermediate_size": payload.get("intermediate_size"),
        "vocab_size": payload.get("vocab_size"),
        "sliding_window": payload.get("sliding_window"),
        "max_position_embeddings": payload.get("max_position_embeddings"),
        "router_aux_loss_coef": payload.get("router_aux_loss_coef"),
    }
    expected = {
        "architectures": [MODEL_CLASS],
        "model_type": MODEL_TYPE,
        "hidden_size": HIDDEN_SIZE,
        "num_hidden_layers": MODEL_LAYERS,
        "num_local_experts": NUM_EXPERTS,
        "num_experts_per_tok": ROUTER_TOP_K,
        "intermediate_size": INTERMEDIATE_SIZE,
        "vocab_size": VOCAB_SIZE,
        "sliding_window": None,
        "max_position_embeddings": MAX_POSITION_EMBEDDINGS,
        "router_aux_loss_coef": 0.001,
    }
    if observed != expected:
        raise Q36UpwardMoEMixtralHostError(
            f"Mixtral config differs: expected={expected!r} observed={observed!r}"
        )


def load_pinned_config(path: Path) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != MODEL_CONFIG_SHA256
    ):
        raise Q36UpwardMoEMixtralHostError("Mixtral config hash differs")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Q36UpwardMoEMixtralHostError("Mixtral config is unreadable") from error
    if not isinstance(payload, dict):
        raise Q36UpwardMoEMixtralHostError("Mixtral config payload differs")
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
    config = getattr(backbone, "config", None)
    layers = _sequence(getattr(getattr(backbone, "model", None), "layers", None))
    if (
        type(backbone).__name__ != MODEL_CLASS
        or getattr(config, "model_type", None) != MODEL_TYPE
        or getattr(config, "hidden_size", None) != HIDDEN_SIZE
        or getattr(config, "num_hidden_layers", None) != MODEL_LAYERS
        or getattr(config, "num_local_experts", None) != NUM_EXPERTS
        or getattr(config, "num_experts_per_tok", None) != ROUTER_TOP_K
        or getattr(config, "intermediate_size", None) != INTERMEDIATE_SIZE
        or layers is None
        or len(layers) != MODEL_LAYERS
    ):
        raise Q36UpwardMoEMixtralHostError("loaded Mixtral host geometry differs")

    rows: list[dict[str, Any]] = []
    for index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        gate = getattr(mlp, "gate", None)
        experts = getattr(mlp, "experts", None)
        row = {
            "layer": index,
            "layer_class": type(layer).__name__,
            "attention_class": type(getattr(layer, "self_attn", None)).__name__,
            "mlp_class": type(mlp).__name__,
            "mlp_top_k": getattr(mlp, "top_k", None),
            "gate_class": type(gate).__name__,
            "gate_top_k": getattr(gate, "top_k", None),
            "gate_experts": getattr(gate, "num_experts", None),
            "gate_hidden": getattr(gate, "hidden_dim", None),
            "experts_class": type(experts).__name__,
            "expert_count": getattr(experts, "num_experts", None),
            "expert_hidden": getattr(experts, "hidden_dim", None),
            "expert_intermediate": getattr(experts, "intermediate_dim", None),
            "gate_up_shape": _shape(getattr(experts, "gate_up_proj", None)),
            "down_shape": _shape(getattr(experts, "down_proj", None)),
        }
        expected = {
            "layer": index,
            "layer_class": "MixtralDecoderLayer",
            "attention_class": "MixtralAttention",
            "mlp_class": "MixtralSparseMoeBlock",
            "mlp_top_k": ROUTER_TOP_K,
            "gate_class": "MixtralTopKRouter",
            "gate_top_k": ROUTER_TOP_K,
            "gate_experts": NUM_EXPERTS,
            "gate_hidden": HIDDEN_SIZE,
            "experts_class": "MixtralExperts",
            "expert_count": NUM_EXPERTS,
            "expert_hidden": HIDDEN_SIZE,
            "expert_intermediate": INTERMEDIATE_SIZE,
            "gate_up_shape": (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
            "down_shape": (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        }
        if row != expected:
            raise Q36UpwardMoEMixtralHostError(
                f"loaded Mixtral layer differs: expected={expected!r} observed={row!r}"
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
        "native_router_expert_trainables": 0,
    }
