"""Frozen role and causal-visibility contract for Q36-MTR.

This module contains no scheduler or model-acquisition capability.  It is the
small, importable boundary shared by the prospective Q36 trainer, generator,
mechanics gate, and dry-run graph compiler.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
MODEL_CONFIG_SHA256 = "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
MODEL_MANIFEST_SHA256 = (
    "06c9d8d8419244f2d001cb351e164f356718d9d77138e898b13afee35856f56e"
)
ARCHITECTURE = "shohin-q36-mtr-shared-post-mlp-v1"
ROLE_CHECKPOINT_SCHEMA = "shohin-hf-product-reasoning-checkpoint-v1"
HIDDEN_SIZE = 2048
MODEL_LAYERS = 40
NUM_EXPERTS = 256
ROUTER_TOP_K = 8
MOE_INTERMEDIATE_SIZE = 512
SHARED_EXPERT_INTERMEDIATE_SIZE = 512
VOCAB_SIZE = 248_320
MODEL_TYPE = "qwen3_5_moe"
TEXT_MODEL_TYPE = "qwen3_5_moe_text"
CAUSAL_MODEL_CLASS = "Qwen3_5MoeForCausalLM"
LAYER_TYPES = tuple(
    "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
    for index in range(MODEL_LAYERS)
)
CONTROLLED_LAYERS = 16
CONTROLLED_LAYER_INDICES = tuple(range(MODEL_LAYERS - CONTROLLED_LAYERS, MODEL_LAYERS))
RANK = 18
ALPHA = 18.0
TRAINABLE_PARAMETERS = 1_179_648
QUANTIZATION = "nf4"
COMPUTE_DTYPE = "bfloat16"
TRAINABLE_MASTER_DTYPE = "float32"
ADAPTER_UPDATE_SCHEMA = "shohin-q36-mtr-adapter-update-v1"
OWNER_UPDATES = 256
REVISION_UPDATES = 256
OWNER_MAX_ROWS = 100_000
REVISION_PRESENTATIONS = 9_655
OWNER_MAX_SEQUENCE_LENGTH = 1_024
REVISION_MAX_SEQUENCE_LENGTH = 4_096
OWNER_LEARNING_RATE = 2e-4
REVISION_LEARNING_RATE = 2e-5
OWNER_GRADIENT_ACCUMULATION = 16
REVISION_GRADIENT_ACCUMULATION = 8
OWNER_SEED = 2026080711
OWNER_DATA_SEED = 20260802
REVISION_SEED = 2026080815
REVISION_DATA_SEED = 2026080814
DRAFT_SEED = 2026080818
DRAFT_SHARDS = 16
DRAFT_IDENTITIES = 7_113
DRAFT_MAX_NEW_TOKENS = 768
COMMIT_UPDATES = 128


class Q36MTRRoleError(RuntimeError):
    """The prospective Q36-MTR role contract differs."""


def load_role_checkpoint_payload(path: Path) -> dict[str, Any]:
    """Load only the exact Q36 residual payload without executable pickle data."""

    import torch

    from shared_post_mlp_revision import trainable_state_sha256

    if path.is_symlink() or not path.is_file():
        raise Q36MTRRoleError("Q36-MTR role checkpoint is absent or symbolic")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    saved = payload.get("trainable_state") if isinstance(payload, dict) else None
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "update", "trainable_state", "metadata"}
        or payload.get("schema") != ROLE_CHECKPOINT_SCHEMA
        or payload.get("update") != OWNER_UPDATES
        or not isinstance(saved, dict)
        or len(saved) != CONTROLLED_LAYERS * 2
        or sum(int(tensor.numel()) for tensor in saved.values()) != TRAINABLE_PARAMETERS
        or any(
            not isinstance(name, str)
            or not (
                name.endswith("adapter_a.weight") or name.endswith("adapter_b.weight")
            )
            or not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            for name, tensor in saved.items()
        )
        or not isinstance(metadata, dict)
        or metadata.get("optimizer_state_serialized") is not False
        or metadata.get("checkpoint_trainable_only") is not True
        or metadata.get("router_expert_checkpoint_tensors") != 0
        or metadata.get("serialization_restore_exact") is not True
        or metadata.get("final_trainable_state_sha256") != trainable_state_sha256(saved)
    ):
        raise Q36MTRRoleError("Q36-MTR role checkpoint payload differs")
    return payload


@dataclass(frozen=True)
class RoleSpec:
    name: str
    data_kind: str
    draft_control: str
    warm_start_role: str | None
    updates: int
    max_rows: int
    max_sequence_length: int
    learning_rate: float
    gradient_accumulation: int
    seed: int
    data_seed: int


ROLE_SPECS = {
    "owner": RoleSpec(
        name="owner",
        data_kind="source_only",
        draft_control="normal",
        warm_start_role=None,
        updates=OWNER_UPDATES,
        max_rows=OWNER_MAX_ROWS,
        max_sequence_length=OWNER_MAX_SEQUENCE_LENGTH,
        learning_rate=OWNER_LEARNING_RATE,
        gradient_accumulation=OWNER_GRADIENT_ACCUMULATION,
        seed=OWNER_SEED,
        data_seed=OWNER_DATA_SEED,
    ),
    "aligned": RoleSpec(
        name="aligned",
        data_kind="natural_trajectory_revision",
        draft_control="normal",
        warm_start_role="owner",
        updates=REVISION_UPDATES,
        max_rows=REVISION_PRESENTATIONS,
        max_sequence_length=REVISION_MAX_SEQUENCE_LENGTH,
        learning_rate=REVISION_LEARNING_RATE,
        gradient_accumulation=REVISION_GRADIENT_ACCUMULATION,
        seed=REVISION_SEED,
        data_seed=REVISION_DATA_SEED,
    ),
    "draft_hidden": RoleSpec(
        name="draft_hidden",
        data_kind="natural_trajectory_revision",
        draft_control="draft_unavailable",
        warm_start_role="owner",
        updates=REVISION_UPDATES,
        max_rows=REVISION_PRESENTATIONS,
        max_sequence_length=REVISION_MAX_SEQUENCE_LENGTH,
        learning_rate=REVISION_LEARNING_RATE,
        gradient_accumulation=REVISION_GRADIENT_ACCUMULATION,
        seed=REVISION_SEED,
        data_seed=REVISION_DATA_SEED,
    ),
}


def role_spec(role: str) -> RoleSpec:
    try:
        return ROLE_SPECS[role]
    except KeyError as error:
        raise Q36MTRRoleError(f"unknown Q36-MTR role: {role}") from error


def trainable_name_sha256(names: Iterable[str]) -> str:
    ordered = sorted(names)
    if not ordered or len(ordered) != len(set(ordered)):
        raise Q36MTRRoleError("Q36-MTR trainable parameter names differ")
    return hashlib.sha256("\n".join(ordered).encode()).hexdigest()


def native_moe_surface_contract() -> dict[str, Any]:
    rows = [
        {
            "layer": index,
            "layer_type": layer_type,
            "router_top_k": ROUTER_TOP_K,
            "router_experts": NUM_EXPERTS,
            "router_hidden_size": HIDDEN_SIZE,
            "expert_count": NUM_EXPERTS,
            "expert_hidden_size": HIDDEN_SIZE,
            "expert_intermediate_size": MOE_INTERMEDIATE_SIZE,
            "shared_hidden_size": HIDDEN_SIZE,
            "shared_intermediate_size": SHARED_EXPERT_INTERMEDIATE_SIZE,
            "shared_gate_in_features": HIDDEN_SIZE,
            "shared_gate_out_features": 1,
        }
        for index, layer_type in enumerate(LAYER_TYPES)
    ]
    encoded = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    return {
        "layers": MODEL_LAYERS,
        "controlled_layer_indices": list(CONTROLLED_LAYER_INDICES),
        "native_router_expert_geometry_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def role_contract(role: str) -> dict[str, Any]:
    spec = role_spec(role)
    return {
        "architecture": ARCHITECTURE,
        "role": role,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "model_loader": "causal",
        "causal_model_class": CAUSAL_MODEL_CLASS,
        "model_type": MODEL_TYPE,
        "text_model_type": TEXT_MODEL_TYPE,
        "hidden_size": HIDDEN_SIZE,
        "model_layers": MODEL_LAYERS,
        "num_experts": NUM_EXPERTS,
        "router_top_k": ROUTER_TOP_K,
        "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
        "shared_expert_intermediate_size": SHARED_EXPERT_INTERMEDIATE_SIZE,
        "vocab_size": VOCAB_SIZE,
        "layer_types": list(LAYER_TYPES),
        "controlled_layers": CONTROLLED_LAYERS,
        "controlled_layer_indices": list(CONTROLLED_LAYER_INDICES),
        "native_moe_surface": native_moe_surface_contract(),
        "rank": RANK,
        "alpha": ALPHA,
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "quantization": QUANTIZATION,
        "compute_dtype": COMPUTE_DTYPE,
        "trainable_master_dtype": TRAINABLE_MASTER_DTYPE,
        "role_spec": asdict(spec),
        "router_expert_trainables": 0,
        "token_geometry": "identical_aligned_and_draft_hidden",
        "position_geometry": "explicit_full_sequence_positions",
        "hidden_intervention": "draft_attention_only_masked_not_deleted",
        # Prompt geometry and causal availability are separate claims.  The
        # hidden arm retains the exact draft bytes and positions while making
        # those tokens unavailable as attention keys.
        "draft_token_bytes_present": role != "owner",
        "draft_information_available": role == "aligned",
        "external_proposer": False,
        "task_router": False,
    }


def validate_backbone_geometry(backbone: Any) -> list[int]:
    """Pin the exact cached Q36 causal host before any trainable is attached."""

    config = getattr(backbone, "config", None)
    text_config = getattr(config, "text_config", None)
    observed = {
        "model_class": type(backbone).__name__,
        "model_type": getattr(config, "model_type", None),
        "text_model_type": getattr(text_config, "model_type", None),
        "hidden_size": getattr(text_config, "hidden_size", None),
        "model_layers": getattr(text_config, "num_hidden_layers", None),
        "num_experts": getattr(text_config, "num_experts", None),
        "router_top_k": getattr(text_config, "num_experts_per_tok", None),
        "moe_intermediate_size": getattr(text_config, "moe_intermediate_size", None),
        "shared_expert_intermediate_size": getattr(
            text_config, "shared_expert_intermediate_size", None
        ),
        "vocab_size": getattr(text_config, "vocab_size", None),
        "layer_types": tuple(getattr(text_config, "layer_types", ())),
    }
    expected = {
        "model_class": CAUSAL_MODEL_CLASS,
        "model_type": MODEL_TYPE,
        "text_model_type": TEXT_MODEL_TYPE,
        "hidden_size": HIDDEN_SIZE,
        "model_layers": MODEL_LAYERS,
        "num_experts": NUM_EXPERTS,
        "router_top_k": ROUTER_TOP_K,
        "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
        "shared_expert_intermediate_size": SHARED_EXPERT_INTERMEDIATE_SIZE,
        "vocab_size": VOCAB_SIZE,
        "layer_types": LAYER_TYPES,
    }
    if observed != expected:
        raise Q36MTRRoleError(
            "Q36-MTR exact causal host geometry differs: "
            f"expected={expected!r} observed={observed!r}"
        )
    return list(CONTROLLED_LAYER_INDICES)


def validate_backbone_moe_surface(backbone: Any) -> dict[str, Any]:
    """Replay the exact native router/expert structure of every loaded layer."""

    text_model = getattr(backbone, "model", None)
    layers = getattr(text_model, "layers", None)
    if not isinstance(layers, (list, tuple)) and type(layers).__name__ != "ModuleList":
        raise Q36MTRRoleError("Q36-MTR loaded text-layer surface is absent")
    if len(layers) != MODEL_LAYERS:
        raise Q36MTRRoleError("Q36-MTR loaded text-layer count differs")
    rows: list[dict[str, Any]] = []
    for index, (layer, expected_type) in enumerate(
        zip(layers, LAYER_TYPES, strict=True)
    ):
        mlp = getattr(layer, "mlp", None)
        gate = getattr(mlp, "gate", None)
        experts = getattr(mlp, "experts", None)
        shared = getattr(mlp, "shared_expert", None)
        shared_gate = getattr(mlp, "shared_expert_gate", None)
        row = {
            "layer": index,
            "layer_type": getattr(layer, "block_type", None),
            "router_top_k": getattr(gate, "top_k", None),
            "router_experts": getattr(gate, "num_experts", None),
            "router_hidden_size": getattr(gate, "hidden_dim", None),
            "expert_count": getattr(experts, "num_experts", None),
            "expert_hidden_size": getattr(experts, "hidden_dim", None),
            "expert_intermediate_size": getattr(experts, "intermediate_dim", None),
            "shared_hidden_size": getattr(shared, "hidden_size", None),
            "shared_intermediate_size": getattr(shared, "intermediate_size", None),
            "shared_gate_in_features": getattr(shared_gate, "in_features", None),
            "shared_gate_out_features": getattr(shared_gate, "out_features", None),
        }
        expected = {
            "layer": index,
            "layer_type": expected_type,
            "router_top_k": ROUTER_TOP_K,
            "router_experts": NUM_EXPERTS,
            "router_hidden_size": HIDDEN_SIZE,
            "expert_count": NUM_EXPERTS,
            "expert_hidden_size": HIDDEN_SIZE,
            "expert_intermediate_size": MOE_INTERMEDIATE_SIZE,
            "shared_hidden_size": HIDDEN_SIZE,
            "shared_intermediate_size": SHARED_EXPERT_INTERMEDIATE_SIZE,
            "shared_gate_in_features": HIDDEN_SIZE,
            "shared_gate_out_features": 1,
        }
        if row != expected:
            raise Q36MTRRoleError(
                f"Q36-MTR loaded native MoE layer differs: expected={expected!r} "
                f"observed={row!r}"
            )
        rows.append(row)
    encoded = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    receipt = {
        "layers": MODEL_LAYERS,
        "controlled_layer_indices": list(CONTROLLED_LAYER_INDICES),
        "native_router_expert_geometry_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    if receipt != native_moe_surface_contract():
        raise Q36MTRRoleError("Q36-MTR loaded native MoE surface digest differs")
    return receipt


def validate_controlled_layer_geometry(layer_count: int) -> list[int]:
    if isinstance(layer_count, bool) or layer_count != MODEL_LAYERS:
        raise Q36MTRRoleError("Q36-MTR loaded language-layer geometry differs")
    return list(CONTROLLED_LAYER_INDICES)


def validate_contract(payload: Mapping[str, Any], role: str) -> None:
    expected = role_contract(role)
    observed = {key: payload.get(key) for key in expected}
    if observed != expected:
        raise Q36MTRRoleError(
            "Q36-MTR role contract differs: "
            f"expected={json.dumps(expected, sort_keys=True)} "
            f"observed={json.dumps(observed, sort_keys=True)}"
        )


def validate_adapter_update_receipt(payload: Any) -> None:
    """Validate evidence that commit fitting changed the FP32 adapter state."""

    if not isinstance(payload, Mapping):
        raise Q36MTRRoleError("Q36-MTR adapter update receipt is absent")
    initial = payload.get("initial_state_sha256")
    final = payload.get("final_state_sha256")
    numeric = (
        payload.get("l2_delta"),
        payload.get("relative_l2_delta"),
        payload.get("maximum_absolute_delta"),
    )
    if (
        payload.get("schema") != ADAPTER_UPDATE_SCHEMA
        or not isinstance(initial, str)
        or len(initial) != 64
        or not isinstance(final, str)
        or len(final) != 64
        or initial == final
        or isinstance(payload.get("changed_tensor_count"), bool)
        or not isinstance(payload.get("changed_tensor_count"), int)
        or payload["changed_tensor_count"] <= 0
        or isinstance(payload.get("changed_parameter_count"), bool)
        or not isinstance(payload.get("changed_parameter_count"), int)
        or not 0 < payload["changed_parameter_count"] <= TRAINABLE_PARAMETERS
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in numeric
        )
        or payload.get("nonzero_finite_update") is not True
    ):
        raise Q36MTRRoleError("Q36-MTR adapter update receipt differs")


def validate_commit_gradient_receipt(payload: Mapping[str, Any]) -> None:
    minimum = payload.get("minimum_adapter_gradient_l2")
    maximum = payload.get("maximum_adapter_gradient_l2")
    if (
        payload.get("adapter_gradient_nonzero_updates") != COMMIT_UPDATES
        or isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not math.isfinite(float(minimum))
        or minimum <= 0
        or isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not math.isfinite(float(maximum))
        or maximum < minimum
    ):
        raise Q36MTRRoleError("Q36-MTR commit task-gradient receipt differs")


def validate_owner_warm_start(
    metadata: Mapping[str, Any],
    *,
    checkpoint_update: int,
    trainable_parameters: int,
    trainable_parameter_name_sha256: str,
    loaded_trainable_state_sha256: str,
) -> None:
    """Require both revisers to start from the exact source-only owner state."""

    validate_contract(metadata, "owner")
    expected = {
        "update": OWNER_UPDATES,
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "trainable_parameter_name_sha256": trainable_parameter_name_sha256,
        "selected_rows": OWNER_MAX_ROWS,
        "source_only_model_visible": True,
        "internal_draft_visible": False,
        "final_trainable_state_sha256": loaded_trainable_state_sha256,
        "serialization_restore_exact": True,
    }
    observed = {
        "update": checkpoint_update,
        "trainable_parameters": trainable_parameters,
        "trainable_parameter_name_sha256": metadata.get(
            "trainable_parameter_name_sha256"
        ),
        "selected_rows": metadata.get("selected_rows"),
        "source_only_model_visible": metadata.get("source_only_model_visible"),
        "internal_draft_visible": metadata.get("internal_draft_visible"),
        "final_trainable_state_sha256": metadata.get("final_trainable_state_sha256"),
        "serialization_restore_exact": metadata.get("serialization_restore_exact"),
    }
    if observed != expected:
        raise Q36MTRRoleError(
            "Q36-MTR owner warm start differs: "
            f"expected={expected} observed={observed}"
        )


def sequence_geometry_receipt(
    prompt_rows: list[list[int]],
    response_rows: list[list[int]],
    draft_attention_rows: list[list[int]],
) -> dict[str, Any]:
    """Hash exact token/position geometry independently of visibility control."""

    if not prompt_rows or not (
        len(prompt_rows) == len(response_rows) == len(draft_attention_rows)
    ):
        raise Q36MTRRoleError("Q36-MTR sequence batch geometry differs")
    rows: list[dict[str, Any]] = []
    masked_tokens = 0
    for prompt, response, mask in zip(
        prompt_rows, response_rows, draft_attention_rows, strict=True
    ):
        if (
            not prompt
            or not response
            or len(prompt) != len(mask)
            or any(value not in (0, 1) for value in mask)
        ):
            raise Q36MTRRoleError("Q36-MTR sequence row geometry differs")
        masked = sum(value == 0 for value in mask)
        masked_tokens += masked
        rows.append(
            {
                "prompt": prompt,
                "response": response,
                # Positions are always based on the full unmasked sequence.
                "position_ids": list(range(len(prompt) + len(response))),
                "draft_mask": mask,
            }
        )
    token_preimage = b"".join(
        (
            json.dumps({"prompt": row["prompt"], "response": row["response"]}) + "\n"
        ).encode()
        for row in rows
    )
    position_preimage = b"".join(
        (json.dumps(row["position_ids"]) + "\n").encode() for row in rows
    )
    mask_preimage = b"".join(
        (json.dumps(row["draft_mask"]) + "\n").encode() for row in rows
    )
    return {
        "rows": len(rows),
        "prompt_tokens": sum(len(row["prompt"]) for row in rows),
        "response_tokens": sum(len(row["response"]) for row in rows),
        "draft_masked_tokens": masked_tokens,
        "token_geometry_sha256": hashlib.sha256(token_preimage).hexdigest(),
        "position_geometry_sha256": hashlib.sha256(position_preimage).hexdigest(),
        "draft_attention_sha256": hashlib.sha256(mask_preimage).hexdigest(),
    }


def validate_matched_revision_geometry(
    aligned: Mapping[str, Any], draft_hidden: Mapping[str, Any]
) -> None:
    """Prove the two revisers differ only in draft visibility."""

    stable = (
        "rows",
        "prompt_tokens",
        "response_tokens",
        "token_geometry_sha256",
        "position_geometry_sha256",
    )
    if any(aligned.get(key) != draft_hidden.get(key) for key in stable):
        raise Q36MTRRoleError("Q36-MTR aligned/hidden token geometry differs")
    if aligned.get("draft_masked_tokens") != draft_hidden.get("draft_masked_tokens"):
        raise Q36MTRRoleError("Q36-MTR aligned/hidden draft span differs")
    if (
        not isinstance(aligned.get("draft_masked_tokens"), int)
        or aligned["draft_masked_tokens"] <= 0
    ):
        raise Q36MTRRoleError("Q36-MTR draft span is empty")
