"""Trainable-only owner-to-revision lineage for upward MoE temporal gates."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from upward_moe_temporal_gate import UpwardMoETemporalGateSpec

CHECKPOINT_SCHEMA = "shohin-upward-moe-role-checkpoint-v1"
METADATA_SCHEMA = "shohin-upward-moe-role-metadata-v1"
ROLE_UPDATES = 256
ROLES = ("owner", "aligned")


class UpwardMoERoleLineageError(RuntimeError):
    """The upward MoE role checkpoint or warm-start lineage differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def trainable_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def expected_state_names(spec: UpwardMoETemporalGateSpec) -> set[str]:
    return {
        f"backbone.model.layers.{index}.{spec.module_attribute}.adapter_{factor}.weight"
        for index in spec.controlled_layer_indices
        for factor in ("a", "b")
    }


def validate_role_state(
    state: Any, spec: UpwardMoETemporalGateSpec
) -> Mapping[str, torch.Tensor]:
    if not isinstance(state, Mapping) or set(state) != expected_state_names(spec):
        raise UpwardMoERoleLineageError("upward MoE role state names differ")
    for name, tensor in state.items():
        factor = "a" if name.endswith("adapter_a.weight") else "b"
        expected_shape = (
            (spec.rank, spec.hidden_size)
            if factor == "a"
            else (spec.hidden_size, spec.rank)
        )
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != expected_shape
            or tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all()
        ):
            raise UpwardMoERoleLineageError("upward MoE role tensor differs")
    return state


def _hex_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _metadata(
    *,
    role: str,
    spec: UpwardMoETemporalGateSpec,
    initial_state_sha256: str,
    final_state_sha256: str,
    warm_start_checkpoint_sha256: str | None,
    warm_start_state_sha256: str | None,
    training_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": METADATA_SCHEMA,
        "role": role,
        "host": spec.host,
        "model_revision": spec.model_revision,
        "model_config_sha256": spec.model_config_sha256,
        "attachment_surface": spec.attachment_surface,
        "module_attribute": spec.module_attribute,
        "controlled_layer_indices": list(spec.controlled_layer_indices),
        "hidden_size": spec.hidden_size,
        "rank": spec.rank,
        "alpha": spec.alpha,
        "updates": ROLE_UPDATES,
        "initial_trainable_state_sha256": initial_state_sha256,
        "final_trainable_state_sha256": final_state_sha256,
        "warm_start_checkpoint_sha256": warm_start_checkpoint_sha256,
        "warm_start_state_sha256": warm_start_state_sha256,
        "optimizer_state_serialized": False,
        "checkpoint_trainable_only": True,
        "router_expert_checkpoint_tensors": 0,
        "native_router_expert_trainables": 0,
        "training_receipt": dict(training_receipt),
    }


def save_role_checkpoint(
    path: Path,
    *,
    role: str,
    state: Mapping[str, torch.Tensor],
    spec: UpwardMoETemporalGateSpec,
    initial_state_sha256: str,
    training_receipt: Mapping[str, Any],
    warm_start_checkpoint: Path | None = None,
) -> dict[str, Any]:
    if path.exists() or path.is_symlink() or role not in ROLES:
        raise UpwardMoERoleLineageError("upward MoE checkpoint target differs")
    validate_role_state(state, spec)
    final_state_sha256 = trainable_state_sha256(state)
    if (
        not _hex_digest(initial_state_sha256)
        or initial_state_sha256 == final_state_sha256
    ):
        raise UpwardMoERoleLineageError("upward MoE role update differs")
    warm_checkpoint_sha = warm_state_sha = None
    if role == "owner":
        if warm_start_checkpoint is not None:
            raise UpwardMoERoleLineageError("upward MoE owner warm start differs")
    else:
        if warm_start_checkpoint is None:
            raise UpwardMoERoleLineageError("upward MoE aligned warm start is absent")
        warm_payload = load_role_checkpoint(warm_start_checkpoint, spec)
        warm_metadata = warm_payload["metadata"]
        if warm_metadata["role"] != "owner":
            raise UpwardMoERoleLineageError("upward MoE aligned donor differs")
        warm_checkpoint_sha = sha256_file(warm_start_checkpoint)
        warm_state_sha = warm_metadata["final_trainable_state_sha256"]
        if initial_state_sha256 != warm_state_sha:
            raise UpwardMoERoleLineageError("upward MoE aligned initial state differs")
    metadata = _metadata(
        role=role,
        spec=spec,
        initial_state_sha256=initial_state_sha256,
        final_state_sha256=final_state_sha256,
        warm_start_checkpoint_sha256=warm_checkpoint_sha,
        warm_start_state_sha256=warm_state_sha,
        training_receipt=training_receipt,
    )
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "update": ROLE_UPDATES,
        "trainable_state": dict(state),
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise UpwardMoERoleLineageError("upward MoE checkpoint temporary exists")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    restored = load_role_checkpoint(path, spec)
    if restored["metadata"] != metadata:
        raise UpwardMoERoleLineageError("upward MoE checkpoint restore differs")
    return restored


def load_role_checkpoint(path: Path, spec: UpwardMoETemporalGateSpec) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise UpwardMoERoleLineageError("upward MoE role checkpoint is absent")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "update", "trainable_state", "metadata"}
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("update") != ROLE_UPDATES
        or not isinstance(payload.get("metadata"), dict)
    ):
        raise UpwardMoERoleLineageError("upward MoE role checkpoint differs")
    state = validate_role_state(payload["trainable_state"], spec)
    metadata = payload["metadata"]
    role = metadata.get("role")
    shared = {
        "schema": METADATA_SCHEMA,
        "host": spec.host,
        "model_revision": spec.model_revision,
        "model_config_sha256": spec.model_config_sha256,
        "attachment_surface": spec.attachment_surface,
        "module_attribute": spec.module_attribute,
        "controlled_layer_indices": list(spec.controlled_layer_indices),
        "hidden_size": spec.hidden_size,
        "rank": spec.rank,
        "alpha": spec.alpha,
        "updates": ROLE_UPDATES,
        "optimizer_state_serialized": False,
        "checkpoint_trainable_only": True,
        "router_expert_checkpoint_tensors": 0,
        "native_router_expert_trainables": 0,
    }
    if (
        role not in ROLES
        or any(metadata.get(key) != value for key, value in shared.items())
        or not isinstance(metadata.get("training_receipt"), dict)
        or not metadata["training_receipt"]
        or not _hex_digest(metadata.get("initial_trainable_state_sha256"))
        or metadata["initial_trainable_state_sha256"]
        == metadata.get("final_trainable_state_sha256")
        or metadata.get("final_trainable_state_sha256") != trainable_state_sha256(state)
    ):
        raise UpwardMoERoleLineageError("upward MoE role metadata differs")
    warm_checkpoint = metadata.get("warm_start_checkpoint_sha256")
    warm_state = metadata.get("warm_start_state_sha256")
    if (
        role == "owner" and (warm_checkpoint is not None or warm_state is not None)
    ) or (
        role == "aligned"
        and (not _hex_digest(warm_checkpoint) or not _hex_digest(warm_state))
    ):
        raise UpwardMoERoleLineageError("upward MoE role warm-start metadata differs")
    return payload


def load_role_pair(
    owner_checkpoint: Path,
    revision_checkpoint: Path,
    spec: UpwardMoETemporalGateSpec,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor], dict[str, Any]]:
    owner = load_role_checkpoint(owner_checkpoint, spec)
    revision = load_role_checkpoint(revision_checkpoint, spec)
    owner_metadata = owner["metadata"]
    revision_metadata = revision["metadata"]
    owner_checkpoint_sha = sha256_file(owner_checkpoint)
    owner_state_sha = owner_metadata["final_trainable_state_sha256"]
    revision_state_sha = revision_metadata["final_trainable_state_sha256"]
    if (
        owner_metadata["role"] != "owner"
        or revision_metadata["role"] != "aligned"
        or revision_metadata["warm_start_checkpoint_sha256"] != owner_checkpoint_sha
        or revision_metadata["warm_start_state_sha256"] != owner_state_sha
        or revision_metadata["initial_trainable_state_sha256"] != owner_state_sha
        or revision_state_sha == owner_state_sha
    ):
        raise UpwardMoERoleLineageError("upward MoE owner/revision lineage differs")
    return (
        owner["trainable_state"],
        revision["trainable_state"],
        {
            "host": spec.host,
            "model_revision": spec.model_revision,
            "owner_checkpoint_sha256": owner_checkpoint_sha,
            "revision_checkpoint_sha256": sha256_file(revision_checkpoint),
            "owner_state_sha256": owner_state_sha,
            "revision_state_sha256": revision_state_sha,
            "warm_start_exact": True,
            "native_router_expert_trainables": 0,
        },
    )


__all__ = [
    "CHECKPOINT_SCHEMA",
    "METADATA_SCHEMA",
    "UpwardMoERoleLineageError",
    "expected_state_names",
    "load_role_checkpoint",
    "load_role_pair",
    "save_role_checkpoint",
    "sha256_file",
    "trainable_state_sha256",
    "validate_role_state",
]
