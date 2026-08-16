#!/usr/bin/env python3
"""Lift a trained Nemotron Super Shohin residual into Nemotron Ultra space."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

SCHEMA = "shohin-nemotron-super-ultra-adapter-transfer-v1"
CHECKPOINT_SCHEMA = "shohin-nemotron-ultra-transferred-revision-checkpoint-v1"
SUPER_CHECKPOINT_SCHEMA = "shohin-nemotron-super-revision-checkpoint-v1"
SUPER_MODEL_REVISION = "7d7e5797b8a3c7abbab54033b6004e93e8b6bc91"
ULTRA_MODEL_REVISION = "183968f87ae4cedce3039313cac1fd43d112c578"
ULTRA_CONFIG_SHA256 = "0c939f324c8910f5ebdafbe2a56d7e4e074c50042a3b4f26326bf71a3fe33929"
FACTOR_SHA256 = "74c65a5ee08d058f565c2598bc3005f7b441977d16ed7bc99ea978468a5c617b"
SUPER_HIDDEN_SIZE = 4096
ULTRA_HIDDEN_SIZE = 8192
RANK = 18
ALPHA = 18.0
ULTRA_MODEL_LAYERS = 108
CONTROLLED_LAYERS = 16
SUPER_LAYERS = (54, 56, 59, 61, 63, 65, 67, 70, 72, 74, 76, 79, 81, 83, 85, 87)
ULTRA_LAYERS = (74, 76, 78, 80, 83, 85, 87, 90, 92, 94, 96, 99, 101, 103, 105, 107)
DATA_SHA256 = "802c85662570c5bcb72f3e4430dbd093e901081f114213831292750894c3feff"
NAME = re.compile(r"^backbone\.model\.layers\.(\d+)\.mixer\.adapter_([ab])\.weight$")


class UltraAdapterTransferError(RuntimeError):
    """The zero-label Super-to-Ultra adapter transfer contract failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise UltraAdapterTransferError("input is absent, symbolic, or nonregular")
    return resolved


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def transfer_directions(
    directions: torch.Tensor,
    super_anchor: torch.Tensor,
    ultra_anchor: torch.Tensor,
    cholesky: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transfer columns while preserving each source direction's L2 norm."""
    if (
        directions.ndim != 2
        or super_anchor.ndim != 2
        or ultra_anchor.ndim != 2
        or cholesky.ndim != 2
        or directions.shape[0] != super_anchor.shape[1]
        or super_anchor.shape[0] != ultra_anchor.shape[0]
        or cholesky.shape != (super_anchor.shape[0], super_anchor.shape[0])
    ):
        raise UltraAdapterTransferError("transfer direction geometry differs")
    source = directions.float()
    norms = source.norm(dim=0)
    if not torch.isfinite(source).all() or (norms <= 0).any():
        raise UltraAdapterTransferError("source direction is zero or nonfinite")
    unit = source / norms
    scores = super_anchor.float() @ unit
    solved = torch.cholesky_solve(scores, cholesky.float())
    mapped_unit = F.normalize(ultra_anchor.float().T @ solved, dim=0)
    mapped = mapped_unit * norms
    correlations = F.cosine_similarity(
        scores - scores.mean(dim=0, keepdim=True),
        (ultra_anchor.float() @ mapped_unit)
        - (ultra_anchor.float() @ mapped_unit).mean(dim=0, keepdim=True),
        dim=0,
    )
    if not torch.isfinite(mapped).all() or not torch.isfinite(correlations).all():
        raise UltraAdapterTransferError("transferred direction is nonfinite")
    return mapped, correlations


def _validate_ultra_config(path: Path) -> None:
    path = _regular(path)
    if sha256_file(path) != ULTRA_CONFIG_SHA256:
        raise UltraAdapterTransferError("Ultra config hash differs")
    payload = json.loads(path.read_text(encoding="utf-8"))
    block_types = payload.get("layers_block_type")
    observed_layers = (
        tuple(index for index, value in enumerate(block_types) if value == "moe")[
            -CONTROLLED_LAYERS:
        ]
        if isinstance(block_types, list)
        else ()
    )
    if (
        payload.get("architectures") != ["NemotronHForCausalLM"]
        or payload.get("model_type") != "nemotron_h"
        or payload.get("hidden_size") != ULTRA_HIDDEN_SIZE
        or len(block_types or []) != ULTRA_MODEL_LAYERS
        or payload.get("n_routed_experts") != 512
        or payload.get("num_experts_per_tok") != 22
        or observed_layers != ULTRA_LAYERS
    ):
        raise UltraAdapterTransferError("Ultra host geometry differs")


def _load_factor(path: Path, expected_sha256: str) -> dict[str, torch.Tensor]:
    path = _regular(path)
    if sha256_file(path) != expected_sha256:
        raise UltraAdapterTransferError("transfer factor hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    factor = payload.get("factor") if isinstance(payload, dict) else None
    if (
        payload.get("schema") != "shohin-nemotron-super-ultra-transfer-basis-v1"
        or payload.get("label_rows_read") not in (None, 0)
        or payload.get("benchmark_rows_read") not in (None, 0)
        or not isinstance(factor, dict)
        or set(factor)
        != {
            "anchor_ids",
            "super_anchor",
            "ultra_anchor",
            "ultra_kernel_cholesky",
        }
    ):
        raise UltraAdapterTransferError("transfer factor payload differs")
    anchors = factor["anchor_ids"]
    super_anchor = factor["super_anchor"]
    ultra_anchor = factor["ultra_anchor"]
    cholesky = factor["ultra_kernel_cholesky"]
    count = int(anchors.numel()) if isinstance(anchors, torch.Tensor) else -1
    if (
        count < 16
        or tuple(super_anchor.shape) != (count, SUPER_HIDDEN_SIZE)
        or tuple(ultra_anchor.shape) != (count, ULTRA_HIDDEN_SIZE)
        or tuple(cholesky.shape) != (count, count)
        or len(set(int(value) for value in anchors.tolist())) != count
    ):
        raise UltraAdapterTransferError("transfer factor geometry differs")
    return factor


def _load_checkpoint(path: Path, expected_sha256: str) -> dict[str, Any]:
    path = _regular(path)
    if sha256_file(path) != expected_sha256:
        raise UltraAdapterTransferError("Super checkpoint hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    state = payload.get("trainable_state") if isinstance(payload, dict) else None
    if (
        payload.get("schema") != SUPER_CHECKPOINT_SCHEMA
        or payload.get("update") != 256
        or not isinstance(metadata, dict)
        or metadata.get("model_revision") != SUPER_MODEL_REVISION
        or metadata.get("data_sha256") != DATA_SHA256
        or metadata.get("native_router_expert_trainables") != 0
        or not isinstance(state, dict)
    ):
        raise UltraAdapterTransferError("Super checkpoint metadata differs")
    expected_names = {
        f"backbone.model.layers.{layer}.mixer.adapter_{kind}.weight"
        for layer in SUPER_LAYERS
        for kind in ("a", "b")
    }
    if set(state) != expected_names:
        raise UltraAdapterTransferError("Super checkpoint names differ")
    for name, tensor in state.items():
        match = NAME.fullmatch(name)
        expected_shape = (
            (RANK, SUPER_HIDDEN_SIZE)
            if match and match.group(2) == "a"
            else (SUPER_HIDDEN_SIZE, RANK)
        )
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != expected_shape
        ):
            raise UltraAdapterTransferError("Super checkpoint tensor differs")
    return payload


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise UltraAdapterTransferError("refusing existing checkpoint output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise UltraAdapterTransferError("refusing existing report output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_ultra_config(args.ultra_config)
    factor = _load_factor(args.factor, args.expected_factor_sha256)
    checkpoint = _load_checkpoint(
        args.super_checkpoint, args.expected_super_checkpoint_sha256
    )
    super_anchor = factor["super_anchor"]
    ultra_anchor = factor["ultra_anchor"]
    cholesky = factor["ultra_kernel_cholesky"]
    source_state = checkpoint["trainable_state"]
    target_state: dict[str, torch.Tensor] = {}
    correlations: list[torch.Tensor] = []
    for super_layer, ultra_layer in zip(SUPER_LAYERS, ULTRA_LAYERS, strict=True):
        prefix = f"backbone.model.layers.{super_layer}.mixer"
        target_prefix = f"backbone.model.layers.{ultra_layer}.mixer"
        adapter_a = source_state[f"{prefix}.adapter_a.weight"]
        adapter_b = source_state[f"{prefix}.adapter_b.weight"]
        mapped_a, corr_a = transfer_directions(
            adapter_a.T, super_anchor, ultra_anchor, cholesky
        )
        mapped_b, corr_b = transfer_directions(
            adapter_b, super_anchor, ultra_anchor, cholesky
        )
        target_state[f"{target_prefix}.adapter_a.weight"] = mapped_a.T.contiguous()
        target_state[f"{target_prefix}.adapter_b.weight"] = mapped_b.contiguous()
        correlations.extend((corr_a, corr_b))
    all_correlations = torch.cat(correlations)
    if len(target_state) != len(ULTRA_LAYERS) * 2 or sum(
        x.numel() for x in target_state.values()
    ) != (len(ULTRA_LAYERS) * 2 * ULTRA_HIDDEN_SIZE * RANK):
        raise UltraAdapterTransferError("Ultra checkpoint geometry differs")
    source_sha256 = sha256_file(args.super_checkpoint)
    factor_sha256 = sha256_file(args.factor)
    metadata = {
        "schema": SCHEMA,
        "source_model_revision": SUPER_MODEL_REVISION,
        "target_model_revision": ULTRA_MODEL_REVISION,
        "source_checkpoint_sha256": source_sha256,
        "source_trainable_state_sha256": _state_sha256(source_state),
        "factor_sha256": factor_sha256,
        "ultra_config_sha256": ULTRA_CONFIG_SHA256,
        "super_controlled_layers": list(SUPER_LAYERS),
        "ultra_controlled_layers": list(ULTRA_LAYERS),
        "rank": RANK,
        "alpha": ALPHA,
        "trainable_parameters": sum(x.numel() for x in target_state.values()),
        "target_trainable_state_sha256": _state_sha256(target_state),
        "anchor_direction_correlation_mean": float(all_correlations.mean()),
        "anchor_direction_correlation_min": float(all_correlations.min()),
        "anchor_direction_correlation_max": float(all_correlations.max()),
        "label_rows_read": 0,
        "benchmark_rows_read": 0,
        "optimizer_updates": 0,
        "model_weight_mutations": 0,
        "native_router_expert_trainables": 0,
    }
    if not all(
        math.isfinite(metadata[key])
        for key in (
            "anchor_direction_correlation_mean",
            "anchor_direction_correlation_min",
            "anchor_direction_correlation_max",
        )
    ):
        raise UltraAdapterTransferError("transfer metrics are nonfinite")
    _atomic_torch(
        args.output_checkpoint,
        {
            "schema": CHECKPOINT_SCHEMA,
            "trainable_state": target_state,
            "metadata": metadata,
        },
    )
    report = {
        **metadata,
        "status": "complete",
        "checkpoint": str(args.output_checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.output_checkpoint),
    }
    _atomic_json(args.output_report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--super-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-super-checkpoint-sha256", required=True)
    parser.add_argument("--factor", type=Path, required=True)
    parser.add_argument("--expected-factor-sha256", default=FACTOR_SHA256)
    parser.add_argument("--ultra-config", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
