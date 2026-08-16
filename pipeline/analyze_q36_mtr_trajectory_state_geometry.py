#!/usr/bin/env python3
"""Measure factorization-invariant geometry of the three Q36 trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

from q36_mtr_roles import (
    CONTROLLED_LAYER_INDICES,
    MODEL_CONFIG_SHA256,
    MODEL_REVISION,
    load_role_checkpoint_payload,
)

SCHEMA = "shohin-q36-mtr-trajectory-state-geometry-v1"
ROLES = ("owner", "revision", "draft_hidden")
PAIRS = (
    ("owner", "revision"),
    ("owner", "draft_hidden"),
    ("revision", "draft_hidden"),
)


class Q36MTRTrajectoryGeometryError(RuntimeError):
    """The role lineage or factorization-invariant analysis differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _factors(
    state: dict[str, torch.Tensor], layer: int
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = f"backbone.model.layers.{layer}.mlp.adapter_"
    try:
        factor_a = state[prefix + "a.weight"].double()
        factor_b = state[prefix + "b.weight"].double()
    except KeyError as error:
        raise Q36MTRTrajectoryGeometryError("trajectory factor is absent") from error
    if (
        factor_a.ndim != 2
        or factor_b.ndim != 2
        or factor_a.shape[0] != factor_b.shape[1]
        or factor_a.shape[1] != factor_b.shape[0]
        or not torch.isfinite(factor_a).all()
        or not torch.isfinite(factor_b).all()
    ):
        raise Q36MTRTrajectoryGeometryError("trajectory factor geometry differs")
    return factor_a, factor_b


def _operator_inner(
    left: tuple[torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor],
) -> float:
    """Return <B1 A1, B2 A2> without materializing a hidden-square matrix."""

    left_a, left_b = left
    right_a, right_b = right
    if left_a.shape != right_a.shape or left_b.shape != right_b.shape:
        raise Q36MTRTrajectoryGeometryError("trajectory operator geometry differs")
    value = ((left_b.T @ right_b) * (left_a @ right_a.T)).sum()
    result = float(value)
    if not math.isfinite(result):
        raise Q36MTRTrajectoryGeometryError("trajectory operator inner is nonfinite")
    return result


def _operator_norm(factors: tuple[torch.Tensor, torch.Tensor]) -> float:
    return math.sqrt(max(_operator_inner(factors, factors), 0.0))


def _effective_rank(factors: tuple[torch.Tensor, torch.Tensor]) -> float:
    factor_a, factor_b = factors
    _, triangular_b = torch.linalg.qr(factor_b, mode="reduced")
    _, triangular_a = torch.linalg.qr(factor_a.T, mode="reduced")
    singular = torch.linalg.svdvals(triangular_b @ triangular_a.T)
    total = float(singular.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise Q36MTRTrajectoryGeometryError("trajectory singular spectrum differs")
    probability = singular / total
    entropy = -(probability * probability.log()).sum()
    result = float(torch.exp(entropy))
    if not math.isfinite(result):
        raise Q36MTRTrajectoryGeometryError("trajectory effective rank is nonfinite")
    return result


def analyze_states(
    states: dict[str, dict[str, torch.Tensor]],
    controlled_layers: tuple[int, ...] = CONTROLLED_LAYER_INDICES,
) -> dict[str, Any]:
    if tuple(states) != ROLES or not controlled_layers:
        raise Q36MTRTrajectoryGeometryError("trajectory role order differs")
    rows = []
    for layer in controlled_layers:
        factors = {role: _factors(states[role], layer) for role in ROLES}
        norms = {role: _operator_norm(factors[role]) for role in ROLES}
        pair_metrics = {}
        for left, right in PAIRS:
            inner = _operator_inner(factors[left], factors[right])
            cosine = inner / (norms[left] * norms[right])
            delta = math.sqrt(
                max(norms[left] ** 2 + norms[right] ** 2 - 2.0 * inner, 0.0)
            )
            pair_metrics[f"{left}_vs_{right}"] = {
                "operator_cosine": cosine,
                "relative_delta_to_left": delta / norms[left],
            }
        rows.append(
            {
                "layer": layer,
                "operator_norms": norms,
                "effective_ranks": {
                    role: _effective_rank(factors[role]) for role in ROLES
                },
                "pairs": pair_metrics,
            }
        )
    aggregate = {}
    for left, right in PAIRS:
        name = f"{left}_vs_{right}"
        cosines = [row["pairs"][name]["operator_cosine"] for row in rows]
        deltas = [row["pairs"][name]["relative_delta_to_left"] for row in rows]
        aggregate[name] = {
            "operator_cosine_min": min(cosines),
            "operator_cosine_mean": fmean(cosines),
            "operator_cosine_max": max(cosines),
            "relative_delta_min": min(deltas),
            "relative_delta_mean": fmean(deltas),
            "relative_delta_max": max(deltas),
            "early_four_relative_delta_mean": fmean(deltas[:4]),
            "late_four_relative_delta_mean": fmean(deltas[-4:]),
        }
    return {"layers": rows, "aggregate_pairs": aggregate}


def _load_roles(paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    payloads = {role: load_role_checkpoint_payload(paths[role]) for role in ROLES}
    metadata = {role: payloads[role]["metadata"] for role in ROLES}
    if (
        metadata["owner"].get("role") != "owner"
        or metadata["revision"].get("role") != "aligned"
        or metadata["draft_hidden"].get("role") != "draft_hidden"
        or any(
            values.get("model_revision") != MODEL_REVISION
            or values.get("model_config_sha256") != MODEL_CONFIG_SHA256
            or values.get("controlled_layer_indices") != list(CONTROLLED_LAYER_INDICES)
            for values in metadata.values()
        )
    ):
        raise Q36MTRTrajectoryGeometryError("trajectory role lineage differs")
    return (
        {role: payloads[role]["trainable_state"] for role in ROLES},
        {
            role: {
                "path": str(paths[role].resolve()),
                "sha256": sha256_file(paths[role]),
                "final_trainable_state_sha256": metadata[role][
                    "final_trainable_state_sha256"
                ],
            }
            for role in ROLES
        },
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRTrajectoryGeometryError("trajectory geometry output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    states, receipts = _load_roles(
        {
            "owner": args.owner_checkpoint,
            "revision": args.revision_checkpoint,
            "draft_hidden": args.draft_hidden_checkpoint,
        }
    )
    analysis = analyze_states(states)
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "model_revision": MODEL_REVISION,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "controlled_layer_indices": list(CONTROLLED_LAYER_INDICES),
        "checkpoint_receipts": receipts,
        **analysis,
        "interpretation": {
            "flat_router_limitation": "owner separation and revision-vs-draft sibling discrimination have materially different operator scales",
            "next_architecture": "hierarchical owner-vs-adapted then revision-vs-draft token routing",
        },
    }
    _atomic_json(args.output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-checkpoint", type=Path, required=True)
    parser.add_argument("--revision-checkpoint", type=Path, required=True)
    parser.add_argument("--draft-hidden-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result["aggregate_pairs"], sort_keys=True))


if __name__ == "__main__":
    main()
