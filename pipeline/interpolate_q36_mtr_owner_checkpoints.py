#!/usr/bin/env python3
"""Interpolate two compatible Q36 owner adapters for engineering evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from q36_mtr_roles import (
    OWNER_UPDATES,
    ROLE_CHECKPOINT_SCHEMA,
    TRAINABLE_PARAMETERS,
    validate_contract,
)
from shared_post_mlp_revision import trainable_state_sha256


class Q36MTROwnerInterpolationError(RuntimeError):
    """Raised when owner checkpoints cannot be interpolated exactly."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError) as error:
        raise Q36MTROwnerInterpolationError(
            f"owner checkpoint is unreadable: {path}"
        ) from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema", "update", "trainable_state", "metadata"}
        or payload.get("schema") != ROLE_CHECKPOINT_SCHEMA
        or payload.get("update") != OWNER_UPDATES
        or not isinstance(payload.get("trainable_state"), dict)
        or not isinstance(payload.get("metadata"), dict)
    ):
        raise Q36MTROwnerInterpolationError("owner checkpoint schema differs")
    validate_contract(payload["metadata"], "owner")
    if payload["metadata"].get(
        "final_trainable_state_sha256"
    ) != trainable_state_sha256(payload["trainable_state"]):
        raise Q36MTROwnerInterpolationError("owner checkpoint state receipt differs")
    return payload


def interpolate(
    first_path: Path,
    second_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    second_weight: float,
) -> dict[str, Any]:
    import torch

    if (
        not math.isfinite(second_weight)
        or not 0.0 < second_weight < 1.0
        or first_path.resolve() == second_path.resolve()
        or output_path.exists()
        or output_path.is_symlink()
        or report_path.exists()
        or report_path.is_symlink()
    ):
        raise Q36MTROwnerInterpolationError("owner interpolation settings differ")
    first = _load(first_path)
    second = _load(second_path)
    first_state = first["trainable_state"]
    second_state = second["trainable_state"]
    if set(first_state) != set(second_state):
        raise Q36MTROwnerInterpolationError("owner interpolation tensor names differ")
    output_state = {}
    squared_delta = squared_reference = 0.0
    for name in sorted(first_state):
        left = first_state[name]
        right = second_state[name]
        if (
            not torch.is_tensor(left)
            or not torch.is_tensor(right)
            or left.shape != right.shape
            or left.dtype != right.dtype
            or not left.dtype.is_floating_point
        ):
            raise Q36MTROwnerInterpolationError(
                "owner interpolation tensor geometry differs"
            )
        left_fp32 = left.float()
        right_fp32 = right.float()
        output_state[name] = (
            left_fp32.mul(1.0 - second_weight).add(right_fp32, alpha=second_weight)
        ).to(dtype=left.dtype)
        delta = left_fp32 - right_fp32
        squared_delta += float((delta * delta).sum())
        squared_reference += float((left_fp32 * left_fp32).sum())
    parameters = sum(tensor.numel() for tensor in output_state.values())
    if parameters != TRAINABLE_PARAMETERS:
        raise Q36MTROwnerInterpolationError(
            "owner interpolation parameter count differs"
        )
    state_sha256 = trainable_state_sha256(output_state)
    metadata = dict(first["metadata"])
    metadata.update(
        {
            "final_trainable_state_sha256": state_sha256,
            "interpolation": {
                "schema": "shohin-q36-mtr-owner-interpolation-v1",
                "first_checkpoint_sha256": sha256_file(first_path),
                "second_checkpoint_sha256": sha256_file(second_path),
                "second_weight": second_weight,
            },
        }
    )
    payload = {
        "schema": ROLE_CHECKPOINT_SCHEMA,
        "update": OWNER_UPDATES,
        "trainable_state": output_state,
        "metadata": metadata,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    output_sha256 = sha256_file(output_path)
    report = {
        "schema": "shohin-q36-mtr-owner-interpolation-report-v1",
        "status": "complete",
        "interpretation": "exploratory_owner_weight_space_interpolation",
        "first_checkpoint": str(first_path.resolve()),
        "first_checkpoint_sha256": sha256_file(first_path),
        "second_checkpoint": str(second_path.resolve()),
        "second_checkpoint_sha256": sha256_file(second_path),
        "second_weight": second_weight,
        "parameters": parameters,
        "endpoint_relative_l2": math.sqrt(squared_delta / squared_reference),
        "output_checkpoint": str(output_path.resolve()),
        "output_checkpoint_sha256": output_sha256,
        "output_trainable_state_sha256": state_sha256,
    }
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_name(f".{report_path.name}.tmp.{os.getpid()}")
    with temporary_report.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_report, report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--second-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = interpolate(
        args.first,
        args.second,
        args.output,
        args.report,
        second_weight=args.second_weight,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
