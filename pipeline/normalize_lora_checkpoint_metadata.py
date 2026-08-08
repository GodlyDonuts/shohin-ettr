#!/usr/bin/env python3
"""Normalize legacy frozen-trunk LoRA checkpoint metadata without changing state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Iterable

import torch


SCHEMA = "shohin-lora-checkpoint-metadata-normalization-v1"


class CheckpointNormalizationError(ValueError):
    """A checkpoint is not eligible for the metadata-only migration."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> None:
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise CheckpointNormalizationError(f"{label} is not a physical file")


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left, right)
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _equal(first, second) for first, second in zip(left, right)
        )
    return bool(left == right)


def normalize_checkpoint(
    *,
    source: Path,
    source_sha256: str,
    output: Path,
    report: Path,
) -> dict[str, Any]:
    _regular_file(source, "source checkpoint")
    if _sha256(source) != source_sha256:
        raise CheckpointNormalizationError("source checkpoint SHA-256 differs")
    for path in (output, report):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing existing output: {path}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise CheckpointNormalizationError("checkpoint is not an object")
    metadata = checkpoint.get("metadata")
    trainable_state = checkpoint.get("trainable_state")
    if (
        not isinstance(metadata, dict)
        or metadata.get("unfreeze_layers") is not None
        or not isinstance(trainable_state, dict)
        or not trainable_state
        or any(
            ".lora_a." not in name and ".lora_b." not in name
            for name in trainable_state
        )
    ):
        raise CheckpointNormalizationError(
            "checkpoint is not a legacy LoRA-only frozen-trunk state"
        )
    original = dict(checkpoint)
    original_metadata = dict(metadata)
    checkpoint["metadata"] = {**metadata, "unfreeze_layers": 0}

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary_report = report.with_name(f".{report.name}.tmp.{os.getpid()}")
    try:
        torch.save(checkpoint, temporary_output)
        migrated = torch.load(temporary_output, map_location="cpu", weights_only=False)
        if not isinstance(migrated, dict) or migrated.keys() != checkpoint.keys():
            raise CheckpointNormalizationError("migrated checkpoint keys differ")
        migrated_metadata = migrated.get("metadata")
        if migrated_metadata != checkpoint["metadata"]:
            raise CheckpointNormalizationError("migrated metadata differs")
        for key in checkpoint:
            if key == "metadata":
                continue
            if not _equal(original[key], migrated[key]):
                raise CheckpointNormalizationError(
                    f"checkpoint state changed during migration: {key}"
                )
        output_sha256 = _sha256(temporary_output)
        receipt = {
            "schema": SCHEMA,
            "source": {
                "path": str(source.resolve()),
                "bytes": source.stat().st_size,
                "sha256": source_sha256,
            },
            "output": {
                "path": str(output.resolve()),
                "bytes": temporary_output.stat().st_size,
                "sha256": output_sha256,
            },
            "changed_field": "metadata.unfreeze_layers",
            "source_value": original_metadata.get("unfreeze_layers"),
            "target_value": 0,
            "trainable_state_tensors": len(trainable_state),
            "all_non_metadata_state_bitwise_equal": True,
        }
        payload = json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        receipt["payload_sha256"] = hashlib.sha256(payload).hexdigest()
        report.parent.mkdir(parents=True, exist_ok=True)
        with temporary_report.open("x", encoding="ascii") as destination:
            json.dump(receipt, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_output, output)
        os.replace(temporary_report, report)
        return receipt
    finally:
        temporary_output.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    arguments = parser.parse_args(argv)
    receipt = normalize_checkpoint(
        source=arguments.source,
        source_sha256=arguments.source_sha256,
        output=arguments.output,
        report=arguments.report,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
