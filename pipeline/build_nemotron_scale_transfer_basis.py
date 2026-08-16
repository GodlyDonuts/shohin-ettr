#!/usr/bin/env python3
"""Build a label-free Super-to-Ultra hidden-space transfer factor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import safe_open

SCHEMA = "shohin-nemotron-super-ultra-transfer-basis-v1"
SEED = 2026081523
EMBEDDING_KEY = "backbone.embeddings.weight"
SUPER_EMBEDDING_SHA256 = (
    "f30de33bed00fac1b451f06ef82e64468976ee52a811cb429bc3adc34cc48c5b"
)
ULTRA_EMBEDDING_SHA256 = (
    "eea43a334457de1c87902bcd9c5621d206aba8e8e45fe39abc7752f31b6c6f5b"
)
TOKENIZER_SHA256 = "623c34567aebb18582765289fbe23d901c62704d6518d71866e0e58db892b5b7"
SUPER_SHAPE = (131072, 4096)
ULTRA_SHAPE = (131072, 8192)


class ScaleTransferBasisError(RuntimeError):
    """The label-free scale-transfer basis contract failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ScaleTransferBasisError("input is absent, symbolic, or nonregular")
    return resolved


def _embedding(
    path: Path, expected_sha256: str, shape: tuple[int, int]
) -> torch.Tensor:
    path = _regular(path)
    if sha256_file(path) != expected_sha256:
        raise ScaleTransferBasisError("embedding shard hash differs")
    with safe_open(path, framework="pt", device="cpu") as handle:
        if EMBEDDING_KEY not in handle.keys():
            raise ScaleTransferBasisError("embedding tensor is absent")
        value = handle.get_tensor(EMBEDDING_KEY)
    if tuple(value.shape) != shape or value.dtype != torch.bfloat16:
        raise ScaleTransferBasisError("embedding tensor geometry differs")
    return value


def choose_identities(
    vocab_size: int,
    anchor_count: int,
    holdout_count: int,
    excluded: set[int],
    seed: int = SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    if anchor_count < 16 or holdout_count < 16:
        raise ScaleTransferBasisError("transfer identity geometry is too small")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(vocab_size, generator=generator)
    keep = torch.tensor(
        [int(value) not in excluded for value in order], dtype=torch.bool
    )
    selected = order[keep]
    if selected.numel() < anchor_count + holdout_count:
        raise ScaleTransferBasisError("transfer identities are insufficient")
    return (
        selected[:anchor_count],
        selected[anchor_count : anchor_count + holdout_count],
    )


def build_factor(
    super_embedding: torch.Tensor,
    ultra_embedding: torch.Tensor,
    anchor_ids: torch.Tensor,
    holdout_ids: torch.Tensor,
    ridge: float,
    validation_directions: int,
    seed: int = SEED,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if (
        super_embedding.ndim != 2
        or ultra_embedding.ndim != 2
        or super_embedding.shape[0] != ultra_embedding.shape[0]
        or anchor_ids.ndim != 1
        or holdout_ids.ndim != 1
        or ridge <= 0.0
        or validation_directions < 8
    ):
        raise ScaleTransferBasisError("transfer factor inputs differ")
    super_anchor = F.normalize(super_embedding[anchor_ids].float(), dim=-1)
    ultra_anchor = F.normalize(ultra_embedding[anchor_ids].float(), dim=-1)
    super_holdout = F.normalize(super_embedding[holdout_ids].float(), dim=-1)
    ultra_holdout = F.normalize(ultra_embedding[holdout_ids].float(), dim=-1)

    kernel = ultra_anchor @ ultra_anchor.T
    ridge_scale = float(kernel.diagonal().mean()) * ridge
    kernel.diagonal().add_(ridge_scale)
    cholesky = torch.linalg.cholesky(kernel)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    coefficients = torch.randn(
        super_anchor.shape[0], validation_directions, generator=generator
    )
    super_directions = F.normalize(super_anchor.T @ coefficients, dim=0)
    anchor_scores = super_anchor @ super_directions
    solved = torch.cholesky_solve(anchor_scores, cholesky)
    ultra_directions = F.normalize(ultra_anchor.T @ solved, dim=0)
    super_scores = super_holdout @ super_directions
    ultra_scores = ultra_holdout @ ultra_directions
    super_scores = super_scores - super_scores.mean(dim=0, keepdim=True)
    ultra_scores = ultra_scores - ultra_scores.mean(dim=0, keepdim=True)
    correlations = F.cosine_similarity(super_scores, ultra_scores, dim=0)
    if not torch.isfinite(correlations).all():
        raise ScaleTransferBasisError("transfer validation is nonfinite")

    factor = {
        "anchor_ids": anchor_ids.to(torch.int64),
        "super_anchor": super_anchor.to(torch.bfloat16),
        "ultra_anchor": ultra_anchor.to(torch.bfloat16),
        "ultra_kernel_cholesky": cholesky.to(torch.float32),
    }
    metrics = {
        "ridge": ridge,
        "ridge_scale": ridge_scale,
        "validation_directions": validation_directions,
        "holdout_correlation_mean": float(correlations.mean()),
        "holdout_correlation_min": float(correlations.min()),
        "holdout_correlation_max": float(correlations.max()),
        "holdout_correlation_std": float(correlations.std()),
    }
    return factor, metrics


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ScaleTransferBasisError("refusing existing factor output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ScaleTransferBasisError("refusing existing report output")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    super_tokenizer = _regular(args.super_tokenizer)
    ultra_tokenizer = _regular(args.ultra_tokenizer)
    if (
        sha256_file(super_tokenizer) != TOKENIZER_SHA256
        or sha256_file(ultra_tokenizer) != TOKENIZER_SHA256
    ):
        raise ScaleTransferBasisError("Super and Ultra tokenizer identity differs")
    super_embedding = _embedding(
        args.super_embedding, SUPER_EMBEDDING_SHA256, SUPER_SHAPE
    )
    ultra_embedding = _embedding(
        args.ultra_embedding, ULTRA_EMBEDDING_SHA256, ULTRA_SHAPE
    )
    anchor_ids, holdout_ids = choose_identities(
        SUPER_SHAPE[0],
        args.anchor_count,
        args.holdout_count,
        set(args.exclude_token_id),
    )
    factor, metrics = build_factor(
        super_embedding,
        ultra_embedding,
        anchor_ids,
        holdout_ids,
        args.ridge,
        args.validation_directions,
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "seed": SEED,
        "super_embedding_sha256": SUPER_EMBEDDING_SHA256,
        "ultra_embedding_sha256": ULTRA_EMBEDDING_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "super_shape": list(SUPER_SHAPE),
        "ultra_shape": list(ULTRA_SHAPE),
        "anchor_count": args.anchor_count,
        "holdout_count": args.holdout_count,
        "excluded_token_ids": sorted(args.exclude_token_id),
        "metrics": metrics,
        "factor": factor,
    }
    _atomic_torch(args.output_factor, payload)
    report = {key: value for key, value in payload.items() if key != "factor"}
    report.update(
        {
            "factor_sha256": sha256_file(args.output_factor),
            "factor_bytes": args.output_factor.stat().st_size,
            "label_rows_read": 0,
            "benchmark_rows_read": 0,
            "model_weight_mutations": 0,
        }
    )
    _atomic_json(args.output_report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--super-embedding", type=Path, required=True)
    parser.add_argument("--ultra-embedding", type=Path, required=True)
    parser.add_argument("--super-tokenizer", type=Path, required=True)
    parser.add_argument("--ultra-tokenizer", type=Path, required=True)
    parser.add_argument("--output-factor", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--anchor-count", type=int, default=1024)
    parser.add_argument("--holdout-count", type=int, default=4096)
    parser.add_argument("--validation-directions", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--exclude-token-id", type=int, action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
