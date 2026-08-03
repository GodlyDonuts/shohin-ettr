#!/usr/bin/env python3
"""Measure exact held-in fit before and after a product-reasoning checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any

import torch

from hf_product_reasoning_train import (
    ProductReasoningModel,
    _tokenize_rows,
    load_trainable_checkpoint,
    reservoir_rows_with_sha256,
)


class ProductFitError(RuntimeError):
    """The checkpoint fit-score contract was violated."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise ProductFitError(f"refusing to replace fit report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


@torch.inference_mode()
def _per_example_nll(
    model: ProductReasoningModel,
    tokenizer: Any,
    rows: list[dict[str, str]],
    max_sequence_length: int,
) -> tuple[list[float], list[int]]:
    model.eval()
    losses: list[float] = []
    charged: list[int] = []
    workspace_slots = (
        model.workspace_config.workspace_slots if model.workspace_config else 0
    )
    for row in rows:
        prompts, responses = _tokenize_rows(
            tokenizer,
            [row],
            max_sequence_length,
            workspace_slots,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, metrics = model.forward_batch(
                prompts,
                responses,
                tokenizer.pad_token_id,
            )
        losses.append(float(metrics["language_loss"]))
        charged.append(int(metrics["charged_tokens"]))
    return losses, charged


def _weighted_mean(losses: list[float], charged: list[int]) -> float:
    denominator = sum(charged)
    if denominator <= 0 or len(losses) != len(charged):
        raise ProductFitError("fit-score token accounting differs")
    return sum(loss * tokens for loss, tokens in zip(losses, charged, strict=True)) / denominator


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ProductFitError("checkpoint metadata is missing")
    if metadata.get("model_revision") != args.model_revision:
        raise ProductFitError("checkpoint model revision differs")

    seed = int(metadata["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    backbone = AutoModelForMultimodalLM.from_pretrained(
        args.model_root,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    workspace = metadata.get("workspace_config") or {}
    model = ProductReasoningModel(
        backbone=backbone,
        arm=str(metadata["arm"]),
        lora_layers=int(metadata["lora_layers"]),
        lora_rank=int(metadata["lora_rank"]),
        lora_alpha=float(metadata["lora_alpha"]),
        workspace_width=int(workspace.get("workspace_width", 512)),
        workspace_slots=int(workspace.get("workspace_slots", 16)),
        recurrent_steps=int(workspace.get("recurrent_steps", 8)),
        dense_width=(
            int(workspace.get("workspace_width", 192))
            if metadata["arm"] == "dense"
            else 192
        ),
    ).to("cuda:0")
    if model.trainable_parameter_count() != int(metadata["trainable_parameters"]):
        raise ProductFitError("reconstructed trainable parameter count differs")

    rows, data_sha256 = reservoir_rows_with_sha256(
        args.data,
        int(metadata["selected_rows"]),
        int(metadata["data_seed"]),
    )
    if data_sha256 != metadata["data_sha256"]:
        raise ProductFitError("training data hash differs")

    torch.cuda.reset_peak_memory_stats()
    initial_losses, charged = _per_example_nll(
        model, tokenizer, rows, args.max_sequence_length
    )
    update, restored = load_trainable_checkpoint(args.checkpoint, model)
    if restored != metadata:
        raise ProductFitError("restored checkpoint metadata differs")
    final_losses, final_charged = _per_example_nll(
        model, tokenizer, rows, args.max_sequence_length
    )
    if charged != final_charged:
        raise ProductFitError("before/after charged-token accounting differs")

    initial_mean = _weighted_mean(initial_losses, charged)
    final_mean = _weighted_mean(final_losses, charged)
    improved = sum(
        after < before
        for before, after in zip(initial_losses, final_losses, strict=True)
    )
    return {
        "schema": "shohin-product-reasoning-training-fit-v1",
        "status": "complete",
        "model_revision": args.model_revision,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "arm": metadata["arm"],
        "update": update,
        "data": str(args.data.resolve()),
        "data_sha256": data_sha256,
        "examples": len(rows),
        "charged_tokens": sum(charged),
        "max_sequence_length": args.max_sequence_length,
        "initial_token_weighted_nll": initial_mean,
        "final_token_weighted_nll": final_mean,
        "absolute_nll_change": final_mean - initial_mean,
        "relative_nll_change": (final_mean - initial_mean) / initial_mean,
        "examples_improved": improved,
        "example_initial_nll": initial_losses,
        "example_final_nll": final_losses,
        "example_charged_tokens": charged,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    args = parser.parse_args()
    if args.max_sequence_length <= 32:
        parser.error("max sequence length is too small")
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    _atomic_json(args.output, report)
    print(
        f"[product-fit] arm={report['arm']} initial={report['initial_token_weighted_nll']:.6f} "
        f"final={report['final_token_weighted_nll']:.6f} "
        f"improved={report['examples_improved']}/{report['examples']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
