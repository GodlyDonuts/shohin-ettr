"""Train DIVERGE-QPT1 on a pinned product-reasoning backbone."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch

from diverge_qpt1_product import (
    QPT1ProductModel,
    frozen_parameter_sha256,
    qpt1_architecture_sha256,
)
from hf_product_reasoning_train import (
    ProductReasoningTrainError,
    _atomic_json,
    _batches,
    _save_checkpoint,
    _tokenize_rows,
    load_product_backbone,
    reservoir_rows_with_sha256,
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.output.exists():
        raise ProductReasoningTrainError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    backbone, resolved_model_loader = load_product_backbone(
        args.model_root,
        args.model_loader,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = QPT1ProductModel(
        backbone,
        lora_layers=args.lora_layers,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        workspace_width=args.workspace_width,
        source_slots=args.source_slots,
        query_slots=args.query_slots,
        recurrent_steps=args.recurrent_steps,
        attention_heads=args.attention_heads,
        ff_multiplier=args.ff_multiplier,
        pointer_temperature=args.pointer_temperature,
        binding_weight=args.binding_weight,
        coverage_weight=args.coverage_weight,
        reset_weight=args.reset_weight,
    ).to("cuda:0")

    rows, data_hash = reservoir_rows_with_sha256(
        args.data, args.max_rows, args.data_seed
    )
    batch_stream = list(_batches(rows, args.batch_size))
    if not batch_stream:
        raise ProductReasoningTrainError("training population is smaller than a batch")

    metadata = {
        "architecture": model.architecture,
        "arm": model.arm,
        "model_root": str((args.model_source_root or args.model_root).resolve()),
        "loaded_model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": resolved_model_loader,
        "backbone_layout": model.backbone_layout,
        "data": str(args.data.resolve()),
        "data_sha256": data_hash,
        "selected_rows": len(rows),
        "seed": args.seed,
        "data_seed": args.data_seed,
        "lora_layers": args.lora_layers,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_projection_count": model.lora_projection_count,
        "unfreeze_layers": 0,
        "trainable_parameters": model.trainable_parameter_count(),
        "workspace_config": asdict(model.workspace_config),
        "workspace_architecture_sha256": qpt1_architecture_sha256(
            model.workspace_config
        ),
        "binding_weight": args.binding_weight,
        "coverage_weight": args.coverage_weight,
        "reset_weight": args.reset_weight,
        "inference_control": "normal",
    }

    frozen_before = frozen_parameter_sha256(model)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    optimizer.zero_grad(set_to_none=True)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    update = 0
    microstep = 0
    logical_tokens = 0
    trace: list[dict[str, float | int]] = []

    while update < args.updates:
        raw_batch = batch_stream[microstep % len(batch_stream)]
        prompt_rows, response_rows = _tokenize_rows(
            tokenizer,
            raw_batch,
            args.max_sequence_length,
            model.sequence_workspace_slots(),
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = model.forward_batch(
                prompt_rows,
                response_rows,
                tokenizer.pad_token_id,
            )
            scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()
        logical_tokens += int(metrics["logical_charged_tokens"])
        microstep += 1
        if microstep % args.gradient_accumulation:
            continue

        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        progress = update / max(args.updates - 1, 1)
        learning_rate = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1

        if update == 1 or update % args.log_interval == 0:
            elapsed = time.monotonic() - started
            event = {
                "update": update,
                "loss": float(loss.detach()),
                "language_loss": metrics["language_loss"],
                "binding_loss": metrics["binding_loss"],
                "coverage_loss": metrics["coverage_loss"],
                "reset_loss": metrics["reset_loss"],
                "mean_step_delta": metrics["mean_step_delta"],
                "mean_commit_gate": metrics["mean_commit_gate"],
                "mean_release_gate": metrics["mean_release_gate"],
                "gradient_norm": float(gradient_norm),
                "learning_rate": learning_rate,
                "logical_charged_tokens": logical_tokens,
                "logical_tokens_per_second": logical_tokens / elapsed,
            }
            trace.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)

        if update % args.checkpoint_interval == 0 or update == args.updates:
            _save_checkpoint(
                args.output / f"checkpoint_{update:07d}.pt",
                model,
                optimizer,
                update,
                metadata,
            )

    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    frozen_after = frozen_parameter_sha256(model)
    report = {
        "schema": "shohin-diverge-qpt1-training-v1",
        "status": "complete",
        **metadata,
        "updates": update,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "max_sequence_length": args.max_sequence_length,
        "learning_rate": args.learning_rate,
        "logical_charged_tokens": logical_tokens,
        "elapsed_seconds": elapsed,
        "logical_tokens_per_second": logical_tokens / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "frozen_parameter_sha256_before": frozen_before,
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameters_unchanged": frozen_before == frozen_after,
        "trace": trace,
    }
    _atomic_json(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--model-loader", choices=("auto", "causal", "multimodal"), default="auto"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-layers", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--workspace-width", type=int, default=512)
    parser.add_argument("--source-slots", type=int, default=8)
    parser.add_argument("--query-slots", type=int, default=4)
    parser.add_argument("--recurrent-steps", type=int, default=8)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--ff-multiplier", type=int, default=2)
    parser.add_argument("--pointer-temperature", type=float, default=0.50)
    parser.add_argument("--binding-weight", type=float, default=0.05)
    parser.add_argument("--coverage-weight", type=float, default=0.02)
    parser.add_argument("--reset-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2026080702)
    parser.add_argument("--data-seed", type=int, default=20260802)
    parser.add_argument("--log-interval", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int, default=64)
    args = parser.parse_args()
    positive = (
        args.updates,
        args.batch_size,
        args.gradient_accumulation,
        args.max_rows,
        args.max_sequence_length,
        args.lora_layers,
        args.lora_rank,
        args.workspace_width,
        args.source_slots,
        args.query_slots,
        args.recurrent_steps,
        args.attention_heads,
        args.ff_multiplier,
        args.log_interval,
        args.checkpoint_interval,
    )
    if any(value <= 0 for value in positive) or args.learning_rate <= 0:
        parser.error("QPT1 dimensions and learning rate must be positive")
    if min(args.binding_weight, args.coverage_weight, args.reset_weight) < 0:
        parser.error("QPT1 loss weights must be nonnegative")
    if args.pointer_temperature <= 0:
        parser.error("QPT1 pointer temperature must be positive")
    return args


def main() -> int:
    report = run(parse_args())
    print(
        "[qpt1-train] "
        f"updates={report['updates']} "
        f"tokens/s={report['logical_tokens_per_second']:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
