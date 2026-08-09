#!/usr/bin/env python3
"""Train draft-conditioned recurrent expert modulation on revision traces."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time

import torch

from drem1_moe_revision import DREM1Config, DREM1ProductModel
from hf_product_reasoning_train import (
    ProductReasoningTrainError,
    _batches,
    _save_checkpoint,
    _tokenize_rows_with_syndrome,
    load_product_backbone,
    reservoir_rows_with_sha256,
)


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    if args.output.exists() or args.quantization != "none":
        raise ProductReasoningTrainError("DREM1 output exists or quantization differs")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    backbone, resolved_loader = load_product_backbone(
        args.model_root,
        args.model_loader,
        dtype=torch.bfloat16,
        device_map={"": 0},
        quantization=args.quantization,
    )
    config = DREM1Config(
        hidden_size=int(backbone.config.hidden_size),
        num_experts=int(backbone.config.num_experts),
        experts_per_token=int(backbone.config.num_experts_per_tok),
        controlled_layers=args.controlled_layers,
        controller_width=args.controller_width,
        adapter_rank=args.adapter_rank,
        recurrent_steps=args.recurrent_steps,
        router_scale=args.router_scale,
        entropy_floor=args.entropy_floor,
    )
    model = DREM1ProductModel(
        backbone,
        config,
        mode=args.mode,
        collapse_weight=args.collapse_weight,
        context_control=args.context_control,
    ).to("cuda:0")
    trainable_names = sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if not trainable_names or any(".base." in name for name in trainable_names):
        raise ProductReasoningTrainError("DREM1 exposed a frozen base parameter")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    rows, data_sha256 = reservoir_rows_with_sha256(
        args.data, args.max_rows, args.data_seed
    )
    batch_stream = list(_batches(rows, args.batch_size))
    if not batch_stream:
        raise ProductReasoningTrainError("DREM1 training population is too small")
    metadata = {
        "architecture": "shohin-drem1-moe-revision-v1",
        "arm": "baseline",
        "model_root": str((args.model_source_root or args.model_root).resolve()),
        "loaded_model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": resolved_loader,
        "backbone_layout": model.backbone_layout,
        "quantization": args.quantization,
        "data": str(args.data.resolve()),
        "data_sha256": data_sha256,
        "selected_rows": len(rows),
        "seed": args.seed,
        "data_seed": args.data_seed,
        "drem1_config": asdict(config),
        "drem1_mode": args.mode,
        "drem1_context_control": args.context_control,
        "collapse_weight": args.collapse_weight,
        "trainable_parameters": model.trainable_parameter_count(),
        "trainable_parameter_name_sha256": model.trainable_parameter_name_sha256(),
        "protected_router_expert_trainables": 0,
        "protected_router_expert_parameters": model.protected_parameter_count(),
        "lora_layers": 0,
        "lora_rank": 0,
        "lora_alpha": 0.0,
        "lora_scope": "none",
        "unfreeze_layers": 0,
        "workspace_config": None,
        "warm_start_checkpoint": None,
    }
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    total_charged = 0
    update = 0
    microstep = 0
    trace = []
    while update < args.updates:
        raw_batch = batch_stream[microstep % len(batch_stream)]
        prompt_rows, response_rows, indicators = _tokenize_rows_with_syndrome(
            tokenizer, raw_batch, args.max_sequence_length, 0
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = model.forward_batch(
                prompt_rows, response_rows, tokenizer.pad_token_id, indicators
            )
            scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()
        total_charged += int(metrics["charged_tokens"])
        microstep += 1
        if microstep % args.gradient_accumulation:
            continue
        trainable = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        progress = update / max(args.updates - 1, 1)
        learning_rate = args.learning_rate * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
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
                "gradient_norm": float(gradient_norm),
                "learning_rate": learning_rate,
                "charged_tokens": total_charged,
                "charged_tokens_per_second": total_charged / elapsed,
                **metrics,
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
    report = {
        "schema": "shohin-drem1-product-training-v1",
        "status": "complete",
        **metadata,
        "updates": update,
        "gradient_accumulation": args.gradient_accumulation,
        "batch_size": args.batch_size,
        "max_sequence_length": args.max_sequence_length,
        "learning_rate": args.learning_rate,
        "charged_tokens": total_charged,
        "elapsed_seconds": elapsed,
        "charged_tokens_per_second": total_charged / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "trace": trace,
    }
    _atomic_json(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", choices=("auto", "causal"), default="causal")
    parser.add_argument("--quantization", choices=("none",), default="none")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "router_only", "expert_only"), required=True)
    parser.add_argument(
        "--context-control", choices=("normal", "draft_masked"), default="normal"
    )
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--controlled-layers", type=int, default=4)
    parser.add_argument("--controller-width", type=int, default=256)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--recurrent-steps", type=int, default=4)
    parser.add_argument("--router-scale", type=float, default=1.0)
    parser.add_argument("--entropy-floor", type=float, default=0.80)
    parser.add_argument("--collapse-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026080901)
    parser.add_argument("--data-seed", type=int, default=2026080814)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=256)
    args = parser.parse_args()
    positive = (
        args.updates,
        args.batch_size,
        args.gradient_accumulation,
        args.max_rows,
        args.max_sequence_length,
        args.controlled_layers,
        args.controller_width,
        args.adapter_rank,
        args.recurrent_steps,
        args.log_interval,
        args.checkpoint_interval,
    )
    if any(value <= 0 for value in positive) or args.learning_rate <= 0:
        parser.error("DREM1 training dimensions differ")
    return args


def main() -> int:
    report = run(parse_args())
    print(
        f"[drem1-train] updates={report['updates']} "
        f"tokens/s={report['charged_tokens_per_second']:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
