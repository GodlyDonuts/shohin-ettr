#!/usr/bin/env python3
"""Train the frozen ECR1 treatment or its matched shared residual."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch

from ecr1_moe_revision import ECR1Config, ECR1ProductModel
from hf_product_reasoning_train import (
    PRODUCT_SYSTEM_PROMPT,
    ProductReasoningTrainError,
    _batches,
    _save_checkpoint,
    load_product_backbone,
    render_reasoning_messages,
    reservoir_rows_with_sha256,
)
from ttr1_revision import tokenize_with_draft_mask


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def tokenize_complete_revision_rows(
    tokenizer,
    rows,
    maximum: int,
    *,
    fail_on_overflow: bool = True,
):
    """Tokenize without truncation and prove complete draft/source retention."""

    prompt_rows, response_rows, draft_attention_rows = [], [], []
    source_tokens = draft_tokens = target_tokens = 0
    maximum_observed = 0
    maximum_required = 0
    row_receipts = []
    overflow_receipts = []
    for index, row in enumerate(rows):
        rendered = render_reasoning_messages(
            tokenizer,
            [
                {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
            ],
            enable_thinking=False,
        )
        prompt, draft_attention, _ = tokenize_with_draft_mask(tokenizer, rendered)
        response = tokenizer.encode(row["response"], add_special_tokens=False)
        response.append(tokenizer.eos_token_id)
        total = len(prompt) + len(response)
        maximum_required = max(maximum_required, total)
        if total > maximum:
            overflow_receipts.append(
                {
                    "row_index": index,
                    "identity_sha256": hashlib.sha256(
                        row["question"].encode("utf-8")
                    ).hexdigest(),
                    "source_tokens": sum(draft_attention),
                    "draft_tokens": sum(1 - value for value in draft_attention),
                    "target_tokens": len(response),
                    "total_tokens": total,
                }
            )
            if fail_on_overflow:
                raise ProductReasoningTrainError(
                    f"complete row {index} requires {total} tokens, exceeds {maximum}"
                )
            continue
        draft_count = sum(1 - value for value in draft_attention)
        source_count = sum(draft_attention)
        if draft_count == 0 or source_count == 0:
            raise ProductReasoningTrainError("complete source or draft span is empty")
        prompt_rows.append(prompt)
        response_rows.append(response)
        draft_attention_rows.append(draft_attention)
        source_tokens += source_count
        draft_tokens += draft_count
        target_tokens += len(response)
        maximum_observed = max(maximum_observed, total)
        row_receipts.append(
            {
                "identity_sha256": hashlib.sha256(
                    row["question"].encode("utf-8")
                ).hexdigest(),
                "original_source_tokens": source_count,
                "retained_source_tokens": source_count,
                "original_draft_tokens": draft_count,
                "retained_draft_tokens": draft_count,
                "original_target_tokens": len(response),
                "retained_target_tokens": len(response),
                "total_tokens": total,
            }
        )
    custody = {
        "rows": len(prompt_rows),
        "max_sequence_length": maximum,
        "maximum_observed_tokens": maximum_observed,
        "maximum_required_tokens": maximum_required,
        "overflow_rows": len(overflow_receipts),
        "overflow_receipts": overflow_receipts,
        "original_source_tokens": source_tokens,
        "retained_source_tokens": source_tokens,
        "original_draft_tokens": draft_tokens,
        "retained_draft_tokens": draft_tokens,
        "original_target_tokens": target_tokens,
        "retained_target_tokens": target_tokens,
        "source_retention": 1.0 if not overflow_receipts else None,
        "draft_retention": 1.0 if not overflow_receipts else None,
        "target_retention": 1.0 if not overflow_receipts else None,
        "row_receipts": row_receipts,
    }
    return prompt_rows, response_rows, draft_attention_rows, custody


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    if args.output.exists() or args.quantization != "none":
        raise ProductReasoningTrainError("ECR1 output exists or quantization differs")
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
    config = ECR1Config(
        hidden_size=int(backbone.config.hidden_size),
        num_experts=int(backbone.config.num_experts),
        experts_per_token=int(backbone.config.num_experts_per_tok),
        controlled_layers=args.controlled_layers,
        rank=args.rank,
        alpha=args.alpha,
        mode=args.mode,
    )
    model = ECR1ProductModel(
        backbone,
        config,
        draft_control=args.draft_control,
    ).to("cuda:0")
    trainable_names = sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if not trainable_names or any(".base." in name for name in trainable_names):
        raise ProductReasoningTrainError("ECR1 exposed a protected base parameter")
    expected = 515_840 if args.mode == "expert_conditioned" else 524_288
    if args.controlled_layers == 4 and args.rank in {31, 32}:
        if model.trainable_parameter_count() != expected:
            raise ProductReasoningTrainError("ECR1 parameter receipt differs")
    rows, data_sha256 = reservoir_rows_with_sha256(args.data, args.max_rows, args.data_seed)
    prompt_rows, response_rows, attention_rows, sequence_custody = (
        tokenize_complete_revision_rows(tokenizer, rows, args.max_sequence_length)
    )
    examples = list(zip(prompt_rows, response_rows, attention_rows, strict=True))
    batch_stream = list(_batches(examples, args.batch_size))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    adapter_macs_per_token_per_layer = (
        2 * config.hidden_size * config.rank
        + (config.experts_per_token * config.rank if config.mode == "expert_conditioned" else 0)
    )
    metadata = {
        "architecture": "shohin-ecr1-moe-revision-v1",
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
        "ecr1_config": asdict(config),
        "ecr1_draft_control": args.draft_control,
        "trainable_parameters": model.trainable_parameter_count(),
        "trainable_parameter_name_sha256": model.trainable_parameter_name_sha256(),
        "protected_router_expert_trainables": 0,
        "protected_router_expert_parameters": model.protected_parameter_count(),
        "adapter_macs_per_token_per_layer": adapter_macs_per_token_per_layer,
        "adapter_flops_per_token_per_layer": 2 * adapter_macs_per_token_per_layer,
        "sequence_custody": sequence_custody,
        "lora_layers": 0,
        "lora_rank": 0,
        "lora_alpha": 0.0,
        "lora_scope": "none",
        "unfreeze_layers": 0,
        "workspace_config": None,
        "warm_start_checkpoint": None,
    }
    model.train()
    model.reset_routing_receipt()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    update = microstep = charged = 0
    trace = []
    while update < args.updates:
        raw_batch = batch_stream[microstep % len(batch_stream)]
        prompts = [item[0] for item in raw_batch]
        responses = [item[1] for item in raw_batch]
        attentions = [item[2] for item in raw_batch]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = model.forward_batch(
                prompts, responses, tokenizer.pad_token_id, attentions
            )
            scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()
        charged += int(metrics["charged_tokens"])
        microstep += 1
        if microstep % args.gradient_accumulation:
            continue
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
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
                "gradient_norm": float(gradient_norm),
                "learning_rate": learning_rate,
                "charged_tokens": charged,
                "charged_tokens_per_second": charged / elapsed,
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
        "schema": "shohin-ecr1-product-training-v1",
        "status": "complete",
        **metadata,
        "updates": update,
        "gradient_accumulation": args.gradient_accumulation,
        "batch_size": args.batch_size,
        "max_sequence_length": args.max_sequence_length,
        "learning_rate": args.learning_rate,
        "charged_tokens": charged,
        "elapsed_seconds": elapsed,
        "charged_tokens_per_second": charged / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "routing_receipt": model.routing_receipt(),
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
    parser.add_argument("--mode", choices=("expert_conditioned", "shared"), required=True)
    parser.add_argument("--draft-control", choices=("normal", "draft_unavailable"), default="normal")
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=9655)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--controlled-layers", type=int, default=4)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
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
        args.rank,
        args.log_interval,
        args.checkpoint_interval,
    )
    if any(value <= 0 for value in positive) or args.alpha <= 0 or args.learning_rate <= 0:
        parser.error("ECR1 training dimensions differ")
    if args.mode == "expert_conditioned" and (args.rank, args.alpha) != (31, 31):
        parser.error("final-four ECR1 geometry differs")
    if args.mode == "shared" and (args.rank, args.alpha) != (32, 32):
        parser.error("final-four shared geometry differs")
    return args


def main() -> int:
    report = run(parse_args())
    print(
        f"[ecr1-train] updates={report['updates']} tokens/s={report['charged_tokens_per_second']:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
