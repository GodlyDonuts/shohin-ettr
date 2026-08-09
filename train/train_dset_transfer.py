#!/usr/bin/env python3
"""Train DSET on a modern host through generic post-MLP residuals."""

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

from dset1_edit_transducer import normalized_script_loss
from hf_product_reasoning_train import (
    _save_checkpoint,
    load_product_backbone,
    pack_training_embeddings,
)
from shared_post_mlp_revision import SharedPostMLPConfig, SharedPostMLPProductModel
from train_dset1_span_edit import atomic_json, load_pairs, sha256_file, tokenize_pairs


SCHEMA = "shohin-dset-modern-transfer-training-v1"


class DSETTransferError(RuntimeError):
    """The frozen DSET transfer contract differs."""


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    if args.output.exists() or args.arm not in {"aligned", "hidden"}:
        raise DSETTransferError("DSET transfer output exists or arm differs")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    pairs, data_report = load_pairs(args.data, args.data_report)
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenized, token_receipt = tokenize_pairs(
        tokenizer, pairs, args.arm, args.max_sequence_length
    )
    random.Random(args.data_seed).shuffle(tokenized)

    backbone, loader = load_product_backbone(
        args.model_root,
        "auto",
        dtype=torch.bfloat16,
        device_map={"": 0},
        quantization=args.quantization,
    )
    text_config = getattr(backbone.config, "text_config", backbone.config)
    config = SharedPostMLPConfig(
        hidden_size=int(text_config.hidden_size),
        controlled_layers=args.controlled_layers,
        rank=args.rank,
        alpha=float(args.rank),
    )
    control = "draft_unavailable" if args.arm == "hidden" else "normal"
    model = SharedPostMLPProductModel(backbone, config, draft_control=control).to("cuda:0")
    expected_trainables = 2 * config.hidden_size * config.rank * config.controlled_layers
    if model.trainable_parameter_count() != expected_trainables:
        raise DSETTransferError("DSET transfer trainable count differs")
    protected_trainables = sum(
        p.numel()
        for name, p in model.named_parameters()
        if p.requires_grad and any(token in name.lower() for token in ("router", "expert"))
    )
    if protected_trainables:
        raise DSETTransferError("DSET transfer exposes protected trainables")
    if hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    metadata = {
        "architecture": "shohin-shared-post-mlp-revision-v1",
        "model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": loader,
        "quantization": args.quantization,
        "backbone_layout": model.backbone_layout,
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "data_report_sha256": sha256_file(args.data_report),
        "seed": args.seed,
        "data_seed": args.data_seed,
        "shared_post_mlp_config": asdict(config),
        "draft_control": control,
        "dset1_arm": args.arm,
        "trainable_parameters": model.trainable_parameter_count(),
        "trainable_parameter_name_sha256": model.trainable_parameter_name_sha256(),
        "protected_router_expert_trainables": protected_trainables,
        "adapter_flops_per_token_per_layer": 4 * config.hidden_size * config.rank,
        "dset1_pairs": len(pairs),
        "dset1_pairs_per_update": args.gradient_accumulation,
        "dset1_script_loss": "mean_per_presentation_then_mean_pair",
        "dset1_data_report": data_report,
        "sequence_custody": token_receipt,
    }
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    update = microstep = charged = script_tokens = 0
    trace = []
    while update < args.updates:
        pair = tokenized[microstep % len(tokenized)]
        pair_loss = 0.0
        for prompt, script, draft_mask in pair:
            masks = [draft_mask] if control == "draft_unavailable" else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                inputs, attention, labels, batch_charged = pack_training_embeddings(
                    model.text_model.embed_tokens,
                    [prompt],
                    [script],
                    None,
                    tokenizer.pad_token_id,
                    prompt_attention_rows=masks,
                )
                outputs = model.text_model(
                    inputs_embeds=inputs, attention_mask=attention, use_cache=False
                )
                logits = model.lm_head(outputs.last_hidden_state)
                loss = normalized_script_loss(logits, labels)
            (loss / (2 * args.gradient_accumulation)).backward()
            pair_loss += float(loss.detach()) / 2
            charged += int(batch_charged)
            script_tokens += len(script)
            del inputs, attention, labels, outputs, logits, loss
        microstep += 1
        if microstep % args.gradient_accumulation:
            continue
        trainable = [p for p in model.parameters() if p.requires_grad]
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        progress = update / max(args.updates - 1, 1)
        lr = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        if update == 1 or update % args.log_interval == 0:
            elapsed = time.monotonic() - started
            event = {
                "update": update,
                "script_ce": pair_loss,
                "gradient_norm": float(gradient_norm),
                "learning_rate": lr,
                "charged_tokens": charged,
                "charged_tokens_per_second": charged / elapsed,
            }
            trace.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)
    _save_checkpoint(args.output / f"checkpoint_{update:07d}.pt", model, optimizer, update, metadata)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    report = {
        "schema": SCHEMA,
        "status": "complete",
        **metadata,
        "updates": update,
        "gradient_accumulation_pairs": args.gradient_accumulation,
        "presentations_per_pair": 2,
        "max_sequence_length": args.max_sequence_length,
        "learning_rate": args.learning_rate,
        "charged_tokens": charged,
        "charged_script_tokens": script_tokens,
        "elapsed_seconds": elapsed,
        "charged_tokens_per_second": charged / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "residual_receipt": model.routing_receipt(),
        "trace": trace,
    }
    atomic_json(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=["aligned", "hidden"], required=True)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--controlled-layers", type=int, default=16)
    parser.add_argument("--rank", type=int, default=18)
    parser.add_argument("--quantization", choices=["none", "nf4"], default="nf4")
    parser.add_argument("--seed", type=int, default=2026080920)
    parser.add_argument("--data-seed", type=int, default=2026080920)
    parser.add_argument("--log-interval", type=int, default=8)
    args = parser.parse_args()
    if min(args.updates, args.gradient_accumulation, args.controlled_layers, args.rank) <= 0:
        parser.error("DSET transfer dimensions differ")
    return args


def main() -> int:
    report = run(parse_args())
    print(json.dumps({"arm": report["dset1_arm"], "updates": report["updates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
