"""Train DIVERGE-SAG1 from a qualified frozen B1 adapter."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch

from diverge_sag1_product import (
    SAG1ProductModel,
    frozen_parameter_sha256,
    sag1_architecture_sha256,
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


RESUME_MATCH_FIELDS = (
    "architecture",
    "arm",
    "model_root",
    "model_revision",
    "data_sha256",
    "selected_rows",
    "seed",
    "data_seed",
    "lora_layers",
    "lora_rank",
    "lora_alpha",
    "workspace_config",
    "workspace_architecture_sha256",
    "base_checkpoint_sha256",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sag1_training_checkpoint(
    path: Path,
    model: SAG1ProductModel,
    optimizer: torch.optim.Optimizer,
    expected_metadata: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Restore a SAG1 training checkpoint and fail closed on lineage drift."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "shohin-hf-product-reasoning-checkpoint-v1":
        raise ProductReasoningTrainError("SAG1 resume checkpoint schema differs")
    saved = payload.get("trainable_state")
    metadata = payload.get("metadata")
    optimizer_state = payload.get("optimizer")
    if not isinstance(saved, dict) or not isinstance(metadata, dict):
        raise ProductReasoningTrainError("SAG1 resume checkpoint is incomplete")
    if not isinstance(optimizer_state, dict):
        raise ProductReasoningTrainError("SAG1 resume optimizer state is missing")
    mismatches = {
        field: {"expected": expected_metadata.get(field), "actual": metadata.get(field)}
        for field in RESUME_MATCH_FIELDS
        if metadata.get(field) != expected_metadata.get(field)
    }
    if mismatches:
        raise ProductReasoningTrainError(
            f"SAG1 resume metadata differs: {json.dumps(mismatches, sort_keys=True)}"
        )
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(saved) != set(current):
        raise ProductReasoningTrainError("SAG1 resume parameter contract differs")
    with torch.no_grad():
        for name, parameter in current.items():
            tensor = saved[name]
            if tensor.shape != parameter.shape:
                raise ProductReasoningTrainError(
                    f"SAG1 resume tensor shape differs: {name}"
                )
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    optimizer.load_state_dict(optimizer_state)
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(next(iter(current.values())).device)
    update = payload.get("update")
    if not isinstance(update, int) or update <= 0:
        raise ProductReasoningTrainError("SAG1 resume update is invalid")
    return update, metadata


def _charged_tokens_before(
    *,
    tokenizer: Any,
    batch_stream: list[list[dict[str, str]]],
    microsteps: int,
    max_sequence_length: int,
    workspace_slots: int,
) -> int:
    total = 0
    for microstep in range(microsteps):
        _, response_rows = _tokenize_rows(
            tokenizer,
            batch_stream[microstep % len(batch_stream)],
            max_sequence_length,
            workspace_slots,
        )
        total += sum(len(row) for row in response_rows)
    return total


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.resume_checkpoint is None:
        if args.output.exists():
            raise ProductReasoningTrainError(f"output already exists: {args.output}")
        args.output.mkdir(parents=True)
    else:
        if not args.output.is_dir() or not args.resume_checkpoint.is_file():
            raise ProductReasoningTrainError("SAG1 resume output/checkpoint is missing")
        if args.resume_checkpoint.resolve().parent != args.output.resolve():
            raise ProductReasoningTrainError("SAG1 resume checkpoint is outside output")
        if (args.output / "report.json").exists():
            raise ProductReasoningTrainError("SAG1 output is already complete")
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
    model = SAG1ProductModel(
        backbone,
        base_checkpoint=args.base_checkpoint,
        base_checkpoint_sha256=args.base_checkpoint_sha256,
        model_revision=args.model_revision,
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
        router_hidden=args.router_hidden,
        advantage_margin=args.advantage_margin,
        router_threshold=args.router_threshold,
        router_weight=args.router_weight,
        risk_weight=args.risk_weight,
        sparsity_weight=args.sparsity_weight,
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
        "workspace_architecture_sha256": sag1_architecture_sha256(
            model,
            advantage_margin=args.advantage_margin,
            router_threshold=args.router_threshold,
        ),
        "binding_weight": args.binding_weight,
        "coverage_weight": args.coverage_weight,
        "reset_weight": args.reset_weight,
        "router_hidden": args.router_hidden,
        "advantage_margin": args.advantage_margin,
        "router_threshold": args.router_threshold,
        "router_weight": args.router_weight,
        "risk_weight": args.risk_weight,
        "sparsity_weight": args.sparsity_weight,
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_sha256": args.base_checkpoint_sha256,
        "base_checkpoint_metadata": model.base_metadata,
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
    resume_update = 0
    resume_sha256 = None
    if args.resume_checkpoint is not None:
        resume_update, _ = load_sag1_training_checkpoint(
            args.resume_checkpoint,
            model,
            optimizer,
            metadata,
        )
        if resume_update >= args.updates:
            raise ProductReasoningTrainError("SAG1 resume is already at target update")
        resume_sha256 = _sha256_file(args.resume_checkpoint)
        metadata["resume_checkpoint"] = str(args.resume_checkpoint.resolve())
        metadata["resume_checkpoint_sha256"] = resume_sha256
        metadata["resume_update"] = resume_update
    optimizer.zero_grad(set_to_none=True)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    update = resume_update
    microstep = resume_update * args.gradient_accumulation
    logical_tokens = _charged_tokens_before(
        tokenizer=tokenizer,
        batch_stream=batch_stream,
        microsteps=microstep,
        max_sequence_length=args.max_sequence_length,
        workspace_slots=model.sequence_workspace_slots(),
    )
    segment_logical_tokens = 0
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
        segment_logical_tokens += int(metrics["logical_charged_tokens"])
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
                "gradient_norm": float(gradient_norm),
                "learning_rate": learning_rate,
                **metrics,
                "logical_charged_tokens": logical_tokens,
                "logical_tokens_per_second": segment_logical_tokens / elapsed,
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
        "schema": "shohin-diverge-sag1-training-v1",
        "status": "complete",
        **metadata,
        "updates": update,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "max_sequence_length": args.max_sequence_length,
        "learning_rate": args.learning_rate,
        "logical_charged_tokens": logical_tokens,
        "segment_logical_charged_tokens": segment_logical_tokens,
        "elapsed_seconds": elapsed,
        "logical_tokens_per_second": segment_logical_tokens / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "frozen_parameter_sha256_before": frozen_before,
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameters_unchanged": frozen_before == frozen_after,
        "trace": trace,
        "resume_checkpoint_sha256": resume_sha256,
        "resume_update": resume_update or None,
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
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint-sha256", required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
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
    parser.add_argument("--pointer-temperature", type=float, default=0.5)
    parser.add_argument("--binding-weight", type=float, default=0.05)
    parser.add_argument("--coverage-weight", type=float, default=0.02)
    parser.add_argument("--reset-weight", type=float, default=0.02)
    parser.add_argument("--router-hidden", type=int, default=256)
    parser.add_argument("--advantage-margin", type=float, default=0.02)
    parser.add_argument("--router-threshold", type=float, default=0.5)
    parser.add_argument("--router-weight", type=float, default=0.2)
    parser.add_argument("--risk-weight", type=float, default=0.5)
    parser.add_argument("--sparsity-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026080711)
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
        args.router_hidden,
        args.log_interval,
        args.checkpoint_interval,
    )
    if any(value <= 0 for value in positive) or args.learning_rate <= 0:
        parser.error("SAG1 dimensions and learning rate must be positive")
    if min(
        args.binding_weight,
        args.coverage_weight,
        args.reset_weight,
        args.advantage_margin,
        args.router_weight,
        args.risk_weight,
        args.sparsity_weight,
    ) < 0:
        parser.error("SAG1 margins and loss weights must be nonnegative")
    if not 0.0 < args.router_threshold < 1.0:
        parser.error("SAG1 router threshold must be in (0, 1)")
    return args


def main() -> int:
    report = run(parse_args())
    print(
        "[sag1-train] "
        f"updates={report['updates']} "
        f"tokens/s={report['logical_tokens_per_second']:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
