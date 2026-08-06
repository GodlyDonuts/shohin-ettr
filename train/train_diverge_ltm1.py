"""Frozen real-language training gate for DIVERGE-LTM1."""

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

from diverge_ltm1_product import LTM1ProductModel, frozen_parameter_sha256
from diverge_ltm1_workspace import latent_trajectory_architecture_sha256
from hf_product_reasoning_train import (
    ProductReasoningTrainError,
    _atomic_json,
    _save_checkpoint,
    _tokenize_rows,
    load_product_backbone,
    reservoir_rows_with_sha256,
)


def _audit_fit(
    model: LTM1ProductModel,
    tokenizer: Any,
    rows: list[dict[str, str]],
    *,
    max_sequence_length: int,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    nlls: list[float] = []
    trace_cosines: list[float] = []
    prior_indices: list[int] = []
    losses: list[float] = []
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            prompt_rows, response_rows = _tokenize_rows(
                tokenizer,
                batch,
                max_sequence_length,
                model.sequence_workspace_slots(),
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = model.forward_batch(
                    prompt_rows,
                    response_rows,
                    tokenizer.pad_token_id,
                )
            losses.append(float(loss.detach()))
            nlls.extend(float(value) for value in metrics["prior_selected_nll_rows"])
            trace_cosines.extend(
                float(value) for value in metrics["trace_cosine_rows"]
            )
            prior_indices.extend(int(value) for value in metrics["prior_indices"])
    model.train()
    return {
        "loss": sum(losses) / len(losses),
        "prior_selected_nll_rows": nlls,
        "mean_prior_selected_nll": sum(nlls) / len(nlls),
        "trace_cosine_rows": trace_cosines,
        "mean_trace_cosine": sum(trace_cosines) / len(trace_cosines),
        "prior_indices": prior_indices,
        "selected_trajectory_ids": sorted(set(prior_indices)),
    }


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
    model = LTM1ProductModel(
        backbone,
        lora_layers=args.lora_layers,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        latent_width=args.latent_width,
        trajectory_slots=args.trajectory_slots,
        recurrent_steps=args.recurrent_steps,
        fault_bits=args.fault_bits,
        attention_heads=args.attention_heads,
        ff_multiplier=args.ff_multiplier,
        trace_weight=args.trace_weight,
        balance_weight=args.balance_weight,
        halting_weight=args.halting_weight,
    ).to("cuda:0")
    rows, data_hash = reservoir_rows_with_sha256(
        args.data, args.max_rows, args.data_seed
    )
    if len(rows) < args.batch_size:
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
        "workspace_architecture_sha256": latent_trajectory_architecture_sha256(
            model.workspace_config
        ),
        "trace_weight": args.trace_weight,
        "balance_weight": args.balance_weight,
        "halting_weight": args.halting_weight,
        "selection_strategy": "highest_prior",
    }

    frozen_before = frozen_parameter_sha256(model)
    fit_before = None
    if len(rows) <= 16:
        fit_before = _audit_fit(
            model,
            tokenizer,
            rows,
            max_sequence_length=args.max_sequence_length,
            batch_size=args.batch_size,
        )

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
    candidate_tokens = 0
    trace: list[dict[str, float | int]] = []

    while update < args.updates:
        offset = (microstep * args.batch_size) % len(rows)
        raw_batch = [rows[(offset + index) % len(rows)] for index in range(args.batch_size)]
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
        candidate_tokens += int(metrics["candidate_charged_tokens"])
        microstep += 1
        if microstep % args.gradient_accumulation:
            continue

        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
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
                "marginal_energy": metrics["marginal_energy"],
                "mean_candidate_nll": metrics["mean_candidate_nll"],
                "best_candidate_nll": metrics["best_candidate_nll"],
                "prior_selected_nll": metrics["prior_selected_nll"],
                "trace_cosine": metrics["trace_cosine"],
                "posterior_entropy": metrics["posterior_entropy"],
                "candidate_similarity": metrics["candidate_similarity"],
                "balance_loss": metrics["balance_loss"],
                "halting_loss": metrics["halting_loss"],
                "final_stop_probability": metrics["final_stop_probability"],
                "mean_step_delta": metrics["mean_step_delta"],
                "gradient_norm": float(gradient_norm),
                "learning_rate": learning_rate,
                "logical_charged_tokens": logical_tokens,
                "candidate_charged_tokens": candidate_tokens,
                "logical_tokens_per_second": logical_tokens / elapsed,
                "candidate_tokens_per_second": candidate_tokens / elapsed,
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
    fit_after = None
    if len(rows) <= 16:
        fit_after = _audit_fit(
            model,
            tokenizer,
            rows,
            max_sequence_length=args.max_sequence_length,
            batch_size=args.batch_size,
        )
    frozen_after = frozen_parameter_sha256(model)
    frozen_unchanged = frozen_before == frozen_after
    fit_internal_gate = None
    if fit_before is not None and fit_after is not None:
        improved = [
            after < before
            for before, after in zip(
                fit_before["prior_selected_nll_rows"],
                fit_after["prior_selected_nll_rows"],
                strict=True,
            )
        ]
        fit_internal_gate = {
            "improved_rows": sum(improved),
            "total_rows": len(improved),
            "all_rows_improved": all(improved),
            "trace_cosine_at_least_0_90": fit_after["mean_trace_cosine"] >= 0.90,
            "at_least_two_trajectory_ids": (
                len(fit_after["selected_trajectory_ids"]) >= 2
            ),
            "frozen_parameters_unchanged": frozen_unchanged,
        }
        fit_internal_gate["qualified_except_matched_b1"] = all(
            bool(value)
            for key, value in fit_internal_gate.items()
            if key not in {"improved_rows", "total_rows"}
        )

    report = {
        "schema": "shohin-diverge-ltm1-training-v1",
        "status": "complete",
        **metadata,
        "updates": update,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "max_sequence_length": args.max_sequence_length,
        "learning_rate": args.learning_rate,
        "logical_charged_tokens": logical_tokens,
        "candidate_charged_tokens": candidate_tokens,
        "elapsed_seconds": elapsed,
        "logical_tokens_per_second": logical_tokens / elapsed,
        "candidate_tokens_per_second": candidate_tokens / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "frozen_parameter_sha256_before": frozen_before,
        "frozen_parameter_sha256_after": frozen_after,
        "frozen_parameters_unchanged": frozen_unchanged,
        "fit_before": fit_before,
        "fit_after": fit_after,
        "fit_internal_gate": fit_internal_gate,
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
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=16)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-layers", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--latent-width", type=int, default=384)
    parser.add_argument("--trajectory-slots", type=int, default=8)
    parser.add_argument("--recurrent-steps", type=int, default=8)
    parser.add_argument("--fault-bits", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--ff-multiplier", type=int, default=2)
    parser.add_argument("--trace-weight", type=float, default=0.25)
    parser.add_argument("--balance-weight", type=float, default=0.01)
    parser.add_argument("--halting-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026080601)
    parser.add_argument("--data-seed", type=int, default=20260802)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    args = parser.parse_args()
    positive = (
        args.updates,
        args.batch_size,
        args.gradient_accumulation,
        args.max_rows,
        args.max_sequence_length,
        args.lora_layers,
        args.lora_rank,
        args.latent_width,
        args.trajectory_slots,
        args.recurrent_steps,
        args.fault_bits,
        args.attention_heads,
        args.ff_multiplier,
        args.log_interval,
        args.checkpoint_interval,
    )
    if any(value <= 0 for value in positive) or args.learning_rate <= 0:
        parser.error("LTM1 dimensions and learning rate must be positive")
    if min(args.trace_weight, args.balance_weight, args.halting_weight) < 0:
        parser.error("LTM1 loss weights must be nonnegative")
    if args.max_sequence_length <= args.trajectory_slots + 16:
        parser.error("maximum sequence length leaves no prompt/target budget")
    return args


def main() -> int:
    report = run(parse_args())
    print(
        "[ltm1-train] "
        f"updates={report['updates']} "
        f"logical_tok/s={report['logical_tokens_per_second']:.1f} "
        f"frozen={report['frozen_parameters_unchanged']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
