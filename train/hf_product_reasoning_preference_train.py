"""Memory-bounded verifier preference training for the product reasoner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn.functional as F

from hf_product_reasoning_train import (
    PRODUCT_SYSTEM_PROMPT,
    ProductReasoningModel,
    _atomic_json,
    _save_checkpoint,
    _sha256_file,
    load_product_backbone,
    load_trainable_checkpoint,
    render_reasoning_messages,
    validate_warm_start_metadata,
)


SCHEMA = "shohin-hf-product-reasoning-preference-training-v1"
PAIR_SCHEMA = "shohin-product-verifier-preference-pairs-v1"


class ProductPreferenceTrainError(RuntimeError):
    """Preference training cannot satisfy its model or data contract."""


def preference_gradient_coefficient(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    *,
    beta: float,
    margin: float,
) -> torch.Tensor:
    """Return d softplus(margin - beta * gap) / d(-chosen_logp)."""

    return beta * torch.sigmoid(
        margin - beta * (chosen_logp - rejected_logp)
    )


def preference_loss_value(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    *,
    beta: float,
    margin: float,
) -> torch.Tensor:
    return F.softplus(margin - beta * (chosen_logp - rejected_logp))


def _reservoir_pairs(
    path: Path,
    limit: int,
    seed: int,
) -> tuple[list[dict[str, str]], str]:
    if limit <= 0:
        raise ProductPreferenceTrainError("pair limit must be positive")
    generator = random.Random(seed)
    digest = hashlib.sha256()
    selected: list[dict[str, str]] = []
    valid = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            try:
                row = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ProductPreferenceTrainError("preference JSONL is malformed") from exc
            if row.get("schema") != PAIR_SCHEMA:
                raise ProductPreferenceTrainError("preference row schema differs")
            question = str(row.get("question") or "").strip()
            chosen = str(row.get("chosen") or "").strip()
            rejected = str(row.get("rejected") or "").strip()
            if not question or not chosen or not rejected or chosen == rejected:
                raise ProductPreferenceTrainError("preference row is invalid")
            normalized = {
                "question": question,
                "chosen": chosen,
                "rejected": rejected,
            }
            valid += 1
            if len(selected) < limit:
                selected.append(normalized)
            else:
                position = generator.randrange(valid)
                if position < limit:
                    selected[position] = normalized
    if not selected:
        raise ProductPreferenceTrainError("preference source has no valid pairs")
    generator.shuffle(selected)
    return selected, digest.hexdigest()


def _tokenize_response(
    tokenizer: Any,
    question: str,
    response: str,
    max_sequence_length: int,
) -> tuple[list[int], list[int]]:
    rendered = render_reasoning_messages(
        tokenizer,
        [
            {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        enable_thinking=False,
    )
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    if len(response_ids) > max_sequence_length - 9:
        response_ids = response_ids[: max_sequence_length - 9]
    prompt_budget = max_sequence_length - len(response_ids) - 1
    if prompt_budget < 8:
        response_ids = response_ids[: max(8, max_sequence_length // 2)]
        prompt_budget = max_sequence_length - len(response_ids) - 1
    prompt_ids = prompt_ids[-prompt_budget:]
    response_ids.append(tokenizer.eos_token_id)
    if not prompt_ids or not response_ids:
        raise ProductPreferenceTrainError("tokenization produced an empty sequence")
    return prompt_ids, response_ids


def _average_logp(
    model: ProductReasoningModel,
    prompt_ids: list[int],
    response_ids: list[int],
    pad_token_id: int,
) -> tuple[torch.Tensor, int]:
    loss, metrics = model.forward_batch([prompt_ids], [response_ids], pad_token_id)
    return -loss, int(metrics["charged_tokens"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.arm != "baseline":
        raise ProductPreferenceTrainError(
            "the memory-bounded preference implementation currently requires baseline arm"
        )
    if args.output.exists():
        raise ProductPreferenceTrainError(f"output already exists: {args.output}")
    if not args.warm_start_checkpoint.is_file():
        raise ProductPreferenceTrainError("warm-start checkpoint is missing")
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
    )
    model = ProductReasoningModel(
        backbone,
        args.arm,
        args.lora_layers,
        args.lora_rank,
        args.lora_alpha,
        workspace_width=512,
        workspace_slots=16,
        recurrent_steps=8,
        dense_width=192,
        unfreeze_layers=args.unfreeze_layers,
    ).to("cuda:0")
    warm_start_update, warm_metadata = load_trainable_checkpoint(
        args.warm_start_checkpoint, model
    )
    validate_warm_start_metadata(warm_metadata, args)
    warm_start_sha256 = _sha256_file(args.warm_start_checkpoint)
    rows, data_sha256 = _reservoir_pairs(args.data, args.max_rows, args.data_seed)

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
    metadata = {
        "arm": args.arm,
        "model_root": str((args.model_source_root or args.model_root).resolve()),
        "loaded_model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": resolved_loader,
        "backbone_layout": model.backbone_layout,
        "data": str(args.data.resolve()),
        "data_sha256": data_sha256,
        "selected_rows": len(rows),
        "seed": args.seed,
        "data_seed": args.data_seed,
        "lora_layers": args.lora_layers,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_projection_count": model.lora_projection_count,
        "unfreeze_layers": model.unfreeze_layers,
        "trainable_parameters": model.trainable_parameter_count(),
        "workspace_config": None,
        "workspace_architecture_sha256": None,
        "warm_start_checkpoint": str(args.warm_start_checkpoint.resolve()),
        "warm_start_sha256": warm_start_sha256,
        "warm_start_update": warm_start_update,
        "warm_start_data_sha256": warm_metadata.get("data_sha256"),
        "preference_beta": args.beta,
        "preference_margin": args.margin,
        "preference_sft_weight": args.sft_weight,
        "preference_gradient_strategy": "detached_coefficient_sequential_exact_v1",
    }

    model.train()
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    update = 0
    microstep = 0
    charged_tokens = 0
    trace: list[dict[str, float | int]] = []
    metric_sums = {
        "preference_loss": 0.0,
        "chosen_logp": 0.0,
        "rejected_logp": 0.0,
        "preference_accuracy": 0.0,
    }
    while update < args.updates:
        row = rows[microstep % len(rows)]
        prompt_chosen, chosen_ids = _tokenize_response(
            tokenizer,
            row["question"],
            row["chosen"],
            args.max_sequence_length,
        )
        prompt_rejected, rejected_ids = _tokenize_response(
            tokenizer,
            row["question"],
            row["rejected"],
            args.max_sequence_length,
        )
        if prompt_chosen != prompt_rejected:
            raise ProductPreferenceTrainError("chosen and rejected prompts differ")

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            chosen_probe, _ = _average_logp(
                model, prompt_chosen, chosen_ids, tokenizer.pad_token_id
            )
            rejected_probe, _ = _average_logp(
                model, prompt_rejected, rejected_ids, tokenizer.pad_token_id
            )
            coefficient = preference_gradient_coefficient(
                chosen_probe,
                rejected_probe,
                beta=args.beta,
                margin=args.margin,
            ).detach()
            preference_value = preference_loss_value(
                chosen_probe,
                rejected_probe,
                beta=args.beta,
                margin=args.margin,
            )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            chosen_logp, chosen_tokens = _average_logp(
                model, prompt_chosen, chosen_ids, tokenizer.pad_token_id
            )
            chosen_objective = -(
                coefficient + args.sft_weight
            ) * chosen_logp / args.gradient_accumulation
        chosen_objective.backward()
        del chosen_logp, chosen_objective

        with torch.autocast("cuda", dtype=torch.bfloat16):
            rejected_logp, rejected_tokens = _average_logp(
                model, prompt_rejected, rejected_ids, tokenizer.pad_token_id
            )
            rejected_objective = (
                coefficient * rejected_logp / args.gradient_accumulation
            )
        rejected_objective.backward()
        del rejected_logp, rejected_objective

        charged_tokens += chosen_tokens + rejected_tokens
        metric_sums["preference_loss"] += float(preference_value)
        metric_sums["chosen_logp"] += float(chosen_probe)
        metric_sums["rejected_logp"] += float(rejected_probe)
        metric_sums["preference_accuracy"] += float(chosen_probe > rejected_probe)
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
            divisor = float(args.gradient_accumulation)
            event: dict[str, float | int] = {
                "update": update,
                "preference_loss": metric_sums["preference_loss"] / divisor,
                "chosen_logp": metric_sums["chosen_logp"] / divisor,
                "rejected_logp": metric_sums["rejected_logp"] / divisor,
                "preference_accuracy": metric_sums["preference_accuracy"] / divisor,
                "gradient_norm": float(gradient_norm),
                "learning_rate": learning_rate,
                "charged_tokens": charged_tokens,
                "charged_tokens_per_second": charged_tokens / elapsed,
            }
            trace.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)
        metric_sums = {key: 0.0 for key in metric_sums}
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
        "schema": SCHEMA,
        "status": "complete",
        **metadata,
        "updates": update,
        "gradient_accumulation": args.gradient_accumulation,
        "max_sequence_length": args.max_sequence_length,
        "learning_rate": args.learning_rate,
        "charged_tokens": charged_tokens,
        "elapsed_seconds": elapsed,
        "charged_tokens_per_second": charged_tokens / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
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
    parser.add_argument("--warm-start-checkpoint", type=Path, required=True)
    parser.add_argument("--arm", choices=("baseline",), default="baseline")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--lora-layers", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--unfreeze-layers", type=int, default=2)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--sft-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--data-seed", type=int, default=20260806)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    args = parser.parse_args()
    positive = (
        args.updates,
        args.gradient_accumulation,
        args.max_rows,
        args.max_sequence_length,
        args.learning_rate,
        args.lora_layers,
        args.lora_rank,
        args.lora_alpha,
        args.beta,
        args.sft_weight,
        args.log_interval,
        args.checkpoint_interval,
    )
    if any(value <= 0 for value in positive) or args.unfreeze_layers < 0:
        parser.error("preference training dimensions must be positive")
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    print(
        f"[product-preference] updates={report['updates']} "
        f"tokens/s={report['charged_tokens_per_second']:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
