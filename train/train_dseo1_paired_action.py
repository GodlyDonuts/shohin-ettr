#!/usr/bin/env python3
"""Train the frozen DSEO1 paired autoregressive edit-action canary."""

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
from typing import Any

import torch

from dseo1_revision import draft_specific_edit_loss
from hf_product_reasoning_train import (
    PRODUCT_SYSTEM_PROMPT,
    ProductReasoningTrainError,
    _save_checkpoint,
    load_product_backbone,
    load_trainable_checkpoint,
    pack_training_embeddings,
    render_reasoning_messages,
)
from rme1_moe_revision import RME1Config, RME1ProductModel
from ttr1_revision import tokenize_with_draft_mask


DATA_SCHEMA = "shohin-dseo1-paired-presentation-v1"
DATA_REPORT_SCHEMA = "shohin-dseo1-paired-data-report-v1"
TRAIN_REPORT_SCHEMA = "shohin-dseo1-paired-training-v1"
ARMS = {"aligned", "swapped", "hidden", "final_only"}


class DSEO1TrainError(RuntimeError):
    """The frozen DSEO1 training contract differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_paired_rows(data: Path, report_path: Path) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    report = json.loads(report_path.read_text())
    expected = report.get("outputs", {}).get("train", {})
    if (
        report.get("schema") != DATA_REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("holdout_used") is not False
        or report.get("pair_balance_exact") is not True
        or report.get("train_diagnostic_source_overlap") != 0
        or report.get("complete_retention") is not True
        or int(report.get("max_sequence_length", 0)) != 4096
        or Path(str(expected.get("path", ""))).resolve() != data.resolve()
        or expected.get("sha256") != sha256_file(data)
    ):
        raise DSEO1TrainError("DSEO1 data report differs")
    rows = [json.loads(line) for line in data.read_text().splitlines() if line]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("schema") != DATA_SCHEMA:
            raise DSEO1TrainError("DSEO1 row schema differs")
        pair = grouped.setdefault(str(row["pair_identity_sha256"]), [])
        pair.append(row)
    pairs = []
    for pair_id, members in grouped.items():
        ordered = sorted(members, key=lambda row: row["pair_member"])
        if (
            len(ordered) != 2
            or {row["pair_member"] for row in ordered} != {"clean", "fault"}
            or len({row["source_identity_sha256"] for row in ordered}) != 1
            or len({row["final_response"] for row in ordered}) != 1
        ):
            raise DSEO1TrainError(f"DSEO1 pair differs: {pair_id}")
        pairs.append(ordered)
    if len(pairs) != int(expected.get("sources", -1)):
        raise DSEO1TrainError("DSEO1 pair count differs")
    return pairs, report


def _neutral_prefix(tokenizer: Any, length: int) -> list[int]:
    """Create a fixed non-action first line with the exact registered span length."""

    newline = tokenizer.encode("\n", add_special_tokens=False)
    neutral = tokenizer.encode(" EDIT", add_special_tokens=False)
    if len(newline) != 1 or len(neutral) != 1 or length < 1:
        raise DSEO1TrainError("DSEO1 neutral prefix tokenization differs")
    return [neutral[0]] * (length - 1) + newline


def tokenize_pairs(
    tokenizer: Any,
    pairs: list[list[dict[str, Any]]],
    arm: str,
    maximum: int,
) -> tuple[list[list[tuple[list[int], list[int], list[int], int]]], dict[str, Any]]:
    tokenized_pairs = []
    maxima = {"prompt": 0, "draft": 0, "action": 0, "final": 0, "total": 0}
    totals = {"prompt": 0, "draft": 0, "action": 0, "final": 0}
    for pair in pairs:
        tokenized = []
        for row in pair:
            rendered = render_reasoning_messages(
                tokenizer,
                [
                    {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                    {"role": "user", "content": row["question"]},
                ],
                enable_thinking=False,
            )
            prompt, draft_attention, _ = tokenize_with_draft_mask(tokenizer, rendered)
            if arm == "swapped":
                action = str(row["swapped_action"])
            else:
                action = str(row["action"])
            action_ids = tokenizer.encode(f"{action}\n", add_special_tokens=False)
            final_ids = tokenizer.encode(
                str(row["final_response"]), add_special_tokens=False
            ) + [tokenizer.eos_token_id]
            if arm == "final_only":
                response = _neutral_prefix(tokenizer, len(action_ids)) + final_ids
            else:
                response = action_ids + final_ids
            if len(prompt) + len(response) > maximum:
                raise DSEO1TrainError("DSEO1 selected row exceeds context")
            if len(action_ids) != int(row["action_token_count"]):
                if arm != "swapped":
                    raise DSEO1TrainError("DSEO1 action token receipt differs")
            if len(final_ids) != int(row["final_token_count"]):
                raise DSEO1TrainError("DSEO1 final token receipt differs")
            draft_count = sum(1 - value for value in draft_attention)
            if draft_count != int(row["draft_token_count"]):
                raise DSEO1TrainError("DSEO1 draft token receipt differs")
            tokenized.append((prompt, response, draft_attention, len(action_ids)))
            values = {
                "prompt": len(prompt),
                "draft": draft_count,
                "action": len(action_ids),
                "final": len(final_ids),
                "total": len(prompt) + len(response),
            }
            for key, value in values.items():
                maxima[key] = max(maxima[key], value)
                if key != "total":
                    totals[key] += value
        tokenized_pairs.append(tokenized)
    return tokenized_pairs, {"maximum_tokens": maxima, "token_totals": totals}


def paired_order(pairs: list[Any], seed: int) -> list[Any]:
    ordered = list(pairs)
    random.Random(seed).shuffle(ordered)
    return ordered


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.output.exists() or args.arm not in ARMS:
        raise DSEO1TrainError("DSEO1 output exists or arm differs")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    pairs, data_report = load_paired_rows(args.data, args.data_report)
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenized_pairs, token_receipt = tokenize_pairs(
        tokenizer, pairs, args.arm, args.max_sequence_length
    )
    ordered_pairs = paired_order(tokenized_pairs, args.data_seed)

    backbone, resolved_loader = load_product_backbone(
        args.model_root,
        "causal",
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    config = RME1Config(
        hidden_size=int(backbone.config.hidden_size),
        num_experts=int(backbone.config.num_experts),
        experts_per_token=int(backbone.config.num_experts_per_tok),
        controlled_layers=16,
        rank=18,
        alpha=18.0,
        mode="shared",
    )
    draft_control = "draft_unavailable" if args.arm == "hidden" else "normal"
    model = RME1ProductModel(
        backbone, config, draft_control=draft_control
    ).to("cuda:0")
    if model.trainable_parameter_count() != 1_179_648:
        raise DSEO1TrainError("DSEO1 trainable parameter count differs")
    owner_update, owner_metadata = load_trainable_checkpoint(
        args.owner_checkpoint, model
    )
    if (
        owner_metadata.get("architecture") != "shohin-rme1-moe-revision-v1"
        or owner_metadata.get("rme1_config") != asdict(config)
        or owner_metadata.get("rme1_draft_control") != "draft_unavailable"
    ):
        raise DSEO1TrainError("DSEO1 owner checkpoint geometry differs")

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        fused=True,
    )
    metadata = {
        "architecture": "shohin-rme1-moe-revision-v1",
        "arm": "baseline",
        "model_root": str((args.model_source_root or args.model_root).resolve()),
        "loaded_model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": resolved_loader,
        "backbone_layout": model.backbone_layout,
        "quantization": "none",
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "data_report_sha256": sha256_file(args.data_report),
        "seed": args.seed,
        "data_seed": args.data_seed,
        "rme1_config": asdict(config),
        "rme1_draft_control": draft_control,
        "trainable_parameters": model.trainable_parameter_count(),
        "trainable_parameter_name_sha256": model.trainable_parameter_name_sha256(),
        "protected_router_expert_trainables": 0,
        "protected_router_expert_parameters": model.protected_parameter_count(),
        "adapter_macs_per_token_per_layer": 2 * config.hidden_size * config.rank,
        "adapter_flops_per_token_per_layer": 4 * config.hidden_size * config.rank,
        "lora_layers": 0,
        "lora_rank": 0,
        "lora_alpha": 0.0,
        "lora_scope": "none",
        "unfreeze_layers": 0,
        "workspace_config": None,
        "warm_start_checkpoint": str(args.owner_checkpoint.resolve()),
        "warm_start_checkpoint_sha256": sha256_file(args.owner_checkpoint),
        "warm_start_update": owner_update,
        "dseo1_arm": args.arm,
        "dseo1_action_weight": 0.5,
        "dseo1_final_weight": 0.5 if args.arm != "final_only" else 1.0,
        "dseo1_final_only_action_weight": 0.0 if args.arm == "final_only" else None,
        "dseo1_pairs": len(pairs),
        "dseo1_pair_batch_size": 1,
        "dseo1_pairs_per_update": args.gradient_accumulation,
        "dseo1_data_report": data_report,
        "sequence_custody": token_receipt,
    }

    model.train()
    model.reset_routing_receipt()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    update = microstep = charged = action_charged = final_charged = 0
    trace = []
    while update < args.updates:
        pair = ordered_pairs[microstep % len(ordered_pairs)]
        prompts = [row[0] for row in pair]
        responses = [row[1] for row in pair]
        attentions = [row[2] for row in pair]
        action_lengths = [row[3] for row in pair]
        masks = attentions if draft_control == "draft_unavailable" else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs, attention, labels, batch_charged = pack_training_embeddings(
                model.text_model.embed_tokens,
                prompts,
                responses,
                None,
                tokenizer.pad_token_id,
                prompt_attention_rows=masks,
            )
            outputs = model.text_model(
                inputs_embeds=inputs, attention_mask=attention, use_cache=False
            )
            logits = model.lm_head(outputs.last_hidden_state)
            loss = draft_specific_edit_loss(
                logits,
                labels,
                action_lengths,
                action_weight=0.5,
                final_only=args.arm == "final_only",
            )
            scaled_loss = loss.total / args.gradient_accumulation
        scaled_loss.backward()
        charged += int(batch_charged)
        action_charged += loss.action_tokens
        final_charged += loss.final_tokens
        microstep += 1
        if microstep % args.gradient_accumulation:
            continue
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
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
                "total_loss": float(loss.total.detach()),
                "action_ce": float(loss.action.detach()),
                "final_ce": float(loss.final.detach()),
                "weighted_action": float(loss.weighted_action.detach()),
                "weighted_final": float(loss.weighted_final.detach()),
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
        "schema": TRAIN_REPORT_SCHEMA,
        "status": "complete",
        **metadata,
        "updates": update,
        "gradient_accumulation_pairs": args.gradient_accumulation,
        "presentations_per_microstep": 2,
        "max_sequence_length": args.max_sequence_length,
        "learning_rate": args.learning_rate,
        "charged_tokens": charged,
        "charged_action_tokens": action_charged,
        "charged_final_tokens": final_charged,
        "elapsed_seconds": elapsed,
        "charged_tokens_per_second": charged / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "routing_receipt": model.routing_receipt(),
        "trace": trace,
    }
    atomic_json(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--owner-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=2026080915)
    parser.add_argument("--data-seed", type=int, default=2026080915)
    parser.add_argument("--log-interval", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int, default=256)
    args = parser.parse_args()
    if (
        min(
            args.updates,
            args.gradient_accumulation,
            args.max_sequence_length,
            args.log_interval,
            args.checkpoint_interval,
        )
        <= 0
        or args.learning_rate <= 0
    ):
        parser.error("DSEO1 dimensions differ")
    return args


def main() -> int:
    report = run(parse_args())
    print(
        f"[dseo1-train] arm={report['dseo1_arm']} updates={report['updates']} "
        f"tokens/s={report['charged_tokens_per_second']:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
