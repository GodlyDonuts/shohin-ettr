#!/usr/bin/env python3
"""Train the frozen DSET1 model-owned span-edit script transducer."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from dset1_edit_transducer import normalized_script_loss
from hf_product_reasoning_train import (
    PRODUCT_SYSTEM_PROMPT,
    _save_checkpoint,
    load_product_backbone,
    load_trainable_checkpoint,
    pack_training_embeddings,
    render_reasoning_messages,
)
from rme1_moe_revision import RME1Config, RME1ProductModel
from ttr1_revision import tokenize_with_draft_mask


DATA_SCHEMA = "shohin-dset1-span-edit-presentation-v1"
DATA_REPORT_SCHEMA = "shohin-dset1-span-edit-data-report-v1"
TRAIN_REPORT_SCHEMA = "shohin-dset1-span-edit-training-v1"
ARMS = {"aligned", "swapped", "hidden"}


class DSET1TrainError(RuntimeError):
    """The DSET1 data, model, or optimization contract differs."""


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


def load_pairs(data: Path, report_path: Path) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    report = json.loads(report_path.read_text())
    expected = report.get("outputs", {}).get("train", {})
    if (
        report.get("schema") != DATA_REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("holdout_used") is not False
        or report.get("complete_retention") is not True
        or report.get("train_diagnostic_source_overlap") != 0
        or int(report.get("max_script_tokens", 0)) != 32
        or Path(str(expected.get("path", ""))).resolve() != data.resolve()
        or expected.get("sha256") != sha256_file(data)
    ):
        raise DSET1TrainError("DSET1 train data report differs")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in data.read_text().splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("schema") != DATA_SCHEMA:
            raise DSET1TrainError("DSET1 train row schema differs")
        grouped[str(row["pair_identity_sha256"])].append(row)
    pairs = []
    for pair_id in sorted(grouped):
        pair = sorted(grouped[pair_id], key=lambda row: row["pair_member"])
        if (
            len(pair) != 2
            or {row["pair_member"] for row in pair} != {"clean", "fault"}
            or len({row["source_identity_sha256"] for row in pair}) != 1
        ):
            raise DSET1TrainError("DSET1 train pair differs")
        pairs.append(pair)
    if len(pairs) != int(expected.get("sources", -1)):
        raise DSET1TrainError("DSET1 train pair count differs")
    return pairs, report


def tokenize_pairs(tokenizer: Any, pairs: list[list[dict[str, Any]]], arm: str, maximum: int):
    output = []
    totals = defaultdict(int)
    maxima = defaultdict(int)
    for pair in pairs:
        encoded_pair = []
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
            script_text = row["swapped_script"] if arm == "swapped" else row["script"]
            script = tokenizer.encode(script_text, add_special_tokens=False) + [
                tokenizer.eos_token_id
            ]
            if len(prompt) + len(script) > maximum:
                raise DSET1TrainError("DSET1 selected presentation overflows")
            encoded_pair.append((prompt, script, draft_attention))
            values = {
                "prompt": len(prompt),
                "draft": sum(1 - value for value in draft_attention),
                "script": len(script),
                "total": len(prompt) + len(script),
            }
            for key, value in values.items():
                maxima[key] = max(maxima[key], value)
                totals[key] += value
        output.append(encoded_pair)
    return output, {"maximum_tokens": dict(maxima), "token_totals": dict(totals)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.output.exists() or args.arm not in ARMS:
        raise DSET1TrainError("DSET1 output exists or arm differs")
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

    backbone, resolved_loader = load_product_backbone(
        args.model_root, "causal", dtype=torch.bfloat16, device_map={"": 0}
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
    model = RME1ProductModel(backbone, config, draft_control=draft_control).to("cuda:0")
    if model.trainable_parameter_count() != 1_179_648:
        raise DSET1TrainError("DSET1 trainable parameter count differs")
    owner_update, owner_metadata = load_trainable_checkpoint(args.owner_checkpoint, model)
    if (
        owner_metadata.get("architecture") != "shohin-rme1-moe-revision-v1"
        or owner_metadata.get("rme1_config") != asdict(config)
        or owner_metadata.get("rme1_draft_control") != "draft_unavailable"
    ):
        raise DSET1TrainError("DSET1 owner checkpoint differs")
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
        "warm_start_checkpoint": str(args.owner_checkpoint.resolve()),
        "warm_start_checkpoint_sha256": sha256_file(args.owner_checkpoint),
        "warm_start_update": owner_update,
        "dset1_arm": args.arm,
        "dset1_pairs": len(pairs),
        "dset1_pairs_per_update": args.gradient_accumulation,
        "dset1_script_loss": "mean_per_presentation_then_mean_pair",
        "dset1_data_report": data_report,
        "sequence_custody": token_receipt,
    }
    model.train()
    model.reset_routing_receipt()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    update = microstep = charged = script_tokens = 0
    trace = []
    while update < args.updates:
        pair = tokenized[microstep % len(tokenized)]
        prompts = [row[0] for row in pair]
        scripts = [row[1] for row in pair]
        masks = [row[2] for row in pair] if draft_control == "draft_unavailable" else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inputs, attention, labels, batch_charged = pack_training_embeddings(
                model.text_model.embed_tokens,
                prompts,
                scripts,
                None,
                tokenizer.pad_token_id,
                prompt_attention_rows=masks,
            )
            outputs = model.text_model(
                inputs_embeds=inputs, attention_mask=attention, use_cache=False
            )
            logits = model.lm_head(outputs.last_hidden_state)
            loss = normalized_script_loss(logits, labels)
            (loss / args.gradient_accumulation).backward()
        charged += int(batch_charged)
        script_tokens += sum(len(script) for script in scripts)
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
                "script_ce": float(loss.detach()),
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
        "charged_script_tokens": script_tokens,
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
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=2026080916)
    parser.add_argument("--data-seed", type=int, default=2026080916)
    parser.add_argument("--log-interval", type=int, default=8)
    parser.add_argument("--checkpoint-interval", type=int, default=512)
    args = parser.parse_args()
    if min(args.updates, args.gradient_accumulation, args.log_interval, args.checkpoint_interval) <= 0:
        parser.error("DSET1 dimensions differ")
    return args


def main() -> int:
    report = run(parse_args())
    print(
        f"[dset1-train] arm={report['dset1_arm']} updates={report['updates']} "
        f"tokens/s={report['charged_tokens_per_second']:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
