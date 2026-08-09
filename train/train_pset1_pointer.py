#!/usr/bin/env python3
"""Train the frozen PSET1 two-stream pointer/edit head."""

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

from hf_product_reasoning_eval import _load_model
from pset1_pointer_transducer import PSET1Config, PSET1PointerHead, pointer_loss
from pset1_runtime import (
    host_hidden,
    load_rows,
    pad_characters,
    pad_ids,
    replacement_batch,
    sha256_file,
    tokenize_rows,
)


REPORT_SCHEMA = "shohin-pset1-pointer-training-v1"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    if args.output.exists() or args.arm not in {"aligned", "permuted"}:
        raise RuntimeError("PSET1 output exists or arm differs")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows, data_report = load_rows(args.data, args.data_report, "train")
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenized = tokenize_rows(tokenizer, rows)
    random.Random(args.seed).shuffle(tokenized)
    host, metadata, loader = _load_model(args.model_root, args.host_checkpoint, "causal")
    if metadata.get("dset1_arm") != "aligned" or int(metadata.get("update", 0)) not in (0, 512):
        # update lives in the checkpoint payload on some historical writers.
        if metadata.get("dset1_arm") != "aligned":
            raise RuntimeError("PSET1 host checkpoint differs")
    host.requires_grad_(False).eval()
    config = PSET1Config(host_hidden_size=int(host.text_model.config.hidden_size))
    head = PSET1PointerHead(config).to("cuda:0")
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), fused=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trace = []
    charged_source = charged_draft = charged_characters = 0
    for update in range(1, args.updates + 1):
        row = tokenized[(update - 1) % len(tokenized)]
        members = [row["members"]["clean"], row["members"]["fault"]]
        source_ids, source_mask = pad_ids([row["source_ids"]], tokenizer.pad_token_id, torch.device("cuda:0"))
        draft_ids, draft_mask = pad_ids([member["draft_ids"] for member in members], tokenizer.pad_token_id, torch.device("cuda:0"))
        mapping, character_ids, character_mask = pad_characters(members, torch.device("cuda:0"))
        source_hidden = host_hidden(host, source_ids, source_mask).expand(2, -1, -1)
        source_mask_pair = source_mask.expand(2, -1)
        draft_hidden = host_hidden(host, draft_ids, draft_mask)
        actions, pointers, replacement_inputs, replacement_labels = replacement_batch(members, args.arm, torch.device("cuda:0"))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            source, characters, action_logits, pointer_logits = head.encode(
                source_hidden,
                source_mask_pair,
                draft_hidden,
                draft_mask,
                mapping,
                character_ids,
                character_mask,
            )
            replacement_logits = head.replacement_logits(
                source, source_mask_pair, characters, pointers, replacement_inputs
            )
            loss, components = pointer_loss(
                action_logits,
                pointer_logits,
                replacement_logits,
                actions,
                pointers,
                replacement_labels,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        progress = (update - 1) / max(args.updates - 1, 1)
        lr = args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        charged_source += len(row["source_ids"])
        charged_draft += sum(len(member["draft_ids"]) for member in members)
        charged_characters += sum(len(member["character_ids"]) for member in members)
        if update == 1 or update % args.log_interval == 0:
            event = {
                "update": update,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient),
                "learning_rate": lr,
                **{name: float(value) for name, value in components.items()},
            }
            trace.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)
    elapsed = time.monotonic() - started
    checkpoint = args.output / f"checkpoint_{args.updates:07d}.pt"
    checkpoint_payload = {
        "head_state_dict": head.state_dict(),
        "metadata": {
            "schema": REPORT_SCHEMA,
            "arm": args.arm,
            "updates": args.updates,
            "config": asdict(config),
            "host_checkpoint": str(args.host_checkpoint.resolve()),
            "host_checkpoint_sha256": sha256_file(args.host_checkpoint),
            "data_sha256": sha256_file(args.data),
            "data_report_sha256": sha256_file(args.data_report),
            "trainable_parameters": head.trainable_parameter_count(),
            "trainable_parameter_name_sha256": head.trainable_parameter_name_sha256(),
            "seed": args.seed,
        },
    }
    torch.save(checkpoint_payload, checkpoint)
    report = {
        **checkpoint_payload["metadata"],
        "status": "complete",
        "model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": loader,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_tokens": charged_source,
        "draft_tokens": charged_draft,
        "draft_characters": charged_characters,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "trace": trace,
        "data_report": data_report,
    }
    atomic_json(args.output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--host-checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=["aligned", "permuted"], required=True)
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026080917)
    parser.add_argument("--log-interval", type=int, default=8)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
