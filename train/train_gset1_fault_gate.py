#!/usr/bin/env python3
"""Fit the paired contrastive GSET1 edit-action owner on frozen decision states."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from gset1_fault_gate import GSET1Config, GSET1FaultGate, save_gate_checkpoint
from train_dset1_span_edit import atomic_json, sha256_file


SCHEMA = "shohin-gset1-fault-gate-training-v1"
FEATURE_SCHEMA = "shohin-gset1-feature-shard-v1"


def load_features(paths: list[Path], expected_control: str):
    rows = []
    states = []
    checkpoint_sha = data_sha = report_sha = hidden_size = None
    shard_indices = set()
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("schema") != FEATURE_SCHEMA
            or payload.get("status") != "complete"
            or payload.get("control") != expected_control
            or payload.get("holdout_used") is not False
        ):
            raise RuntimeError("GSET1 feature shard differs")
        shard_indices.add(int(payload["shard_index"]))
        values = (
            payload["checkpoint_sha256"],
            payload["data_sha256"],
            payload["data_report_sha256"],
            int(payload["hidden_size"]),
        )
        if checkpoint_sha is None:
            checkpoint_sha, data_sha, report_sha, hidden_size = values
        elif values != (checkpoint_sha, data_sha, report_sha, hidden_size):
            raise RuntimeError("GSET1 feature provenance differs")
        tensor = payload["states"]
        records = payload["records"]
        if not isinstance(tensor, torch.Tensor) or len(records) != tensor.shape[0]:
            raise RuntimeError("GSET1 feature tensor differs")
        rows.extend(records)
        states.append(tensor.to(torch.float32))
    if shard_indices != set(range(len(paths))):
        raise RuntimeError("GSET1 feature shard coverage differs")
    state = torch.cat(states, dim=0)
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["pair_identity_sha256"])].append((index, row))
    ordered = []
    for pair_id in sorted(grouped):
        pair = sorted(grouped[pair_id], key=lambda item: item[1]["pair_member"])
        if len(pair) != 2 or {item[1]["pair_member"] for item in pair} != {"clean", "fault"}:
            raise RuntimeError("GSET1 feature pair differs")
        ordered.append(pair)
    indices = torch.tensor([[item[0] for item in pair] for pair in ordered], dtype=torch.long)
    pair_states = state[indices]
    pair_rows = [[item[1] for item in pair] for pair in ordered]
    return pair_states, pair_rows, {
        "checkpoint_sha256": checkpoint_sha,
        "data_sha256": data_sha,
        "data_report_sha256": report_sha,
        "hidden_size": hidden_size,
        "feature_shards": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)} for path in paths
        ],
    }


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise RuntimeError("GSET1 fit output exists")
    args.output.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    control = "hidden" if args.arm == "hidden" else "aligned"
    pair_states, pair_rows, receipt = load_features(args.features, control)
    if args.arm == "swapped":
        labels = torch.tensor([[1, 0]] * len(pair_rows), dtype=torch.long)
    else:
        labels = torch.tensor([[0, 1]] * len(pair_rows), dtype=torch.long)
    config = GSET1Config(hidden_size=int(receipt["hidden_size"]), gate_width=args.gate_width)
    gate = GSET1FaultGate(config).to("cuda:0")
    expected = 2 * config.hidden_size + config.hidden_size * config.gate_width + config.gate_width + 2 * config.gate_width + 2
    if gate.trainable_parameter_count() != expected:
        raise RuntimeError("GSET1 trainable count differs")
    optimizer = torch.optim.AdamW(
        gate.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01, fused=True
    )
    pair_states = pair_states.pin_memory()
    labels = labels.pin_memory()
    generator = torch.Generator().manual_seed(args.data_seed)
    permutation = torch.randperm(len(pair_rows), generator=generator)
    cursor = 0
    started = time.monotonic()
    trace = []
    for update in range(1, args.updates + 1):
        if cursor + args.batch_pairs > len(permutation):
            permutation = torch.randperm(len(pair_rows), generator=generator)
            cursor = 0
        indices = permutation[cursor : cursor + args.batch_pairs]
        cursor += args.batch_pairs
        hidden = pair_states[indices].to("cuda:0", non_blocking=True)
        target = labels[indices].to("cuda:0", non_blocking=True)
        logits = gate(hidden.reshape(-1, config.hidden_size)).reshape(-1, 2, 2)
        cross_entropy = F.cross_entropy(logits.reshape(-1, 2), target.reshape(-1))
        replace_margin = logits[..., 1] - logits[..., 0]
        positive = torch.gather(replace_margin, 1, target.argmax(dim=1, keepdim=True)).squeeze(1)
        negative = torch.gather(replace_margin, 1, (1 - target).argmax(dim=1, keepdim=True)).squeeze(1)
        contrastive = F.softplus(args.margin - (positive - negative)).mean()
        loss = 0.5 * (cross_entropy + contrastive)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
        progress = (update - 1) / max(args.updates - 1, 1)
        learning_rate = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        if update == 1 or update % args.log_interval == 0:
            prediction = logits.argmax(dim=-1)
            event = {
                "update": update,
                "loss": float(loss.detach()),
                "cross_entropy": float(cross_entropy.detach()),
                "paired_contrastive": float(contrastive.detach()),
                "batch_action_accuracy": float((prediction == target).float().mean()),
                "gradient_norm": float(gradient_norm),
                "learning_rate": learning_rate,
            }
            trace.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)
    metadata = {
        "architecture": "shohin-gset1-paired-fault-gate-v1",
        "arm": args.arm,
        "feature_control": control,
        "label_control": "within_pair_swapped" if args.arm == "swapped" else "true",
        "seed": args.seed,
        "data_seed": args.data_seed,
        "paired_sources": len(pair_rows),
        "trainable_parameters": gate.trainable_parameter_count(),
        "trainable_parameter_name_sha256": gate.trainable_parameter_name_sha256(),
        "base_dset_checkpoint_sha256": receipt["checkpoint_sha256"],
        "data_sha256": receipt["data_sha256"],
        "data_report_sha256": receipt["data_report_sha256"],
        "feature_shards": receipt["feature_shards"],
        "loss": {
            "cross_entropy_weight": 0.5,
            "paired_contrastive_weight": 0.5,
            "margin": args.margin,
        },
    }
    checkpoint = args.output / f"checkpoint_{args.updates:07d}.pt"
    save_gate_checkpoint(checkpoint, gate, optimizer, args.updates, metadata)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        **metadata,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "updates": args.updates,
        "batch_pairs": args.batch_pairs,
        "learning_rate": args.learning_rate,
        "elapsed_seconds": time.monotonic() - started,
        "trace": trace,
    }
    atomic_json(args.output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=["aligned", "swapped", "hidden"], required=True)
    parser.add_argument("--updates", type=int, default=1024)
    parser.add_argument("--batch-pairs", type=int, default=256)
    parser.add_argument("--gate-width", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2026080922)
    parser.add_argument("--data-seed", type=int, default=2026080922)
    parser.add_argument("--log-interval", type=int, default=64)
    args = parser.parse_args()
    if min(args.updates, args.batch_pairs, args.gate_width, args.log_interval) <= 0:
        parser.error("GSET1 training geometry differs")
    report = run(args)
    print(json.dumps({"arm": report["arm"], "updates": report["updates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
