#!/usr/bin/env python3
"""Train the one frozen DIVERGE-ATS1 source-role compiler."""

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

from diverge_ats1_data import build_segments
from diverge_ats1_product import evaluate_model, load_jsonl, sha256_path, tensorize_segments
from diverge_ats1_runtime import ATS1Config, SourceRoleCompiler, compiler_loss


SCHEMA = "shohin-diverge-ats1-training-report-v1"


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        value = tensor.detach().cpu().contiguous().view(torch.uint8)
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--development-data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--evaluation-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-multiplier", type=int, default=4)
    parser.add_argument("--development-rows", type=int, default=96)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026080606)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing output: {args.output}")
    if min(args.updates, args.batch_size, args.evaluation_batch_size) <= 0:
        raise SystemExit("training dimensions must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_jsonl(args.data, args.data_sha256)
    development = load_jsonl(
        args.development_data, args.development_data_sha256
    )[: args.development_rows]
    segments = build_segments(rows)
    generator = random.Random(args.seed)
    config = ATS1Config(
        width=args.width,
        layers=args.layers,
        heads=args.heads,
        ff_multiplier=args.ff_multiplier,
    )
    model = SourceRoleCompiler(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01
    )
    initial_sha256 = _state_sha256(model)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    tokens = 0
    history: list[dict[str, float | int]] = []

    for update in range(1, args.updates + 1):
        indices = [generator.randrange(len(segments)) for _ in range(args.batch_size)]
        batch = [segments[index] for index in indices]
        byte_ids, attention, role_targets, operation_targets = tensorize_segments(
            batch, device
        )
        progress = update / max(1, args.updates)
        learning_rate = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with autocast:
            role_logits, operation_logits = model(byte_ids, attention)
            loss, metrics = compiler_loss(
                role_logits,
                operation_logits,
                role_targets,
                operation_targets,
                attention,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens += int(attention.sum())
        if update == 1 or update % args.log_interval == 0 or update == args.updates:
            record = {
                "update": update,
                "learning_rate": learning_rate,
                **metrics,
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    elapsed = time.monotonic() - started
    development_report = evaluate_model(
        model,
        development,
        device=device,
        batch_size=args.evaluation_batch_size,
    )
    final_sha256 = _state_sha256(model)
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    checkpoint = args.output / f"checkpoint_{args.updates:07d}.pt"
    checkpoint_payload = {
        "schema": SCHEMA,
        "update": args.updates,
        "seed": args.seed,
        "config": asdict(config),
        "model_state": model.state_dict(),
        "model_state_sha256": final_sha256,
        "data_sha256": args.data_sha256,
        "development_data_sha256": args.development_data_sha256,
    }
    _atomic_checkpoint(checkpoint, checkpoint_payload)
    checkpoint_sha256 = sha256_path(checkpoint)
    report = {
        "schema": SCHEMA,
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "train_rows": len(rows),
        "train_segments": len(segments),
        "source_bytes_seen": tokens,
        "elapsed_seconds": elapsed,
        "source_bytes_per_second": tokens / max(elapsed, 1e-9),
        "peak_allocated_bytes": peak,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_model_sha256": initial_sha256,
        "final_model_sha256": final_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "data": str(args.data),
        "data_sha256": args.data_sha256,
        "development_data": str(args.development_data),
        "development_data_sha256": args.development_data_sha256,
        "config": asdict(config),
        "history": history,
        "development": development_report,
    }
    _atomic_json(args.output / "report.json", report)
    print(json.dumps({"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha256, "elapsed_seconds": elapsed}, sort_keys=True))


if __name__ == "__main__":
    main()
