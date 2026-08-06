#!/usr/bin/env python3
"""Train the one frozen DIVERGE-TOL1 typed operation compiler."""

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

from diverge_tol1_data import CLAUSE_OPS
from diverge_tol1_product import (
    compiler_loss,
    evaluate_programs,
    flatten_clauses,
    load_rows,
    sha256_path,
    tensorize_clauses,
)
from diverge_tol1_runtime import TOL1Config, TypedOperationCompiler


SCHEMA = "shohin-diverge-tol1-training-report-v1"


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
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
    parser.add_argument("--fta1-checkpoint", type=Path)
    parser.add_argument("--fta1-checkpoint-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--development-rows", type=int, default=512)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026080504)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing output: {args.output}")
    if min(args.updates, args.batch_size, args.evaluation_batch_size) <= 0:
        raise SystemExit("TOL1 training dimensions must be positive")
    if (args.fta1_checkpoint is None) != (args.fta1_checkpoint_sha256 is None):
        raise SystemExit("FTA1 checkpoint and hash must be supplied together")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = load_rows(args.data, args.data_sha256, "train")
    development_rows = load_rows(
        args.development_data, args.development_data_sha256, "development"
    )[: args.development_rows]
    clauses = flatten_clauses(train_rows)
    buckets: dict[int, list[int]] = defaultdict(list)
    for index, clause in enumerate(clauses):
        buckets[clause.operation_id].append(index)
    if set(buckets) != set(range(len(CLAUSE_OPS))):
        raise SystemExit("TOL1 training board does not cover every opcode")
    config = TOL1Config(width=args.width, layers=args.layers)
    model = TypedOperationCompiler(config).to(device)
    warm_start: dict[str, Any] | None = None
    if args.fta1_checkpoint is not None:
        if sha256_path(args.fta1_checkpoint) != args.fta1_checkpoint_sha256:
            raise SystemExit("FTA1 warm-start checkpoint hash differs")
        checkpoint = torch.load(args.fta1_checkpoint, map_location="cpu", weights_only=False)
        loaded = model.initialize_fta1_encoder(checkpoint["model_state"])
        warm_start = {
            "checkpoint": str(args.fta1_checkpoint),
            "checkpoint_sha256": args.fta1_checkpoint_sha256,
            "loaded_tensors": loaded,
        }
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01
    )
    initial_sha256 = _state_sha256(model)
    args.output.mkdir(parents=True)
    generator = random.Random(args.seed)
    started = time.monotonic()
    source_bytes = 0
    history: list[dict[str, float | int]] = []
    opcode_cycle = tuple(range(len(CLAUSE_OPS)))

    for update in range(1, args.updates + 1):
        indices = [
            generator.choice(buckets[opcode_cycle[(update * args.batch_size + offset) % len(opcode_cycle)]])
            for offset in range(args.batch_size)
        ]
        batch = [clauses[index] for index in indices]
        tensors, _ = tensorize_clauses(batch, device)
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
            outputs = model(
                tensors["byte_ids"],
                tensors["attention"],
                tensors["candidate_batch"],
                tensors["candidate_start"],
                tensors["candidate_end"],
            )
            loss, metrics = compiler_loss(outputs, tensors)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise SystemExit("non-finite TOL1 gradient")
        optimizer.step()
        source_bytes += int(tensors["attention"].sum())
        if update == 1 or update % args.log_interval == 0 or update == args.updates:
            record = {
                "update": update,
                "learning_rate": learning_rate,
                "gradient_norm": float(gradient_norm),
                **metrics,
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    elapsed = time.monotonic() - started
    development = evaluate_programs(
        model,
        development_rows,
        device=device,
        batch_size=args.evaluation_batch_size,
    )
    final_sha256 = _state_sha256(model)
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    checkpoint_path = args.output / f"checkpoint_{args.updates:07d}.pt"
    _atomic_checkpoint(
        checkpoint_path,
        {
            "schema": SCHEMA,
            "update": args.updates,
            "seed": args.seed,
            "config": asdict(config),
            "model_state": model.state_dict(),
            "model_state_sha256": final_sha256,
            "data_sha256": args.data_sha256,
            "development_data_sha256": args.development_data_sha256,
            "warm_start": warm_start,
        },
    )
    checkpoint_sha256 = sha256_path(checkpoint_path)
    report = {
        "schema": SCHEMA,
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "train_rows": len(train_rows),
        "train_clauses": len(clauses),
        "source_bytes_seen": source_bytes,
        "elapsed_seconds": elapsed,
        "source_bytes_per_second": source_bytes / max(elapsed, 1e-9),
        "peak_allocated_bytes": peak,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_model_sha256": initial_sha256,
        "final_model_sha256": final_sha256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "data": str(args.data),
        "data_sha256": args.data_sha256,
        "development_data": str(args.development_data),
        "development_data_sha256": args.development_data_sha256,
        "warm_start": warm_start,
        "config": asdict(config),
        "history": history,
        "development": development,
    }
    _atomic_json(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "elapsed_seconds": elapsed,
                "development_answers": development["counts"]["treatment_answer"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
