#!/usr/bin/env python3
"""Train the one frozen DIVERGE-NVE1 natural evidence compiler."""

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
import torch.nn.functional as F

from diverge_nve1_data import TRAIN_ROWS, TRAIN_SEED, validate_training_record
from diverge_nve1_runtime import (
    EvidenceCompilerConfig,
    NaturalEvidenceCompiler,
    hard_role_permutation,
    module_state_sha256,
    tensorize_sources,
)


SCHEMA = "shohin-diverge-nve1-training-report-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError("NVE1 training data hash differs")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            validate_training_record(row)
            rows.append(row)
    if len(rows) != TRAIN_ROWS:
        raise RuntimeError("NVE1 training row count differs")
    return rows


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def evaluate_training(
    model: NaturalEvidenceCompiler,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    numeric_exact = 0
    symbol_exact = 0
    joint_exact = 0
    numeric_roles = 0
    symbol_roles = 0
    total_roles = len(rows) * 2
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        (
            byte_ids,
            attention,
            numeric_bounds,
            symbol_masks,
            numeric_targets,
            symbol_targets,
        ) = tensorize_sources(batch, device)
        numeric_logits, symbol_logits = model(
            byte_ids, attention, numeric_bounds, symbol_masks
        )
        for index in range(len(batch)):
            numeric = hard_role_permutation(numeric_logits[index])
            symbols = hard_role_permutation(symbol_logits[index])
            numeric_gold = tuple(int(value) for value in numeric_targets[index])
            symbol_gold = tuple(int(value) for value in symbol_targets[index])
            numeric_ok = numeric == numeric_gold
            symbol_ok = symbols == symbol_gold
            numeric_exact += numeric_ok
            symbol_exact += symbol_ok
            joint_exact += numeric_ok and symbol_ok
            numeric_roles += sum(
                left == right for left, right in zip(numeric, numeric_gold, strict=True)
            )
            symbol_roles += sum(
                left == right for left, right in zip(symbols, symbol_gold, strict=True)
            )
    return {
        "rows": len(rows),
        "numeric_exact": numeric_exact,
        "numeric_exact_rate": numeric_exact / max(1, len(rows)),
        "symbol_exact": symbol_exact,
        "symbol_exact_rate": symbol_exact / max(1, len(rows)),
        "joint_exact": joint_exact,
        "joint_exact_rate": joint_exact / max(1, len(rows)),
        "numeric_role_accuracy": numeric_roles / max(1, total_roles),
        "symbol_role_accuracy": symbol_roles / max(1, total_roles),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--log-interval", type=int, default=25)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing NVE1 output: {args.output}")
    if (
        args.updates != 1000
        or args.batch_size != 256
        or args.learning_rate != 0.003
        or args.width != 192
        or args.layers != 2
        or args.seed != TRAIN_SEED
    ):
        raise SystemExit("NVE1 frozen training schedule differs")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = _load_jsonl(args.data, args.data_sha256)
    config = EvidenceCompilerConfig(width=args.width, layers=args.layers)
    model = NaturalEvidenceCompiler(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    generator = random.Random(args.seed)
    initial_sha256 = module_state_sha256(model)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    source_bytes = 0
    history: list[dict[str, float | int]] = []

    for update in range(1, args.updates + 1):
        indices = [generator.randrange(len(rows)) for _ in range(args.batch_size)]
        batch = [rows[index] for index in indices]
        (
            byte_ids,
            attention,
            numeric_bounds,
            symbol_masks,
            numeric_targets,
            symbol_targets,
        ) = tensorize_sources(batch, device)
        progress = update / args.updates
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
            numeric_logits, symbol_logits = model(
                byte_ids, attention, numeric_bounds, symbol_masks
            )
            numeric_loss = F.cross_entropy(
                numeric_logits.reshape(-1, 2), numeric_targets.reshape(-1)
            )
            symbol_loss = F.cross_entropy(
                symbol_logits.reshape(-1, 2), symbol_targets.reshape(-1)
            )
            loss = 0.5 * (numeric_loss + symbol_loss)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        source_bytes += int(attention.sum())
        if update == 1 or update % args.log_interval == 0 or update == args.updates:
            record = {
                "update": update,
                "learning_rate": learning_rate,
                "loss": float(loss.detach()),
                "numeric_loss": float(numeric_loss.detach()),
                "symbol_loss": float(symbol_loss.detach()),
                "raw_numeric_accuracy": float(
                    numeric_logits.detach()
                    .argmax(dim=-1)
                    .eq(numeric_targets)
                    .float()
                    .mean()
                ),
                "raw_symbol_accuracy": float(
                    symbol_logits.detach()
                    .argmax(dim=-1)
                    .eq(symbol_targets)
                    .float()
                    .mean()
                ),
                "gradient_norm": float(gradient_norm),
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    elapsed = time.monotonic() - started
    model.eval()
    training_evaluation = evaluate_training(
        model,
        rows,
        device=device,
        batch_size=args.evaluation_batch_size,
    )
    final_sha256 = module_state_sha256(model)
    checkpoint = args.output / f"checkpoint_{args.updates:07d}.pt"
    _atomic_checkpoint(
        checkpoint,
        {
            "schema": SCHEMA,
            "update": args.updates,
            "seed": args.seed,
            "config": asdict(config),
            "model_state": model.state_dict(),
            "model_state_sha256": final_sha256,
            "data_sha256": args.data_sha256,
        },
    )
    report = {
        "schema": SCHEMA,
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "rows": len(rows),
        "source_bytes_seen": source_bytes,
        "elapsed_seconds": elapsed,
        "source_bytes_per_second": source_bytes / max(elapsed, 1e-9),
        "device": str(device),
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        ),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "initial_model_sha256": initial_sha256,
        "final_model_sha256": final_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_path(checkpoint),
        "data": str(args.data),
        "data_sha256": args.data_sha256,
        "config": asdict(config),
        "history": history,
        "training_evaluation": training_evaluation,
    }
    _atomic_json(args.output / "report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
