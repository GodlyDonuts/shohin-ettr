#!/usr/bin/env python3
"""Train the frozen DIVERGE-EWC1 treatment or absolute-role control."""

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
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from diverge_ewc1_data import (
    TRAIN_ROWS,
    TRAIN_SEED,
    scan_integer_spans,
    validate_record,
)
from diverge_ewc1_runtime import (
    CompilerMode,
    EquivariantWorldCompiler,
    WorldCompilerConfig,
    hard_numeric_assignment,
    module_state_sha256,
    tensorize_worlds,
)


SCHEMA = "shohin-diverge-ewc1-training-runtime-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise SystemExit("EWC1 training data hash differs")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != TRAIN_ROWS:
        raise SystemExit("EWC1 training row count differs")
    for row in rows:
        validate_record(row)
    return rows


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_checkpoint(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@torch.no_grad()
def evaluate_rows(
    model: EquivariantWorldCompiler,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    model.eval()
    initial_exact = 0
    operation_exact = 0
    joint_exact = 0
    for start in range(0, len(rows), batch_size):
        subset = rows[start : start + batch_size]
        batch = tensorize_worlds(subset, device)
        numeric_logits, operation_logits = model(batch)
        for index, row in enumerate(subset):
            numeric_count = int(batch.numeric_mask[index].sum())
            assignment = hard_numeric_assignment(numeric_logits[index], numeric_count)
            numeric_spans = scan_integer_spans(str(row["source_text"]))
            predicted_initial = tuple(
                int(str(row["source_text"])[numeric_spans[value][0] : numeric_spans[value][1]])
                for value in assignment
            )
            alias_count = int(batch.alias_mask[index].sum())
            group_ids = batch.alias_group_ids[index, :alias_count]
            selected = tuple(
                int(group_ids[position])
                for position, logit in enumerate(operation_logits[index, :alias_count])
                if float(logit) >= 0.0
            )
            initial_ok = predicted_initial == tuple(int(value) for value in row["initial_state"])
            operation_ok = selected == tuple(int(value) for value in row["symbols"])
            initial_exact += initial_ok
            operation_exact += operation_ok
            joint_exact += initial_ok and operation_ok
    return {
        "rows": len(rows),
        "initial_exact": initial_exact,
        "initial_exact_rate": initial_exact / len(rows),
        "operation_exact": operation_exact,
        "operation_exact_rate": operation_exact / len(rows),
        "joint_exact": joint_exact,
        "joint_exact_rate": joint_exact / len(rows),
    }


def _balanced_operation_loss(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    positive = mask & targets.bool()
    negative = mask & ~targets.bool()
    if not torch.any(positive) or not torch.any(negative):
        raise RuntimeError("EWC1 training batch lacks one operation class")
    return 0.5 * (
        F.softplus(-logits[positive]).mean() + F.softplus(logits[negative]).mean()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("equivariant", "absolute"), required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--log-interval", type=int, default=25)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing EWC1 output: {args.output}")
    if (
        args.updates != 1000
        or args.batch_size != 256
        or args.learning_rate != 0.003
        or args.seed != TRAIN_SEED
    ):
        raise SystemExit("EWC1 frozen training schedule differs")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = _load_jsonl(args.data, args.data_sha256)
    config = WorldCompilerConfig(mode=args.mode)
    model = EquivariantWorldCompiler(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01
    )
    generator = random.Random(args.seed)
    initial_sha256 = module_state_sha256(model)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    history: list[dict[str, float | int]] = []
    source_bytes_seen = 0

    model.train()
    for update in range(1, args.updates + 1):
        indices = [generator.randrange(len(rows)) for _ in range(args.batch_size)]
        subset = [rows[index] for index in indices]
        batch = tensorize_worlds(subset, device)
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
            numeric_logits, operation_logits = model(batch)
            numeric_loss = F.cross_entropy(
                numeric_logits.reshape(-1, numeric_logits.shape[-1]),
                batch.numeric_targets.reshape(-1),
            )
            operation_loss = _balanced_operation_loss(
                operation_logits, batch.operation_targets, batch.alias_mask
            )
            loss = 0.5 * (numeric_loss + operation_loss)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        source_bytes_seen += int(batch.attention_mask.sum())
        if update == 1 or update % args.log_interval == 0 or update == args.updates:
            record = {
                "update": update,
                "learning_rate": learning_rate,
                "loss": float(loss.detach()),
                "numeric_loss": float(numeric_loss.detach()),
                "operation_loss": float(operation_loss.detach()),
                "gradient_norm": float(gradient_norm),
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    elapsed = time.monotonic() - started
    training_evaluation = evaluate_rows(
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
        "mode": args.mode,
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "rows": len(rows),
        "source_bytes_seen": source_bytes_seen,
        "elapsed_seconds": elapsed,
        "source_bytes_per_second": source_bytes_seen / max(elapsed, 1e-9),
        "device": str(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
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
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    print(json.dumps({"report": str(report_path), "report_sha256": sha256_path(report_path), **report}, sort_keys=True))


if __name__ == "__main__":
    main()
