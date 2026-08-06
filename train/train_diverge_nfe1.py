#!/usr/bin/env python3
"""Train the one frozen DIVERGE-NFE1 whole-mention role model."""

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

from diverge_nfe1_data import validate_training_record
from diverge_nfe1_runtime import (
    MentionConfig,
    WholeMentionRoleModel,
    compile_mentions_batch,
    hard_role_permutation,
    tensorize_sources,
)


SCHEMA = "shohin-diverge-nfe1-mention-training-report-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(
            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def _load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError("NFE1 training data hash differs")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            validate_training_record(row)
            rows.append(row)
    return rows


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def evaluate_training(
    model: WholeMentionRoleModel,
    rows: list[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    exact = 0
    mentions = 0
    role_correct = 0
    value_exact = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        byte_ids, attention, bounds, targets = tensorize_sources(batch, device)
        logits = model(byte_ids, attention, bounds).cpu()
        targets = targets.cpu()
        compiled = compile_mentions_batch(model, batch, device=device)
        for index, record in enumerate(batch):
            assignment = hard_role_permutation(logits[index])
            exact += assignment == tuple(int(value) for value in targets[index])
            role_correct += sum(
                predicted == int(target)
                for predicted, target in zip(assignment, targets[index], strict=True)
            )
            mentions += 3
            by_role = {mention.role: mention.value for mention in compiled[index]}
            value_exact += (
                by_role.get("LHS") == int(record["lhs"])
                and by_role.get("ARGUMENT") == int(record["argument"])
                and by_role.get("RHS") == int(record["rhs"])
            )
    return {
        "rows": len(rows),
        "exact_assignments": exact,
        "exact_assignment_rate": exact / max(1, len(rows)),
        "role_accuracy": role_correct / max(1, mentions),
        "exact_values": value_exact,
        "exact_value_rate": value_exact / max(1, len(rows)),
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
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026080608)
    parser.add_argument("--log-interval", type=int, default=25)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing NFE1 output: {args.output}")
    if min(args.updates, args.batch_size, args.evaluation_batch_size) <= 0:
        raise SystemExit("NFE1 training dimensions must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = _load_jsonl(args.data, args.data_sha256)
    if len(rows) != 2179:
        raise SystemExit("NFE1 training row count differs")
    config = MentionConfig(width=args.width, layers=args.layers)
    model = WholeMentionRoleModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    generator = random.Random(args.seed)
    initial_sha256 = _state_sha256(model)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    source_bytes = 0
    history: list[dict[str, float | int]] = []

    for update in range(1, args.updates + 1):
        indices = [generator.randrange(len(rows)) for _ in range(args.batch_size)]
        batch = [rows[index] for index in indices]
        byte_ids, attention, bounds, targets = tensorize_sources(batch, device)
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
            logits = model(byte_ids, attention, bounds)
            loss = F.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1))
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        source_bytes += int(attention.sum())
        if update == 1 or update % args.log_interval == 0 or update == args.updates:
            predicted = logits.detach().argmax(dim=-1)
            record = {
                "update": update,
                "learning_rate": learning_rate,
                "loss": float(loss.detach()),
                "raw_role_accuracy": float(predicted.eq(targets).float().mean()),
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
    final_sha256 = _state_sha256(model)
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
