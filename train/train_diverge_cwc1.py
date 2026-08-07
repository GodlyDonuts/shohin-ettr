#!/usr/bin/env python3
"""Train one frozen DIVERGE-CWC1 matched arm."""

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
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from diverge_cwc1_data import TRAIN_ROWS, TRAIN_SEED, validate_record
from diverge_cwc1_runtime import (
    CWC1Config,
    CounterfactualWorldCommitter,
    module_state_sha256,
    tensorize_records,
)


SCHEMA = "shohin-diverge-cwc1-training-runtime-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise SystemExit("CWC1 training data hash differs")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(rows) != TRAIN_ROWS:
        raise SystemExit("CWC1 training row count differs")
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
def evaluate_fit(
    model: CounterfactualWorldCommitter,
    rows: Sequence[dict[str, Any]],
    *,
    arm: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    exact = 0
    counterfactual_exact = 0
    model.eval()
    for start in range(0, len(rows), batch_size):
        subset = rows[start : start + batch_size]
        normal = tensorize_records(subset, device)
        partner = tensorize_records(subset, device, counterfactual=True)
        if arm == "involution":
            scores = model.projected_scores(normal, partner)
            partner_scores = model.projected_scores(partner, normal)
        else:
            scores = model.raw_scores(normal)
            partner_scores = model.raw_scores(partner)
        exact += int(scores.argmax(-1).eq(normal.targets).sum())
        counterfactual_exact += int(
            partner_scores.argmax(-1).eq(1 - normal.targets).sum()
        )
    return {
        "rows": len(rows),
        "exact": exact,
        "exact_rate": exact / len(rows),
        "counterfactual_exact": counterfactual_exact,
        "counterfactual_exact_rate": counterfactual_exact / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("involution", "duplicate", "augmentation"), required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--log-interval", type=int, default=25)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing CWC1 output: {args.output}")
    if (
        args.updates != 1000
        or args.batch_size != 128
        or args.learning_rate != 0.003
        or args.seed != TRAIN_SEED
    ):
        raise SystemExit("CWC1 frozen schedule differs")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = _load(args.data, args.data_sha256)
    projection = "involution" if args.arm == "involution" else "duplicate"
    config = CWC1Config(projection_mode=projection)
    model = CounterfactualWorldCommitter(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01
    )
    generator = random.Random(args.seed)
    initial_sha256 = module_state_sha256(model)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    history = []
    source_bytes_seen = 0
    model.train()
    for update in range(1, args.updates + 1):
        subset = [rows[generator.randrange(len(rows))] for _ in range(args.batch_size)]
        normal = tensorize_records(subset, device)
        partner = tensorize_records(subset, device, counterfactual=True)
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
            if args.arm == "involution":
                first = model.raw_scores(normal)
                second = model.raw_scores(partner)
                scores = 0.5 * (first + second.flip(dims=(-1,)))
                loss = F.cross_entropy(scores, normal.targets)
            elif args.arm == "duplicate":
                first = model.raw_scores(normal)
                second = model.raw_scores(normal)
                scores = 0.5 * (first + second)
                loss = F.cross_entropy(scores, normal.targets)
            else:
                first = model.raw_scores(normal)
                second = model.raw_scores(partner)
                loss = 0.5 * (
                    F.cross_entropy(first, normal.targets)
                    + F.cross_entropy(second, 1 - normal.targets)
                )
                scores = first
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        source_bytes_seen += int(normal.attention_mask.sum() + partner.attention_mask.sum())
        if update == 1 or update % args.log_interval == 0 or update == args.updates:
            record = {
                "update": update,
                "learning_rate": learning_rate,
                "loss": float(loss.detach()),
                "batch_accuracy": float(scores.detach().argmax(-1).eq(normal.targets).float().mean()),
                "gradient_norm": float(gradient_norm),
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    elapsed = time.monotonic() - started
    fit = evaluate_fit(
        model, rows, arm=args.arm, device=device, batch_size=args.evaluation_batch_size
    )
    final_sha256 = module_state_sha256(model)
    checkpoint = args.output / "checkpoint_0001000.pt"
    _atomic_checkpoint(
        checkpoint,
        {
            "schema": SCHEMA,
            "arm": args.arm,
            "config": asdict(config),
            "model_state": model.state_dict(),
            "model_state_sha256": final_sha256,
            "data_sha256": args.data_sha256,
            "seed": args.seed,
            "updates": args.updates,
        },
    )
    report = {
        "schema": SCHEMA,
        "arm": args.arm,
        "config": asdict(config),
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "data_sha256": args.data_sha256,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_model_sha256": initial_sha256,
        "final_model_sha256": final_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_path(checkpoint),
        "elapsed_seconds": elapsed,
        "source_bytes_seen": source_bytes_seen,
        "source_bytes_per_second": source_bytes_seen / max(elapsed, 1e-9),
        "forwards_per_update": 2,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "training_evaluation": fit,
        "history": history,
    }
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    print(json.dumps({"report": str(report_path), "report_sha256": sha256_path(report_path), **report}, sort_keys=True))


if __name__ == "__main__":
    main()
