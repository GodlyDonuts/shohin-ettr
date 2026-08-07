#!/usr/bin/env python3
"""Train one matched pretrained-query arm for DIVERGE-PQI1."""

from __future__ import annotations

import argparse
from collections import Counter
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
from tokenizers import Tokenizer

from diverge_iem1_runtime import tensorize_queries
from diverge_pqi1_runtime import (
    PQI1Config,
    PretrainedQueryGrounder,
    adapter_state_dict,
    adapter_state_sha256,
)
from diverge_rrg1_data import ROWS_PER_STAGE, validate_training_record
from diverge_rrg1_runtime import permutation_scores, permutation_targets
from frozen_pointer_backbone import load_frozen_pointer_backbone


SCHEMA = "shohin-diverge-pqi1-training-report-v1"
TRAIN_SEED = 2026080631


class PQI1TrainingError(RuntimeError):
    """The frozen PQI1 training contract was violated."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rows(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise PQI1TrainingError("PQI1 query training hash differs")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            validate_training_record(row)
            if row["stage"] != "QUERY":
                raise PQI1TrainingError("PQI1 training stage differs")
            rows.append(row)
    if len(rows) != ROWS_PER_STAGE:
        raise PQI1TrainingError("PQI1 training row count differs")
    return rows


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


@torch.no_grad()
def _evaluate(
    model: PretrainedQueryGrounder,
    rows: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    counts = Counter()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        ids, mask, symbols, targets = tensorize_queries(batch, device)
        predicted = permutation_scores(model(ids, mask, symbols)).argmax(-1)
        expected = permutation_targets(targets)
        counts["total"] += len(batch)
        counts["exact"] += int(predicted.eq(expected).sum())
    return {**dict(counts), "exact_rate": counts["exact"] / counts["total"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--backbone-name", choices=("shohin", "smollm2"), required=True)
    parser.add_argument("--query-data", type=Path, required=True)
    parser.add_argument("--query-data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--shuffle-supervision", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing PQI1 output: {args.output}")
    if args.batch_size != 128 or args.learning_rate != 1e-3 or args.seed != TRAIN_SEED:
        raise SystemExit("PQI1 frozen training schedule differs")
    if sha256_path(args.base) != args.base_sha256:
        raise SystemExit("PQI1 base hash differs")
    if sha256_path(args.tokenizer) != args.tokenizer_sha256:
        raise SystemExit("PQI1 tokenizer hash differs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("PQI1 requested unavailable CUDA")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    rows = _load_rows(args.query_data, args.query_data_sha256)
    order = list(range(len(rows)))
    random.Random(args.seed ^ 0x50514931).shuffle(order)
    shuffled_targets = None
    if args.shuffle_supervision:
        shuffled_targets = [int(row["symbol_role_ids"][0]) for row in rows]
        random.Random(args.seed ^ 0x53485546).shuffle(shuffled_targets)

    backbone, _, receipt = load_frozen_pointer_backbone(args.base, device=device)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    config = PQI1Config()
    model = PretrainedQueryGrounder(backbone, tokenizer, config).to(device)
    parameters = list(model.adapter_parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01
    )
    updates = math.ceil(len(order) / args.batch_size)
    history = []
    started = time.monotonic()
    model.train()
    for update, start in enumerate(range(0, len(order), args.batch_size), start=1):
        indices = order[start : start + args.batch_size]
        batch = [rows[index] for index in indices]
        ids, mask, symbols, targets = tensorize_queries(batch, device)
        expected = permutation_targets(targets)
        if shuffled_targets is not None:
            expected = torch.tensor(
                [shuffled_targets[index] for index in indices],
                dtype=torch.long,
                device=device,
            )
        progress = update / updates
        learning_rate = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        scores = permutation_scores(model(ids, mask, symbols))
        loss = F.cross_entropy(scores, expected)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(gradient_norm):
            raise SystemExit("non-finite PQI1 gradient")
        optimizer.step()
        if update == 1 or update % args.log_interval == 0 or update == updates:
            record = {
                "update": update,
                "updates": updates,
                "learning_rate": learning_rate,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    elapsed = time.monotonic() - started
    training_evaluation = _evaluate(
        model, rows, device=device, batch_size=args.batch_size
    )
    args.output.mkdir(parents=True)
    checkpoint_path = args.output / "checkpoint.pt"
    checkpoint = {
        "schema": SCHEMA,
        "config": asdict(config),
        "backbone_name": args.backbone_name,
        "base_sha256": args.base_sha256,
        "tokenizer_sha256": args.tokenizer_sha256,
        "query_data_sha256": args.query_data_sha256,
        "seed": args.seed,
        "updates": updates,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "shuffle_supervision": args.shuffle_supervision,
        "adapter_state": adapter_state_dict(model),
        "adapter_state_sha256": adapter_state_sha256(model),
    }
    _atomic_checkpoint(checkpoint_path, checkpoint)
    report = {
        "schema": SCHEMA,
        "config": asdict(config),
        "backbone_name": args.backbone_name,
        "base": str(args.base),
        "base_sha256": args.base_sha256,
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": args.tokenizer_sha256,
        "query_data": str(args.query_data),
        "query_data_sha256": args.query_data_sha256,
        "seed": args.seed,
        "updates": updates,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "shuffle_supervision": args.shuffle_supervision,
        "elapsed_seconds": elapsed,
        "examples": len(rows),
        "examples_per_second": len(rows) / max(elapsed, 1e-9),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "adapter_state_sha256": checkpoint["adapter_state_sha256"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "backbone_receipt": asdict(receipt),
        "training_evaluation": training_evaluation,
        "history": history,
    }
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "report": str(report_path),
        "report_sha256": sha256_path(report_path),
        "training_evaluation": training_evaluation,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
