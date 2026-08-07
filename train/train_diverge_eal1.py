#!/usr/bin/env python3
"""Train the frozen natural transition reader for DIVERGE-EAL1."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import itertools
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from diverge_eal1_data import TRAIN_ROWS, TRAIN_SEED, validate_training_record
from diverge_eal1_runtime import (
    CHECKPOINT_SCHEMA,
    NaturalTransitionReader,
    TransitionReaderConfig,
    hard_role_permutation,
    module_state_sha256,
    sha256_path,
    tensorize_sources,
)


REPORT_SCHEMA = "shohin-diverge-eal1-training-report-v1"
UPDATES = 1_000
BATCH_SIZE = 256
LEARNING_RATE = 0.003
EVALUATION_ROWS = 10_000
_PERMUTATIONS = tuple(itertools.permutations(range(4)))
_PERMUTATION_LOOKUP = [-1] * 256
for _index, _permutation in enumerate(_PERMUTATIONS):
    _code = sum(value * scale for value, scale in zip(_permutation, (64, 16, 4, 1)))
    _PERMUTATION_LOOKUP[_code] = _index


def _load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError("EAL1 training data hash differs")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            validate_training_record(row)
            rows.append(row)
    if len(rows) != TRAIN_ROWS:
        raise RuntimeError("EAL1 training row count differs")
    return rows


def _view(record: dict[str, Any], counterfactual: bool) -> dict[str, Any]:
    if counterfactual:
        return {
            "source_text": record["counterfactual_text"],
            "numeric_role_ids": record["counterfactual_role_ids"],
        }
    return {
        "source_text": record["source_text"],
        "numeric_role_ids": record["numeric_role_ids"],
    }


def _permutation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    permutations = torch.tensor(_PERMUTATIONS, device=logits.device, dtype=torch.long)
    scores = torch.stack(
        [
            logits[:, torch.arange(4, device=logits.device), permutation].sum(dim=-1)
            for permutation in permutations
        ],
        dim=-1,
    )
    codes = (targets * targets.new_tensor((64, 16, 4, 1))).sum(dim=-1)
    lookup = targets.new_tensor(_PERMUTATION_LOOKUP)
    target_indices = lookup[codes]
    if torch.any(target_indices < 0):
        raise RuntimeError("EAL1 target is not a permutation")
    return F.cross_entropy(scores, target_indices)


@torch.no_grad()
def _evaluate_view(
    model: NaturalTransitionReader,
    rows: Sequence[dict[str, Any]],
    *,
    counterfactual: bool,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    complete = 0
    roles = 0
    total = 0
    for start in range(0, len(rows), batch_size):
        batch = [_view(row, counterfactual) for row in rows[start : start + batch_size]]
        byte_ids, attention, bounds, targets = tensorize_sources(batch, device)
        logits = model(byte_ids, attention, bounds)
        for index in range(len(batch)):
            predicted = hard_role_permutation(logits[index])
            gold = tuple(int(value) for value in targets[index])
            complete += predicted == gold
            roles += sum(
                left == right for left, right in zip(predicted, gold, strict=True)
            )
            total += 4
    return {
        "rows": len(rows),
        "complete_exact": complete,
        "complete_exact_rate": complete / max(1, len(rows)),
        "role_accuracy": roles / max(1, total),
    }


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--log-interval", type=int, default=25)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing EAL1 output: {args.output}")
    if (
        args.updates != UPDATES
        or args.batch_size != BATCH_SIZE
        or args.learning_rate != LEARNING_RATE
        or args.seed != TRAIN_SEED
    ):
        raise SystemExit("EAL1 frozen training schedule differs")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = _load_jsonl(args.data, args.data_sha256)
    model = NaturalTransitionReader(TransitionReaderConfig()).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01
    )
    generator = random.Random(args.seed ^ 0x45414C31)
    initial_state_sha256 = module_state_sha256(model)
    args.output.mkdir(parents=True)
    history = []
    charged_source_bytes = 0
    started = time.monotonic()
    for update in range(1, args.updates + 1):
        batch = [
            _view(
                rows[generator.randrange(len(rows))],
                counterfactual=bool(index % 2),
            )
            for index in range(args.batch_size)
        ]
        byte_ids, attention, bounds, targets = tensorize_sources(batch, device)
        progress = update / args.updates
        learning_rate = args.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
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
            loss = _permutation_loss(logits, targets)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        charged_source_bytes += int(attention.sum())
        if update == 1 or update % args.log_interval == 0 or update == args.updates:
            record = {
                "update": update,
                "learning_rate": learning_rate,
                "loss": float(loss.detach()),
                "raw_role_accuracy": float(
                    logits.detach().argmax(dim=-1).eq(targets).float().mean()
                ),
                "gradient_norm": float(gradient_norm),
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    elapsed = time.monotonic() - started
    model.eval()
    evaluation_rows = rows[:EVALUATION_ROWS]
    normal = _evaluate_view(
        model,
        evaluation_rows,
        counterfactual=False,
        device=device,
        batch_size=args.evaluation_batch_size,
    )
    counterfactual = _evaluate_view(
        model,
        evaluation_rows,
        counterfactual=True,
        device=device,
        batch_size=args.evaluation_batch_size,
    )
    final_state_sha256 = module_state_sha256(model)
    checkpoint_path = args.output / "checkpoint.pt"
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "source_commit": args.source_commit,
        "data_sha256": args.data_sha256,
        "seed": args.seed,
        "updates": args.updates,
        "config": asdict(model.config),
        "model_state": model.state_dict(),
        "model_state_sha256": final_state_sha256,
    }
    _atomic_torch(checkpoint_path, checkpoint)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "source_commit": args.source_commit,
        "data_sha256": args.data_sha256,
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "charged_examples": args.updates * args.batch_size,
        "charged_source_bytes": charged_source_bytes,
        "elapsed_seconds": elapsed,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha256,
        "model": model.record(),
        "training_sample": {"normal": normal, "counterfactual": counterfactual},
        "history": history,
        "checkpoint": str(checkpoint_path),
    }
    report["checkpoint_sha256"] = sha256_path(checkpoint_path)
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    os.chmod(checkpoint_path, 0o444)
    os.chmod(report_path, 0o444)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": report["checkpoint_sha256"],
                "normal": normal["complete_exact_rate"],
                "counterfactual": counterfactual["complete_exact_rate"],
                "report": str(report_path),
                "report_sha256": sha256_path(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
