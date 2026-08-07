#!/usr/bin/env python3
"""Train one matched DIVERGE-NCP1 command-pointer arm."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import torch

from diverge_eal1_runtime import module_state_sha256, sha256_path
from diverge_ncp1_data import TRAIN_SEED, validate_training_record
from diverge_ncp1_runtime import (
    BLANK_ID,
    CHECKPOINT_SCHEMA,
    CommandPointerConfig,
    NaturalCommandPointer,
    greedy_ctc_decode,
    tensorize_commands,
)


REPORT_SCHEMA = "shohin-diverge-ncp1-training-report-v1"
UPDATES = 1_500
BATCH_SIZE = 128
LEARNING_RATE = 0.001


def _load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError("NCP1 training data hash differs")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    for row in rows:
        validate_training_record(row)
    return rows


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise RuntimeError("NCP1 temporary checkpoint already exists")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _targets(
    rows: Sequence[Mapping[str, Any]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor(
        [len(row["targets"]) for row in rows], dtype=torch.long, device=device
    )
    values = torch.tensor(
        [value for row in rows for value in row["targets"]],
        dtype=torch.long,
        device=device,
    )
    return values, lengths


def _frame_targets(
    rows: Sequence[Mapping[str, Any]], width: int, device: torch.device
) -> torch.Tensor:
    targets = torch.full((len(rows), width), BLANK_ID, dtype=torch.long, device=device)
    for row_index, row in enumerate(rows):
        for target, span in zip(row["targets"], row["alignment_spans"], strict=True):
            start, end = (int(value) for value in span)
            targets[row_index, start:end] = int(target)
    return targets


@torch.no_grad()
def _sample_accuracy(
    model: NaturalCommandPointer,
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    rotate_alias_table: bool,
) -> dict[str, float | int]:
    sample = rows[: min(2_048, len(rows))]
    predicted: list[tuple[int, ...]] = []
    for start in range(0, len(sample), 64):
        batch = sample[start : start + 64]
        command_ids, command_mask, alias_ids, alias_mask, lengths = tensorize_commands(
            batch, device, rotate_alias_table=rotate_alias_table
        )
        predicted.extend(
            greedy_ctc_decode(
                model(command_ids, command_mask, alias_ids, alias_mask), lengths
            )
        )
    gold = [tuple(int(value) for value in row["targets"]) for row in sample]
    exact = sum(left == right for left, right in zip(predicted, gold, strict=True))
    return {"exact": exact, "total": len(gold), "rate": exact / len(gold)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--arm", choices=("treatment", "shuffled_table"), required=True)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing NCP1 output: {args.output}")
    if (
        args.updates != UPDATES
        or args.batch_size != BATCH_SIZE
        or not math.isclose(args.learning_rate, LEARNING_RATE)
    ):
        raise SystemExit("NCP1 frozen training schedule differs")
    args.output.mkdir(parents=True)
    rows = _load_jsonl(args.data, args.data_sha256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rotate = args.arm == "shuffled_table"

    torch.manual_seed(TRAIN_SEED)
    random.seed(TRAIN_SEED)
    model = NaturalCommandPointer().to(device)
    initial_state_sha256 = module_state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    generator = torch.Generator().manual_seed(TRAIN_SEED + 1)
    history = []
    charged = 0
    started = time.perf_counter()
    for update in range(1, args.updates + 1):
        indices = torch.randint(
            len(rows), (args.batch_size,), generator=generator
        ).tolist()
        batch = [rows[index] for index in indices]
        command_ids, command_mask, alias_ids, alias_mask, input_lengths = (
            tensorize_commands(batch, device, rotate_alias_table=rotate)
        )
        target_values, target_lengths = _targets(batch, device)
        logits = model(command_ids, command_mask, alias_ids, alias_mask)
        ctc_loss = torch.nn.functional.ctc_loss(
            logits.log_softmax(dim=-1).transpose(0, 1),
            target_values,
            input_lengths,
            target_lengths,
            blank=BLANK_ID,
            reduction="mean",
            zero_infinity=True,
        )
        frame_targets = _frame_targets(batch, logits.shape[1], device)
        frame_loss = torch.nn.functional.cross_entropy(
            logits[command_mask], frame_targets[command_mask]
        )
        loss = frame_loss + 0.1 * ctc_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        if not torch.isfinite(loss) or not math.isfinite(gradient_norm):
            raise RuntimeError("NCP1 training became nonfinite")
        optimizer.step()
        charged += args.batch_size
        if update in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, args.updates):
            history.append(
                {
                    "update": update,
                    "loss": float(loss.detach()),
                    "frame_loss": float(frame_loss.detach()),
                    "ctc_loss": float(ctc_loss.detach()),
                    "gradient_norm": gradient_norm,
                }
            )
    elapsed = time.perf_counter() - started
    model.eval()
    sample = _sample_accuracy(model, rows, device, rotate_alias_table=rotate)
    final_state_sha256 = module_state_sha256(model)
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "source_commit": args.source_commit,
        "arm": args.arm,
        "config": asdict(CommandPointerConfig()),
        "model_state": model.state_dict(),
        "model_state_sha256": final_state_sha256,
        "initial_state_sha256": initial_state_sha256,
        "data_sha256": args.data_sha256,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }
    checkpoint_path = args.output / "checkpoint.pt"
    _atomic_torch(checkpoint_path, checkpoint)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "source_commit": args.source_commit,
        "arm": args.arm,
        "model": model.record(),
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha256,
        "data": str(args.data),
        "data_sha256": args.data_sha256,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "charged_examples": charged,
        "learning_rate": args.learning_rate,
        "elapsed_seconds": elapsed,
        "examples_per_second": charged / elapsed,
        "history": history,
        "training_sample": sample,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_path(checkpoint_path),
    }
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    os.chmod(checkpoint_path, 0o444)
    os.chmod(report_path, 0o444)
    print(
        json.dumps(
            {
                "arm": args.arm,
                "checkpoint_sha256": report["checkpoint_sha256"],
                "sample_rate": sample["rate"],
                "output": str(args.output),
                "report_sha256": sha256_path(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
