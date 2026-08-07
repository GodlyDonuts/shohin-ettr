#!/usr/bin/env python3
"""Train one matched DIVERGE-NLS1 law-synthesis arm."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import torch

from diverge_eal1_runtime import module_state_sha256, sha256_path
from diverge_nls1_data import TRAIN_SEED, validate_training_record
from diverge_nls1_runtime import (
    CHECKPOINT_SCHEMA,
    NeuralLawSynthesizer,
    NeuralLawSynthesizerConfig,
)


REPORT_SCHEMA = "shohin-diverge-nls1-training-report-v1"
UPDATES = 500
BATCH_SIZE = 2_048
LEARNING_RATE = 0.003
Arm = str


def _load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError("NLS1 training data hash differs")
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
        raise RuntimeError("NLS1 temporary checkpoint already exists")
    torch.save(dict(payload), temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _arm_inputs(values: torch.Tensor, arm: Arm) -> torch.Tensor:
    if arm == "treatment":
        return values
    if arm == "shuffled_outcomes":
        output = values.clone()
        output[:, :, 2:] = torch.roll(output[:, :, 2:], shifts=1, dims=0)
        return output
    raise RuntimeError("NLS1 training arm differs")


@torch.no_grad()
def _sample_accuracy(
    model: NeuralLawSynthesizer,
    values: torch.Tensor,
    targets: torch.Tensor,
    arm: Arm,
) -> dict[str, float | int]:
    sample = min(10_000, values.shape[0])
    predictions = model(
        _arm_inputs(values[:sample], arm),
        torch.ones((sample, 3), dtype=torch.bool, device=values.device),
    ).argmax(dim=-1)
    row_exact = int(predictions.eq(targets[:sample]).sum())
    matrix_exact = int(predictions.eq(targets[:sample]).all(dim=1).sum())
    return {
        "rows": sample * 2,
        "row_exact": row_exact,
        "row_rate": row_exact / (sample * 2),
        "matrices": sample,
        "matrix_exact": matrix_exact,
        "matrix_rate": matrix_exact / sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--arm", choices=("treatment", "shuffled_outcomes"), required=True
    )
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing NLS1 output: {args.output}")
    if (
        args.updates != UPDATES
        or args.batch_size != BATCH_SIZE
        or not math.isclose(args.learning_rate, LEARNING_RATE)
    ):
        raise SystemExit("NLS1 frozen training schedule differs")
    args.output.mkdir(parents=True)
    rows = _load_jsonl(args.data, args.data_sha256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    values = torch.tensor(
        [row["demonstrations"] for row in rows], dtype=torch.long, device=device
    )
    targets = torch.tensor(
        [row["target_row_ids"] for row in rows], dtype=torch.long, device=device
    )
    mask = torch.ones((args.batch_size, 3), dtype=torch.bool, device=device)

    torch.manual_seed(TRAIN_SEED)
    random.seed(TRAIN_SEED)
    model = NeuralLawSynthesizer().to(device)
    initial_state_sha256 = module_state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    generator = torch.Generator(device=device).manual_seed(TRAIN_SEED + 1)
    history = []
    charged = 0
    started = time.perf_counter()
    for update in range(1, args.updates + 1):
        indices = torch.randint(
            values.shape[0],
            (args.batch_size,),
            generator=generator,
            device=device,
        )
        batch_values = _arm_inputs(values[indices], args.arm)
        batch_targets = targets[indices]
        logits = model(batch_values, mask)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), batch_targets.flatten()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        if not torch.isfinite(loss) or not math.isfinite(gradient_norm):
            raise RuntimeError("NLS1 training became nonfinite")
        optimizer.step()
        charged += args.batch_size
        if update in (1, 2, 4, 8, 16, 32, 64, 128, 256, args.updates):
            history.append(
                {
                    "update": update,
                    "loss": float(loss.detach()),
                    "gradient_norm": gradient_norm,
                }
            )
    elapsed = time.perf_counter() - started
    model.eval()
    sample = _sample_accuracy(model, values, targets, args.arm)
    final_state_sha256 = module_state_sha256(model)
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "source_commit": args.source_commit,
        "arm": args.arm,
        "config": asdict(NeuralLawSynthesizerConfig()),
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
                "matrix_rate": sample["matrix_rate"],
                "output": str(args.output),
                "report_sha256": sha256_path(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
