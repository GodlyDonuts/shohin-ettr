#!/usr/bin/env python3
"""Train the frozen BTT1 raw byte-tape compiler."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Sequence

import torch

from byte_tape_compiler import ByteProgram, ByteTapeCompiler, byte_batch, byte_loss, load_byte_program


SCHEMA = "shohin-btt1-training-v1"


class BTT1TrainingError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_programs(path: Path, expected_sha256: str, expected_rows: int) -> list[ByteProgram]:
    if sha256_file(path) != expected_sha256:
        raise BTT1TrainingError("data SHA-256 differs")
    rows = [load_byte_program(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != expected_rows:
        raise BTT1TrainingError("data population differs")
    return rows


def shuffled(programs: Sequence[ByteProgram], seed: int) -> list[ByteProgram]:
    result = list(programs)
    random.Random(seed).shuffle(result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise BTT1TrainingError("refusing existing output")
    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    model = ByteTapeCompiler(width=args.width, encoder_layers=args.encoder_layers, heads=args.heads).to(device=device, dtype=torch.bfloat16)
    trainable = model.parameter_count()
    if trainable >= 10_000_000:
        raise BTT1TrainingError("parameter receipt differs")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.01, fused=True)
    programs = shuffled(load_programs(args.data, args.expected_data_sha256, 75935), args.data_seed)
    cursor = charged_examples = charged_bytes = maximum_bytes = 0
    losses = []
    started = time.time()
    model.train()
    for update in range(1, args.updates + 1):
        if cursor + args.batch_size > len(programs):
            programs = shuffled(programs, args.data_seed + update)
            cursor = 0
        rows = programs[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        batch = byte_batch(rows, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch["byte_ids"], batch["mask"])
            loss = byte_loss(output, batch["role"])
        if not torch.isfinite(loss):
            raise BTT1TrainingError("nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise BTT1TrainingError("nonfinite gradient")
        optimizer.step()
        lengths = batch["mask"].sum(1)
        charged_examples += len(rows)
        charged_bytes += int(lengths.sum())
        maximum_bytes = max(maximum_bytes, int(lengths.max()))
        losses.append(float(loss.detach()))
        if update == 1 or update % args.log_interval == 0 or update == args.updates:
            print(f"update={update}/{args.updates} loss={losses[-1]:.6f} gnorm={float(gradient_norm):.4f}", flush=True)
    elapsed = time.time() - started
    checkpoint = args.output / "compiler.pt"
    temporary = args.output / f".compiler.pt.tmp.{os.getpid()}"
    torch.save(
        {"schema": SCHEMA, "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
         "config": {"width": args.width, "encoder_layers": args.encoder_layers, "heads": args.heads},
         "updates": args.updates, "data_sha256": args.expected_data_sha256},
        temporary,
    )
    os.replace(temporary, checkpoint)
    report = {
        "schema": SCHEMA, "status": "complete", "holdout_used": False,
        "data": str(args.data.resolve()), "data_sha256": args.expected_data_sha256,
        "updates": args.updates, "batch_size": args.batch_size, "charged_examples": charged_examples,
        "charged_bytes": charged_bytes, "maximum_bytes": maximum_bytes,
        "learning_rate": args.learning_rate, "trainable_parameters": trainable,
        "elapsed_seconds": elapsed, "examples_per_second": charged_examples / elapsed,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "loss_first": losses[0], "loss_last": losses[-1],
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
    }
    atomic_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026081061)
    parser.add_argument("--data-seed", type=int, default=2026081062)
    parser.add_argument("--log-interval", type=int, default=10)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
