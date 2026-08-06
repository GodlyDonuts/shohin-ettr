#!/usr/bin/env python3
"""Train the one frozen DIVERGE-TOL3 local semantic anchor head."""

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
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from diverge_tol1_product import load_rows, sha256_path
from diverge_tol3_semantic_anchor import (
    AnchorExample,
    COMPARATOR_NAMES,
    LocalSemanticAnchor,
    OPERATION_NAMES,
    TOL3Config,
    build_anchor_examples,
    module_state_sha256,
    tensorize_texts,
)


SCHEMA = "shohin-diverge-tol3-training-report-v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _examples_sha256(examples: Sequence[AnchorExample]) -> str:
    payload = [asdict(value) for value in examples]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _class_balanced_loss(
    operation_logits: torch.Tensor,
    comparator_logits: torch.Tensor,
    examples: Sequence[AnchorExample],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    labels = torch.tensor([value.label for value in examples], device=device)
    operation_mask = torch.tensor(
        [value.task == "operation" for value in examples],
        dtype=torch.bool,
        device=device,
    )
    comparator_mask = ~operation_mask
    class_losses = []
    task_losses: dict[str, torch.Tensor] = {}
    for task, logits, mask, width in (
        ("operation", operation_logits, operation_mask, len(OPERATION_NAMES)),
        ("comparator", comparator_logits, comparator_mask, len(COMPARATOR_NAMES)),
    ):
        local_labels = labels[mask]
        local_losses = F.cross_entropy(
            logits[mask], local_labels, reduction="none"
        )
        task_classes = []
        for label in range(width):
            selected = local_losses[local_labels.eq(label)]
            if not len(selected):
                raise RuntimeError(f"TOL3 {task} class is absent")
            class_loss = selected.mean()
            class_losses.append(class_loss)
            task_classes.append(class_loss)
        task_losses[task] = torch.stack(task_classes).mean()
    loss = torch.stack(class_losses).mean()
    metrics = {
        "loss": float(loss.detach()),
        "operation_loss": float(task_losses["operation"].detach()),
        "comparator_loss": float(task_losses["comparator"].detach()),
        "operation_accuracy": float(
            operation_logits[operation_mask]
            .argmax(dim=-1)
            .eq(labels[operation_mask])
            .float()
            .mean()
        ),
        "comparator_accuracy": float(
            comparator_logits[comparator_mask]
            .argmax(dim=-1)
            .eq(labels[comparator_mask])
            .float()
            .mean()
        ),
    }
    return loss, metrics


def _final_metrics(
    model: LocalSemanticAnchor,
    ids: torch.Tensor,
    mask: torch.Tensor,
    examples: Sequence[AnchorExample],
) -> dict[str, object]:
    model.eval()
    with torch.inference_mode():
        operation, comparator = model(ids, mask)
    counts: dict[str, Counter[str]] = {
        "operation": Counter(),
        "comparator": Counter(),
    }
    per_class: dict[str, Counter[str]] = {}
    for index, example in enumerate(examples):
        logits = operation[index] if example.task == "operation" else comparator[index]
        prediction = int(logits.argmax())
        local = counts[example.task]
        local["total"] += 1
        local["exact"] += int(prediction == example.label)
        names = OPERATION_NAMES if example.task == "operation" else COMPARATOR_NAMES
        label_name = names[example.label]
        class_counts = per_class.setdefault(
            f"{example.task}:{label_name}", Counter()
        )
        class_counts["total"] += 1
        class_counts["exact"] += int(prediction == example.label)
    return {
        "counts": {name: dict(values) for name, values in counts.items()},
        "per_class": {
            name: dict(values) for name, values in sorted(per_class.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=750)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026080505)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing TOL3 output: {args.output}")
    if args.updates != 750 or args.width != 64 or args.layers != 1:
        raise SystemExit("TOL3 frozen fit geometry or duration differs")
    if args.seed != 2026080505 or args.learning_rate != 3e-3:
        raise SystemExit("TOL3 frozen fit seed or learning rate differs")
    if sha256_path(args.source_checkpoint) != args.source_checkpoint_sha256:
        raise SystemExit("TOL3 source checkpoint hash differs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("TOL3 requested CUDA is unavailable")

    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    rows = load_rows(args.data, args.data_sha256, "train")
    examples = build_anchor_examples(rows)
    examples_sha256 = _examples_sha256(examples)
    ids, mask = tensorize_texts([value.text for value in examples], device)
    config = TOL3Config(width=args.width, layers=args.layers)
    model = LocalSemanticAnchor(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    initial_sha256 = module_state_sha256(model)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    history = []

    model.train()
    for update in range(1, args.updates + 1):
        progress = update / args.updates
        learning_rate = args.learning_rate * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        operation, comparator = model(ids, mask)
        loss, metrics = _class_balanced_loss(
            operation, comparator, examples, device
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise SystemExit("non-finite TOL3 gradient")
        optimizer.step()
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
    final_sha256 = module_state_sha256(model)
    final_metrics = _final_metrics(model, ids, mask, examples)
    checkpoint_path = args.output / "checkpoint_0000750.pt"
    _atomic_checkpoint(
        checkpoint_path,
        {
            "schema": SCHEMA,
            "seed": args.seed,
            "updates": args.updates,
            "config": asdict(config),
            "model_state": model.state_dict(),
            "model_state_sha256": final_sha256,
            "data_sha256": args.data_sha256,
            "examples_sha256": examples_sha256,
            "source_checkpoint_sha256": args.source_checkpoint_sha256,
        },
    )
    checkpoint_sha256 = sha256_path(checkpoint_path)
    report = {
        "schema": SCHEMA,
        "seed": args.seed,
        "updates": args.updates,
        "learning_rate": args.learning_rate,
        "train_rows": len(rows),
        "deduplicated_examples": len(examples),
        "examples_sha256": examples_sha256,
        "example_counts": {
            f"{task}:{label}": count
            for (task, label), count in sorted(
                Counter((value.task, value.label) for value in examples).items()
            )
        },
        "elapsed_seconds": elapsed,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        ),
        "device": str(device),
        "initial_model_sha256": initial_sha256,
        "final_model_sha256": final_sha256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "data": str(args.data),
        "data_sha256": args.data_sha256,
        "source_checkpoint": str(args.source_checkpoint),
        "source_checkpoint_sha256": args.source_checkpoint_sha256,
        "source_checkpoint_used_for_inference": False,
        "config": asdict(config),
        "history": history,
        "final_training_metrics": final_metrics,
    }
    _atomic_json(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "elapsed_seconds": elapsed,
                "final_training_metrics": final_metrics["counts"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
