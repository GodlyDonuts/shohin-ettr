#!/usr/bin/env python3
"""Train the one frozen integrated DIVERGE-IEM1 checkpoint."""

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

from diverge_iem1_data import (
    QUERY_TRAIN_ROWS,
    TRAIN_SEED,
    validate_query_training_record,
)
from diverge_iem1_runtime import (
    IEM1Config,
    IntegratedEpistemicMachine,
    load_nve1_state,
    module_state_sha256,
    tensorize_local_texts,
    tensorize_queries,
)
from diverge_nve1_data import TRAIN_ROWS, validate_training_record
from diverge_nve1_runtime import hard_role_permutation, tensorize_sources
from diverge_tol1_product import load_rows
from diverge_tol3_semantic_anchor import (
    AnchorExample,
    COMPARATOR_NAMES,
    OPERATION_NAMES,
    build_anchor_examples,
)


SCHEMA = "shohin-diverge-iem1-training-report-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("ascii"))
        digest.update(
            tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def _load_jsonl(
    path: Path,
    expected_sha256: str,
    *,
    expected_rows: int,
    validator,
) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError(f"IEM1 data hash differs: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            validator(row)
            rows.append(row)
    if len(rows) != expected_rows:
        raise RuntimeError(f"IEM1 data row count differs: {path}")
    return rows


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _local_loss(
    operation: torch.Tensor,
    comparator: torch.Tensor,
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
    task_accuracy = {}
    for name, logits, mask, width in (
        ("operation", operation, operation_mask, len(OPERATION_NAMES)),
        ("comparator", comparator, comparator_mask, len(COMPARATOR_NAMES)),
    ):
        task_labels = labels[mask]
        losses = F.cross_entropy(logits[mask], task_labels, reduction="none")
        for label in range(width):
            selected = losses[task_labels.eq(label)]
            if not len(selected):
                raise RuntimeError(f"IEM1 local {name} class is absent")
            class_losses.append(selected.mean())
        task_accuracy[name] = float(
            logits[mask].argmax(-1).eq(task_labels).float().mean().detach()
        )
    return torch.stack(class_losses).mean(), task_accuracy


@torch.no_grad()
def _evaluate_evidence(
    model: IntegratedEpistemicMachine,
    rows: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, int]:
    counts = Counter()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        tensors = tensorize_sources(batch, device)
        numeric, symbols = model.forward_evidence(*tensors[:4])
        for index in range(len(batch)):
            numeric_prediction = hard_role_permutation(numeric[index])
            symbol_prediction = hard_role_permutation(symbols[index])
            numeric_gold = tuple(int(value) for value in tensors[4][index])
            symbol_gold = tuple(int(value) for value in tensors[5][index])
            counts["numeric_exact"] += numeric_prediction == numeric_gold
            counts["symbol_exact"] += symbol_prediction == symbol_gold
            counts["joint_exact"] += (
                numeric_prediction == numeric_gold and symbol_prediction == symbol_gold
            )
    counts["rows"] = len(rows)
    return dict(counts)


@torch.no_grad()
def _evaluate_queries(
    model: IntegratedEpistemicMachine,
    rows: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, int]:
    exact = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        tensors = tensorize_queries(batch, device)
        logits = model.forward_query(*tensors[:3])
        exact += sum(
            hard_role_permutation(logits[index])
            == tuple(int(value) for value in tensors[3][index])
            for index in range(len(batch))
        )
    return {"rows": len(rows), "exact": exact}


@torch.no_grad()
def _evaluate_local(
    model: IntegratedEpistemicMachine,
    examples: Sequence[AnchorExample],
    *,
    device: torch.device,
) -> dict[str, int]:
    ids, mask = tensorize_local_texts([value.text for value in examples], device)
    operation, comparator = model.forward_local(ids, mask, hard_transport=True)
    counts = Counter()
    for index, example in enumerate(examples):
        logits = operation[index] if example.task == "operation" else comparator[index]
        counts[f"{example.task}_total"] += 1
        counts[f"{example.task}_exact"] += int(logits.argmax() == example.label)
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tol1-data", type=Path, required=True)
    parser.add_argument("--tol1-data-sha256", required=True)
    parser.add_argument("--evidence-data", type=Path, required=True)
    parser.add_argument("--evidence-data-sha256", required=True)
    parser.add_argument("--query-data", type=Path, required=True)
    parser.add_argument("--query-data-sha256", required=True)
    parser.add_argument("--nve1-checkpoint", type=Path, required=True)
    parser.add_argument("--nve1-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing IEM1 output: {args.output}")
    if (
        args.updates != 1000
        or args.batch_size != 256
        or args.learning_rate != 1e-3
        or args.seed != TRAIN_SEED
    ):
        raise SystemExit("IEM1 frozen training schedule differs")
    if sha256_path(args.nve1_checkpoint) != args.nve1_checkpoint_sha256:
        raise SystemExit("IEM1 NVE1 checkpoint hash differs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("IEM1 requested CUDA is unavailable")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "cuda"
        or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    tol1_rows = load_rows(args.tol1_data, args.tol1_data_sha256, "train")
    local_examples = build_anchor_examples(tol1_rows)
    evidence_rows = _load_jsonl(
        args.evidence_data,
        args.evidence_data_sha256,
        expected_rows=TRAIN_ROWS,
        validator=validate_training_record,
    )
    query_rows = _load_jsonl(
        args.query_data,
        args.query_data_sha256,
        expected_rows=QUERY_TRAIN_ROWS,
        validator=validate_query_training_record,
    )
    nve1 = torch.load(args.nve1_checkpoint, map_location=device, weights_only=False)
    if nve1.get("schema") != "shohin-diverge-nve1-training-report-v1":
        raise SystemExit("IEM1 NVE1 checkpoint schema differs")
    if _state_sha256(nve1["model_state"]) != nve1["model_state_sha256"]:
        raise SystemExit("IEM1 NVE1 embedded model state differs")

    config = IEM1Config()
    model = IntegratedEpistemicMachine(config).to(device)
    load_nve1_state(model, nve1["model_state"])
    initial_sha256 = module_state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    local_ids, local_mask = tensorize_local_texts(
        [value.text for value in local_examples], device
    )
    rng = random.Random(args.seed)
    args.output.mkdir(parents=True)
    history = []
    source_bytes_seen = 0
    started = time.monotonic()
    peak_allocated = 0

    for update in range(1, args.updates + 1):
        evidence_batch = [
            evidence_rows[rng.randrange(len(evidence_rows))]
            for _ in range(args.batch_size)
        ]
        query_batch = [
            query_rows[rng.randrange(len(query_rows))] for _ in range(args.batch_size)
        ]
        evidence_tensors = tensorize_sources(evidence_batch, device)
        query_tensors = tensorize_queries(query_batch, device)
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
            operation, comparator = model.forward_local(local_ids, local_mask)
            local_loss, local_accuracy = _local_loss(
                operation,
                comparator,
                local_examples,
                device,
            )
            numeric, symbols = model.forward_evidence(*evidence_tensors[:4])
            numeric_loss = F.cross_entropy(
                numeric.reshape(-1, 2), evidence_tensors[4].reshape(-1)
            )
            symbol_loss = F.cross_entropy(
                symbols.reshape(-1, 2), evidence_tensors[5].reshape(-1)
            )
            evidence_loss = 0.5 * (numeric_loss + symbol_loss)
            query_logits = model.forward_query(*query_tensors[:3])
            query_loss = F.cross_entropy(
                query_logits.reshape(-1, 2), query_tensors[3].reshape(-1)
            )
            transport_penalty = model.transport_penalty()
            loss = (
                local_loss + evidence_loss + query_loss
            ) / 3.0 + 0.01 * transport_penalty
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise SystemExit("non-finite IEM1 gradient")
        optimizer.step()
        source_bytes_seen += int(local_mask.sum())
        source_bytes_seen += int(evidence_tensors[1].sum())
        source_bytes_seen += int(query_tensors[1].sum())
        if device.type == "cuda":
            peak_allocated = max(
                peak_allocated, torch.cuda.max_memory_allocated(device)
            )
        if update == 1 or update % args.log_interval == 0 or update == args.updates:
            record = {
                "update": update,
                "learning_rate": learning_rate,
                "loss": float(loss.detach()),
                "local_loss": float(local_loss.detach()),
                "evidence_loss": float(evidence_loss.detach()),
                "query_loss": float(query_loss.detach()),
                "transport_penalty": float(transport_penalty.detach()),
                "operation_accuracy": local_accuracy["operation"],
                "comparator_accuracy": local_accuracy["comparator"],
                "numeric_role_accuracy": float(
                    numeric.detach().argmax(-1).eq(evidence_tensors[4]).float().mean()
                ),
                "symbol_role_accuracy": float(
                    symbols.detach().argmax(-1).eq(evidence_tensors[5]).float().mean()
                ),
                "query_role_accuracy": float(
                    query_logits.detach().argmax(-1).eq(query_tensors[3]).float().mean()
                ),
                "gradient_norm": float(gradient_norm),
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    elapsed = time.monotonic() - started
    model.eval()
    local_evaluation = _evaluate_local(model, local_examples, device=device)
    evidence_evaluation = _evaluate_evidence(
        model,
        evidence_rows,
        device=device,
        batch_size=512,
    )
    query_evaluation = _evaluate_queries(
        model,
        query_rows,
        device=device,
        batch_size=512,
    )
    final_sha256 = module_state_sha256(model)
    checkpoint_path = args.output / "checkpoint_0001000.pt"
    _atomic_checkpoint(
        checkpoint_path,
        {
            "schema": SCHEMA,
            "seed": args.seed,
            "update": args.updates,
            "config": asdict(config),
            "model_state": model.state_dict(),
            "model_state_sha256": final_sha256,
            "nve1_checkpoint_sha256": args.nve1_checkpoint_sha256,
            "tol1_data_sha256": args.tol1_data_sha256,
            "evidence_data_sha256": args.evidence_data_sha256,
            "query_data_sha256": args.query_data_sha256,
        },
    )
    checkpoint_sha256 = sha256_path(checkpoint_path)
    report = {
        "schema": SCHEMA,
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "config": asdict(config),
        "device": str(device),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "nve1_checkpoint_sha256": args.nve1_checkpoint_sha256,
        "tol1_data_sha256": args.tol1_data_sha256,
        "evidence_data_sha256": args.evidence_data_sha256,
        "query_data_sha256": args.query_data_sha256,
        "local_examples": len(local_examples),
        "evidence_rows": len(evidence_rows),
        "query_rows": len(query_rows),
        "source_bytes_seen": source_bytes_seen,
        "source_bytes_per_second": source_bytes_seen / max(elapsed, 1e-9),
        "elapsed_seconds": elapsed,
        "peak_allocated_bytes": peak_allocated,
        "initial_model_sha256": initial_sha256,
        "final_model_sha256": final_sha256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "local_evaluation": local_evaluation,
        "evidence_evaluation": evidence_evaluation,
        "query_evaluation": query_evaluation,
        "history": history,
    }
    _atomic_json(args.output / "report.json", report)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "model_state_sha256": final_sha256,
                "local_evaluation": local_evaluation,
                "evidence_evaluation": evidence_evaluation,
                "query_evaluation": query_evaluation,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
