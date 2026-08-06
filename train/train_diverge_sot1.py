#!/usr/bin/env python3
"""Train the one isolated SOT1 QUERY owner and serialize all stage owners."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from diverge_iem1_data import TRAIN_SEED as IEM1_TRAIN_SEED
from diverge_iem1_data import validate_query_training_record
from diverge_iem1_runtime import tensorize_queries
from diverge_nve1_runtime import hard_role_permutation
from diverge_sot1_runtime import (
    SOT1Config,
    StageOwnedEpistemicMachine,
    module_state_sha256,
    query_owner_parameters,
    validate_owner_isolation,
)


SCHEMA = "shohin-diverge-sot1-training-report-v1"
TRAIN_SEED = 2026080616


class SOT1TrainingError(RuntimeError):
    """The frozen SOT1 fit contract was violated."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_jsonl(
    path: Path,
    expected_sha256: str,
    validator: Callable[[Mapping[str, Any]], None],
) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise SOT1TrainingError("SOT1 query data hash differs")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            validator(row)
            rows.append(row)
    if len(rows) != 50_000:
        raise SOT1TrainingError("SOT1 query row count differs")
    return rows


def _load_owner_checkpoint(
    path: Path,
    expected_sha256: str,
    expected_schema: str,
    device: torch.device,
) -> dict[str, Any]:
    if sha256_path(path) != expected_sha256:
        raise SOT1TrainingError("SOT1 owner checkpoint hash differs")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != expected_schema:
        raise SOT1TrainingError("SOT1 owner checkpoint schema differs")
    return checkpoint


@torch.no_grad()
def _evaluate_query_owner(
    model: StageOwnedEpistemicMachine,
    rows: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, int]:
    model.eval()
    exact = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        ids, mask, groups, targets = tensorize_queries(batch, device)
        logits = model.forward_query(ids, mask, groups)
        for index in range(len(batch)):
            exact += hard_role_permutation(logits[index]) == tuple(
                int(value) for value in targets[index].tolist()
            )
    return {"rows": len(rows), "exact": exact}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-data", type=Path, required=True)
    parser.add_argument("--query-data-sha256", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--evidence-checkpoint", type=Path, required=True)
    parser.add_argument("--evidence-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing SOT1 output: {args.output}")
    if (
        args.updates != 1000
        or args.batch_size != 256
        or args.learning_rate != 3e-3
        or args.seed != TRAIN_SEED
    ):
        raise SystemExit("SOT1 frozen training schedule differs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("SOT1 requested CUDA is unavailable")
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    query_rows = _load_jsonl(
        args.query_data,
        args.query_data_sha256,
        validate_query_training_record,
    )
    source_checkpoint = _load_owner_checkpoint(
        args.source_checkpoint,
        args.source_checkpoint_sha256,
        "shohin-diverge-tol3-training-report-v1",
        device,
    )
    evidence_checkpoint = _load_owner_checkpoint(
        args.evidence_checkpoint,
        args.evidence_checkpoint_sha256,
        "shohin-diverge-nve1-training-report-v1",
        device,
    )
    if int(source_checkpoint.get("updates", -1)) != 750:
        raise SOT1TrainingError("SOT1 source owner duration differs")
    if int(evidence_checkpoint.get("update", -1)) != 1000:
        raise SOT1TrainingError("SOT1 evidence owner duration differs")

    model = StageOwnedEpistemicMachine(SOT1Config()).to(device)
    model.source_owner.load_state_dict(source_checkpoint["model_state"], strict=True)
    model.evidence_owner.load_state_dict(
        evidence_checkpoint["model_state"], strict=True
    )
    model.freeze_qualified_owners()
    validate_owner_isolation(model)
    initial_owner_hashes = model.owner_hashes()
    if (
        initial_owner_hashes["WORLD"] != source_checkpoint["model_state_sha256"]
        or initial_owner_hashes["EVIDENCE"] != evidence_checkpoint["model_state_sha256"]
    ):
        raise SOT1TrainingError("SOT1 qualified owner state differs")
    initial_model_sha256 = module_state_sha256(model)
    parameters = query_owner_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    rng = random.Random(args.seed ^ IEM1_TRAIN_SEED)
    history = []
    source_bytes_seen = 0
    started = time.monotonic()
    model.train()
    for update in range(1, args.updates + 1):
        progress = update / args.updates
        learning_rate = args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        batch = [query_rows[rng.randrange(len(query_rows))] for _ in range(256)]
        ids, mask, groups, targets = tensorize_queries(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_query(ids, mask, groups)
        loss = F.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(gradient_norm):
            raise SystemExit("non-finite SOT1 query gradient")
        optimizer.step()
        source_bytes_seen += int(mask.sum())
        if update == 1 or update % args.log_interval == 0 or update == args.updates:
            record = {
                "update": update,
                "learning_rate": learning_rate,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                "query_role_accuracy": float(
                    logits.detach().argmax(-1).eq(targets).float().mean()
                ),
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    elapsed = time.monotonic() - started
    final_owner_hashes = model.owner_hashes()
    if final_owner_hashes["WORLD"] != initial_owner_hashes["WORLD"]:
        raise SystemExit("SOT1 WORLD owner changed during QUERY training")
    if final_owner_hashes["EVIDENCE"] != initial_owner_hashes["EVIDENCE"]:
        raise SystemExit("SOT1 EVIDENCE owner changed during QUERY training")
    query_evaluation = _evaluate_query_owner(
        model, query_rows, device=device, batch_size=512
    )
    manifest = model.owner_manifest()
    final_model_sha256 = module_state_sha256(model)
    args.output.mkdir(parents=True)
    checkpoint_path = args.output / "checkpoint_0001000.pt"
    _atomic_checkpoint(
        checkpoint_path,
        {
            "schema": SCHEMA,
            "seed": args.seed,
            "update": args.updates,
            "config": {"query_width": 192, "query_layers": 2, "query_max_bytes": 192},
            "model_state": model.state_dict(),
            "model_state_sha256": final_model_sha256,
            "initial_owner_hashes": initial_owner_hashes,
            "final_owner_hashes": final_owner_hashes,
            "owner_manifest": manifest,
            "source_checkpoint_sha256": args.source_checkpoint_sha256,
            "evidence_checkpoint_sha256": args.evidence_checkpoint_sha256,
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
        "device": str(device),
        "elapsed_seconds": elapsed,
        "query_trainable_parameters": sum(value.numel() for value in parameters),
        "composite_parameters": sum(value.numel() for value in model.parameters()),
        "source_bytes_seen": source_bytes_seen,
        "source_bytes_per_second": source_bytes_seen / max(elapsed, 1e-9),
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
        ),
        "query_rows": len(query_rows),
        "query_data": str(args.query_data),
        "query_data_sha256": args.query_data_sha256,
        "source_checkpoint": str(args.source_checkpoint),
        "source_checkpoint_sha256": args.source_checkpoint_sha256,
        "evidence_checkpoint": str(args.evidence_checkpoint),
        "evidence_checkpoint_sha256": args.evidence_checkpoint_sha256,
        "initial_model_sha256": initial_model_sha256,
        "final_model_sha256": final_model_sha256,
        "initial_owner_hashes": initial_owner_hashes,
        "final_owner_hashes": final_owner_hashes,
        "owner_manifest": manifest,
        "query_evaluation": query_evaluation,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
    }
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "model_state_sha256": final_model_sha256,
                "query_evaluation": query_evaluation,
                "report": str(report_path),
                "report_sha256": sha256_path(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
