#!/usr/bin/env python3
"""Train the one shared semantic REFERENT owner for DIVERGE-SRP1."""

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

from diverge_iem1_data import validate_query_training_record
from diverge_iem1_runtime import tensorize_queries
from diverge_nve1_data import validate_training_record
from diverge_nve1_runtime import hard_role_permutation, tensorize_sources
from diverge_srp1_runtime import (
    SRP1Config,
    SemanticPrimitiveEpistemicMachine,
    referent_parameters,
    validate_owner_contract,
    warm_start_referent,
)
from diverge_iem1_runtime import module_state_sha256


SCHEMA = "shohin-diverge-srp1-training-report-v1"
TRAIN_SEED = 2026080621


class SRP1TrainingError(RuntimeError):
    """The frozen SRP1 training contract was violated."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(
    path: Path,
    expected_sha256: str,
    validator: Callable[[Mapping[str, Any]], None],
) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise SRP1TrainingError(f"protected data hash differs: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            validator(row)
            rows.append(row)
    if len(rows) != 50_000:
        raise SRP1TrainingError(f"protected data row count differs: {path}")
    return rows


def _load_checkpoint(
    path: Path,
    expected_sha256: str,
    expected_schema: str,
    device: torch.device,
) -> dict[str, Any]:
    if sha256_path(path) != expected_sha256:
        raise SRP1TrainingError(f"protected checkpoint hash differs: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != expected_schema:
        raise SRP1TrainingError(f"protected checkpoint schema differs: {path}")
    return checkpoint


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


@torch.no_grad()
def _evaluate(
    model: SemanticPrimitiveEpistemicMachine,
    rows: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    exact = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        if stage == "EVIDENCE":
            ids, mask, _, symbols, _, targets = tensorize_sources(batch, device)
        elif stage == "QUERY":
            ids, mask, symbols, targets = tensorize_queries(batch, device)
        else:
            raise SRP1TrainingError(f"unknown SRP1 stage: {stage}")
        logits = model.referent_owner(ids, mask, symbols)
        for index in range(len(batch)):
            exact += hard_role_permutation(logits[index]) == tuple(
                int(value) for value in targets[index].tolist()
            )
    return {"rows": len(rows), "exact": exact, "exact_rate": exact / len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-data", type=Path, required=True)
    parser.add_argument("--evidence-data-sha256", required=True)
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
        raise SystemExit(f"refusing existing SRP1 output: {args.output}")
    if (
        args.updates != 1000
        or args.batch_size != 256
        or args.learning_rate != 3e-3
        or args.seed != TRAIN_SEED
    ):
        raise SystemExit("SRP1 frozen training schedule differs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("SRP1 requested unavailable CUDA")
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    evidence_rows = _load_jsonl(
        args.evidence_data,
        args.evidence_data_sha256,
        validate_training_record,
    )
    query_rows = _load_jsonl(
        args.query_data,
        args.query_data_sha256,
        validate_query_training_record,
    )
    source_checkpoint = _load_checkpoint(
        args.source_checkpoint,
        args.source_checkpoint_sha256,
        "shohin-diverge-tol3-training-report-v1",
        device,
    )
    evidence_checkpoint = _load_checkpoint(
        args.evidence_checkpoint,
        args.evidence_checkpoint_sha256,
        "shohin-diverge-nve1-training-report-v1",
        device,
    )

    model = SemanticPrimitiveEpistemicMachine(SRP1Config()).to(device)
    model.source_owner.load_state_dict(source_checkpoint["model_state"], strict=True)
    model.numeric_evidence_owner.load_state_dict(
        evidence_checkpoint["model_state"], strict=True
    )
    warm_start_referent(model.referent_owner, evidence_checkpoint["model_state"])
    model.freeze_qualified_owners()
    validate_owner_contract(model)
    initial_owner_hashes = model.owner_hashes()
    if initial_owner_hashes["WORLD"] != source_checkpoint["model_state_sha256"]:
        raise SRP1TrainingError("SRP1 WORLD warm start differs")
    if (
        initial_owner_hashes["NUMERIC_EVIDENCE"]
        != evidence_checkpoint["model_state_sha256"]
    ):
        raise SRP1TrainingError("SRP1 numeric-evidence warm start differs")

    parameters = referent_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    rng = random.Random(args.seed ^ 0x53525031)
    history = []
    source_bytes_seen = 0
    started = time.monotonic()
    model.train()
    half_batch = args.batch_size // 2
    for update in range(1, args.updates + 1):
        progress = update / args.updates
        learning_rate = args.learning_rate * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        evidence_batch = [
            evidence_rows[rng.randrange(len(evidence_rows))]
            for _ in range(half_batch)
        ]
        query_batch = [
            query_rows[rng.randrange(len(query_rows))] for _ in range(half_batch)
        ]
        e_ids, e_mask, _, e_symbols, _, e_targets = tensorize_sources(
            evidence_batch, device
        )
        q_ids, q_mask, q_symbols, q_targets = tensorize_queries(query_batch, device)
        ids = torch.cat((e_ids, q_ids), dim=0)
        mask = torch.cat((e_mask, q_mask), dim=0)
        symbols = torch.cat((e_symbols, q_symbols), dim=0)
        targets = torch.cat((e_targets, q_targets), dim=0)

        optimizer.zero_grad(set_to_none=True)
        logits = model.referent_owner(ids, mask, symbols)
        evidence_loss = F.cross_entropy(
            logits[:half_batch].reshape(-1, 2),
            targets[:half_batch].reshape(-1),
        )
        query_loss = F.cross_entropy(
            logits[half_batch:].reshape(-1, 2),
            targets[half_batch:].reshape(-1),
        )
        loss = 0.5 * (evidence_loss + query_loss)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(gradient_norm):
            raise SystemExit("non-finite SRP1 REFERENT gradient")
        optimizer.step()
        source_bytes_seen += int(mask.sum())
        if update == 1 or update % args.log_interval == 0 or update == args.updates:
            record = {
                "update": update,
                "learning_rate": learning_rate,
                "loss": float(loss.detach()),
                "evidence_loss": float(evidence_loss.detach()),
                "query_loss": float(query_loss.detach()),
                "gradient_norm": float(gradient_norm),
            }
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    elapsed = time.monotonic() - started
    final_owner_hashes = model.owner_hashes()
    for owner in ("WORLD", "NUMERIC_EVIDENCE"):
        if final_owner_hashes[owner] != initial_owner_hashes[owner]:
            raise SystemExit(f"SRP1 immutable owner changed: {owner}")
    training_evaluation = {
        "evidence": _evaluate(
            model,
            evidence_rows,
            stage="EVIDENCE",
            device=device,
            batch_size=512,
        ),
        "query": _evaluate(
            model,
            query_rows,
            stage="QUERY",
            device=device,
            batch_size=512,
        ),
    }
    model_sha256 = module_state_sha256(model)
    manifest = model.owner_manifest()
    args.output.mkdir(parents=True)
    checkpoint_path = args.output / "checkpoint_0001000.pt"
    checkpoint_payload = {
        "schema": SCHEMA,
        "seed": args.seed,
        "update": args.updates,
        "config": {"width": 192, "layers": 2, "max_bytes": 192},
        "model_state": model.state_dict(),
        "model_state_sha256": model_sha256,
        "initial_owner_hashes": initial_owner_hashes,
        "final_owner_hashes": final_owner_hashes,
        "owner_manifest": manifest,
        "source_checkpoint_sha256": args.source_checkpoint_sha256,
        "evidence_checkpoint_sha256": args.evidence_checkpoint_sha256,
        "evidence_data_sha256": args.evidence_data_sha256,
        "query_data_sha256": args.query_data_sha256,
    }
    _atomic_checkpoint(checkpoint_path, checkpoint_payload)
    checkpoint_sha256 = sha256_path(checkpoint_path)
    report = {
        "schema": SCHEMA,
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "source_bytes_seen": source_bytes_seen,
        "source_bytes_per_second": source_bytes_seen / max(elapsed, 1e-9),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_owner_hashes": initial_owner_hashes,
        "final_owner_hashes": final_owner_hashes,
        "model_state_sha256": model_sha256,
        "owner_manifest": manifest,
        "training_evaluation": training_evaluation,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "evidence_data": str(args.evidence_data),
        "evidence_data_sha256": args.evidence_data_sha256,
        "query_data": str(args.query_data),
        "query_data_sha256": args.query_data_sha256,
    }
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "report": str(report_path),
                "report_sha256": sha256_path(report_path),
                "training_evaluation": training_evaluation,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

