#!/usr/bin/env python3
"""Train the frozen relational REFERENT owner for DIVERGE-RRG1."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

from diverge_iem1_runtime import module_state_sha256, tensorize_queries
from diverge_rrg1_data import ROWS_PER_STAGE, validate_training_record
from diverge_rrg1_runtime import (
    RRG1Config,
    RelationalReferentMachine,
    permutation_scores,
    permutation_targets,
    referent_parameters,
    validate_owner_contract,
)


SCHEMA = "shohin-diverge-rrg1-training-report-v1"
TRAIN_SEED = 2026080625


class RRG1TrainingError(RuntimeError):
    """The frozen RRG1 fit contract was violated."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path, expected_sha256: str, *, stage: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise RRG1TrainingError(f"protected data hash differs: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            validate_training_record(row)
            if row["stage"] != stage:
                raise RRG1TrainingError(f"protected data stage differs: {path}")
            rows.append(row)
    if len(rows) != ROWS_PER_STAGE:
        raise RRG1TrainingError(f"protected data row count differs: {path}")
    return rows


def _load_checkpoint(
    path: Path,
    expected_sha256: str,
    expected_schema: str,
    device: torch.device,
) -> dict[str, Any]:
    if sha256_path(path) != expected_sha256:
        raise RRG1TrainingError(f"protected checkpoint hash differs: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema") != expected_schema:
        raise RRG1TrainingError(f"protected checkpoint schema differs: {path}")
    return checkpoint


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
    model: RelationalReferentMachine,
    rows: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    overall = Counter()
    by_family: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_family_form: defaultdict[str, Counter[str]] = defaultdict(Counter)
    by_order: defaultdict[str, Counter[str]] = defaultdict(Counter)
    pair_outcomes: defaultdict[str, list[bool]] = defaultdict(list)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        ids, mask, symbols, targets = tensorize_queries(batch, device)
        logits = model.referent_owner(ids, mask, symbols)
        predictions = permutation_scores(logits).argmax(dim=-1)
        expected = permutation_targets(targets)
        exact = predictions.eq(expected).detach().cpu().tolist()
        for row, row_exact in zip(batch, exact, strict=True):
            counters = (
                overall,
                by_family[str(int(row["family"]))],
                by_family_form[f'{int(row["family"])}:{int(row["clause_form"])}'],
                by_order[str(int(row["role_order"]))],
            )
            for counter in counters:
                counter["total"] += 1
                counter["exact"] += bool(row_exact)
            pair_outcomes[str(row["pair_identity_sha256"])].append(bool(row_exact))
    paired_exact = sum(len(values) == 2 and all(values) for values in pair_outcomes.values())
    return {
        "stage": stage,
        "rows": len(rows),
        "exact": int(overall["exact"]),
        "exact_rate": overall["exact"] / len(rows),
        "pairs": len(pair_outcomes),
        "paired_exact": paired_exact,
        "paired_exact_rate": paired_exact / len(pair_outcomes),
        "by_family": {key: dict(value) for key, value in sorted(by_family.items())},
        "by_family_form": {
            key: dict(value) for key, value in sorted(by_family_form.items())
        },
        "by_role_order": {
            key: dict(value) for key, value in sorted(by_order.items())
        },
    }


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
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing RRG1 output: {args.output}")
    if (
        args.updates != 2000
        or args.batch_size != 256
        or args.learning_rate != 3e-3
        or args.seed != TRAIN_SEED
    ):
        raise SystemExit("RRG1 frozen training schedule differs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("RRG1 requested unavailable CUDA")
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    evidence_rows = _load_jsonl(
        args.evidence_data, args.evidence_data_sha256, stage="EVIDENCE"
    )
    query_rows = _load_jsonl(
        args.query_data, args.query_data_sha256, stage="QUERY"
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

    model = RelationalReferentMachine(RRG1Config()).to(device)
    model.source_owner.load_state_dict(source_checkpoint["model_state"], strict=True)
    model.numeric_evidence_owner.load_state_dict(
        evidence_checkpoint["model_state"], strict=True
    )
    model.freeze_qualified_owners()
    validate_owner_contract(model)
    initial_owner_hashes = model.owner_hashes()
    if initial_owner_hashes["WORLD"] != source_checkpoint["model_state_sha256"]:
        raise RRG1TrainingError("RRG1 WORLD owner differs")
    if (
        initial_owner_hashes["NUMERIC_EVIDENCE"]
        != evidence_checkpoint["model_state_sha256"]
    ):
        raise RRG1TrainingError("RRG1 numeric-evidence owner differs")

    parameters = referent_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    rng = random.Random(args.seed ^ 0x52524731)
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
        e_ids, e_mask, e_symbols, e_targets = tensorize_queries(
            evidence_batch, device
        )
        q_ids, q_mask, q_symbols, q_targets = tensorize_queries(query_batch, device)
        ids = torch.cat((e_ids, q_ids), dim=0)
        mask = torch.cat((e_mask, q_mask), dim=0)
        symbols = torch.cat((e_symbols, q_symbols), dim=0)
        targets = torch.cat((e_targets, q_targets), dim=0)

        optimizer.zero_grad(set_to_none=True)
        logits = model.referent_owner(ids, mask, symbols)
        scores = permutation_scores(logits)
        target_classes = permutation_targets(targets)
        evidence_loss = F.cross_entropy(
            scores[:half_batch], target_classes[:half_batch]
        )
        query_loss = F.cross_entropy(scores[half_batch:], target_classes[half_batch:])
        loss = 0.5 * (evidence_loss + query_loss)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(gradient_norm):
            raise SystemExit("non-finite RRG1 REFERENT gradient")
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
            raise SystemExit(f"RRG1 immutable owner changed: {owner}")
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
    checkpoint_path = args.output / "checkpoint_0002000.pt"
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
