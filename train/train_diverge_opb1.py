#!/usr/bin/env python3
"""Train one matched DIVERGE-OPB1 evidence-operation pointer arm."""

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
from diverge_opb1_data import (
    REPORT_SCHEMA as DATA_REPORT_SCHEMA,
    TRAIN_ROWS,
    TRAIN_SEED,
    validate_training_record,
)
from diverge_opb1_runtime import (
    CHECKPOINT_SCHEMA,
    EvidenceOperationPointer,
    EvidenceOperationPointerConfig,
    tensorize_operation_sources,
)


REPORT_SCHEMA = "shohin-diverge-opb1-training-report-v1"
UPDATES = 1_000
BATCH_SIZE = 128
LEARNING_RATE = 0.001


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
        raise RuntimeError("OPB1 stale checkpoint temporary exists")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError("OPB1 training data hash differs")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    for row in rows:
        validate_training_record(row)
    return rows


def _load_data_report(
    path: Path, expected_sha256: str, data: Path, data_sha256: str
) -> dict[str, Any]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError("OPB1 data report hash differs")
    report = json.loads(path.read_text())
    entry = report.get("files", {}).get("training", {})
    if (
        report.get("schema") != DATA_REPORT_SCHEMA
        or not report.get("zero_source_name_and_identity_overlap")
        or not report.get("training_deterministic_regeneration")
        or Path(str(entry.get("path", ""))).name != data.name
        or entry.get("sha256") != data_sha256
        or int(entry.get("rows", -1)) != TRAIN_ROWS
    ):
        raise RuntimeError("OPB1 data report custody differs")
    return report


def _model_records(
    rows: Sequence[Mapping[str, Any]], *, aliases_key: str
) -> list[dict[str, Any]]:
    return [
        {"source_text": row["source_text"], "aliases": row[aliases_key]} for row in rows
    ]


@torch.no_grad()
def _sample_accuracy(
    model: EvidenceOperationPointer,
    rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    aliases_key: str,
) -> dict[str, float | int]:
    sample = rows[: min(2_048, len(rows))]
    exact = 0
    for start in range(0, len(sample), 128):
        raw = sample[start : start + 128]
        records = _model_records(raw, aliases_key=aliases_key)
        prediction = model(*tensorize_operation_sources(records, device)).argmax(dim=-1)
        target = torch.tensor([row["operation_target"] for row in raw], device=device)
        exact += int(prediction.eq(target).sum())
    return {"exact": exact, "total": len(sample), "rate": exact / len(sample)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--data-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--arm", choices=("treatment", "decoy_table"), required=True)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing OPB1 output: {args.output}")
    if (
        args.updates != UPDATES
        or args.batch_size != BATCH_SIZE
        or not math.isclose(args.learning_rate, LEARNING_RATE)
    ):
        raise SystemExit("OPB1 frozen training schedule differs")
    args.output.mkdir(parents=True)
    _load_data_report(
        args.data_report,
        args.data_report_sha256,
        args.data,
        args.data_sha256,
    )
    rows = _load_jsonl(args.data, args.data_sha256)
    if len(rows) != TRAIN_ROWS:
        raise SystemExit("OPB1 training row count differs")
    if not torch.cuda.is_available():
        raise SystemExit("OPB1 training requires CUDA")
    device = torch.device("cuda")
    aliases_key = "aliases" if args.arm == "treatment" else "decoy_aliases"

    torch.manual_seed(TRAIN_SEED)
    random.seed(TRAIN_SEED)
    model = EvidenceOperationPointer().to(device)
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
        raw = [rows[index] for index in indices]
        records = _model_records(raw, aliases_key=aliases_key)
        logits = model(*tensorize_operation_sources(records, device))
        targets = torch.tensor([row["operation_target"] for row in raw], device=device)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0))
        if not torch.isfinite(loss) or not math.isfinite(gradient_norm):
            raise RuntimeError("OPB1 training became nonfinite")
        optimizer.step()
        charged += args.batch_size
        if update in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, args.updates):
            history.append(
                {
                    "update": update,
                    "loss": float(loss.detach()),
                    "gradient_norm": gradient_norm,
                }
            )
    elapsed = time.perf_counter() - started
    model.eval()
    sample = _sample_accuracy(model, rows, device, aliases_key=aliases_key)
    final_state_sha256 = module_state_sha256(model)
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "source_commit": args.source_commit,
        "arm": args.arm,
        "config": asdict(EvidenceOperationPointerConfig()),
        "model_state": model.state_dict(),
        "model_state_sha256": final_state_sha256,
        "initial_state_sha256": initial_state_sha256,
        "data_sha256": args.data_sha256,
        "data_report_sha256": args.data_report_sha256,
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
        "data_report": str(args.data_report),
        "data_report_sha256": args.data_report_sha256,
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
