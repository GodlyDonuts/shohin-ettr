#!/usr/bin/env python3
"""Train the one frozen DIVERGE-NPW1 narrative WORLD ingress owner."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from diverge_iem1_runtime import module_state_sha256
from diverge_npw1_runtime import (
    MAX_EVENTS,
    NPW1Config,
    NarrativeStageOwnedMachine,
    ROLE_NAMES,
    tensorize_records,
)
from diverge_sot1_runtime import SOT1Config, validate_owner_isolation


SCHEMA = "shohin-diverge-npw1-training-result-v1"
SEED = 2026080620
UPDATES = 2000
BATCH_SIZE = 32
LEARNING_RATE = 2e-3


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rows(path: Path, expected_sha256: str) -> list[dict[str, object]]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError("NPW1 training data hash differs")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    device = batch["event_count"].device
    positions = torch.arange(MAX_EVENTS + 1, device=device).view(1, -1)
    form_mask = positions <= batch["event_count"].view(-1, 1)
    form_loss = F.cross_entropy(
        output["form_logits"][form_mask], batch["form_targets"][form_mask]
    )
    event_mask = batch["event_mask"]
    start_loss = F.cross_entropy(
        output["start_logits"][event_mask], batch["start_targets"][event_mask]
    )
    end_loss = F.cross_entropy(
        output["end_logits"][event_mask], batch["end_targets"][event_mask]
    )
    role_losses = []
    for role_index in range(len(ROLE_NAMES)):
        role_losses.append(
            F.cross_entropy(
                output["role_logits"][:, :, role_index][event_mask],
                batch["role_targets"][:, :, role_index][event_mask],
            )
        )
    role_loss = torch.stack(role_losses).mean()
    total = form_loss + start_loss + end_loss + role_loss
    with torch.no_grad():
        metrics = {
            "loss": float(total.detach()),
            "form_loss": float(form_loss.detach()),
            "start_loss": float(start_loss.detach()),
            "end_loss": float(end_loss.detach()),
            "role_loss": float(role_loss.detach()),
            "form_exact": float(
                output["form_logits"][form_mask]
                .argmax(dim=-1)
                .eq(batch["form_targets"][form_mask])
                .float()
                .mean()
            ),
            "start_exact": float(
                output["start_logits"][event_mask]
                .argmax(dim=-1)
                .eq(batch["start_targets"][event_mask])
                .float()
                .mean()
            ),
            "end_exact": float(
                output["end_logits"][event_mask]
                .argmax(dim=-1)
                .eq(batch["end_targets"][event_mask])
                .float()
                .mean()
            ),
            "role_exact": float(
                output["role_logits"][event_mask]
                .argmax(dim=-1)
                .eq(batch["role_targets"][event_mask])
                .float()
                .mean()
            ),
        }
    return total, metrics


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sot1-checkpoint", type=Path, required=True)
    parser.add_argument("--sot1-checkpoint-sha256", required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--training-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing NPW1 model output: {args.output}")
    if sha256_path(args.sot1_checkpoint) != args.sot1_checkpoint_sha256:
        raise SystemExit("SOT1 checkpoint hash differs")

    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True
    rows = _load_rows(args.training, args.training_sha256)
    if len(rows) != 20_000:
        raise SystemExit("NPW1 training row count differs")

    payload = torch.load(args.sot1_checkpoint, map_location="cpu", weights_only=False)
    model = NarrativeStageOwnedMachine(SOT1Config(), NPW1Config())
    model.sot1.load_state_dict(payload["model_state"], strict=True)
    model.freeze_inherited_owners()
    validate_owner_isolation(model.sot1)
    initial_hashes = model.sot1.owner_hashes()
    if initial_hashes != payload["final_owner_hashes"]:
        raise SystemExit("SOT1 inherited owner hashes differ")
    model.to(device)
    parameters = tuple(model.world_ingress.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    rng = random.Random(SEED ^ 0x4E505731)
    updates = 2 if args.smoke else UPDATES
    batch_size = 2 if args.smoke else BATCH_SIZE
    started = time.monotonic()
    source_bytes = 0
    final_metrics: dict[str, float] = {}
    skipped = 0
    model.train()
    for update in range(1, updates + 1):
        selected = [rows[rng.randrange(len(rows))] for _ in range(batch_size)]
        source_bytes += sum(
            len(str(row["natural_world"]["source_text"]).encode("ascii"))
            for row in selected
        )
        batch = tensorize_records(selected, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model.world_ingress(
                batch["byte_ids"],
                batch["byte_mask"],
                batch["candidate_masks"],
                batch["candidate_valid"],
                batch["candidate_kind"],
                teacher_starts=batch["start_targets"],
            )
            loss, final_metrics = _losses(output, batch)
        if not torch.isfinite(loss):
            skipped += 1
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        progress = update / updates
        learning_rate = LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        if update == 1 or update % 100 == 0:
            print(
                json.dumps(
                    {
                        "update": update,
                        "lr": learning_rate,
                        **final_metrics,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    elapsed = time.monotonic() - started
    final_hashes = model.sot1.owner_hashes()
    if final_hashes != initial_hashes:
        raise SystemExit("NPW1 training mutated an inherited owner")
    args.output.mkdir(parents=True)
    checkpoint_path = args.output / f"checkpoint_{updates:07d}.pt"
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    checkpoint = {
        "schema": SCHEMA,
        "seed": SEED,
        "update": updates,
        "smoke": args.smoke,
        "sot1_checkpoint_sha256": args.sot1_checkpoint_sha256,
        "training_sha256": args.training_sha256,
        "sot1_config": asdict(SOT1Config()),
        "npw1_config": asdict(NPW1Config()),
        "model_state": model.state_dict(),
        "inherited_owner_hashes": final_hashes,
        "world_ingress_state_sha256": module_state_sha256(model.world_ingress),
    }
    torch.save(checkpoint, temporary)
    os.replace(temporary, checkpoint_path)
    report: dict[str, object] = {
        "schema": SCHEMA,
        "seed": SEED,
        "updates": updates,
        "batch_size": batch_size,
        "smoke": args.smoke,
        "learning_rate": LEARNING_RATE,
        "examples_charged": updates * batch_size,
        "source_bytes": source_bytes,
        "skipped_updates": skipped,
        "elapsed_seconds": elapsed,
        "source_bytes_per_second": source_bytes / elapsed,
        "trainable_parameters": sum(value.numel() for value in parameters),
        "composite_parameters": sum(value.numel() for value in model.parameters()),
        "peak_cuda_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "final_metrics": final_metrics,
        "inherited_owner_hashes_before": initial_hashes,
        "inherited_owner_hashes_after": final_hashes,
        "world_ingress_state_sha256": checkpoint["world_ingress_state_sha256"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "sot1_checkpoint_sha256": args.sot1_checkpoint_sha256,
        "training_sha256": args.training_sha256,
    }
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
