#!/usr/bin/env python3
"""Matched plural-family gate for counterexample-guided sparse revision."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from counterexample_guided_revision import (
    CounterexampleGuidedRevisionCore,
    RevisionArm,
    RevisionConfig,
    RevisionTrajectory,
)
from fcpt_plural_reasoning import (
    BoardConfig,
    EpisodeEncoder,
    FAMILIES,
    PluralBatch,
    batch_sha256,
    generate_batch,
)


SCHEMA = "shohin-cgsgr-plural-reasoning-v1"


class RevisionGateError(RuntimeError):
    """The sparse-revision gate violated its fixed contract."""


class RevisionReasoner(nn.Module):
    def __init__(
        self,
        board_config: BoardConfig,
        revision_config: RevisionConfig,
        arm: RevisionArm,
    ):
        super().__init__()
        if board_config.width != revision_config.width:
            raise RevisionGateError("board and revision widths differ")
        self.encoder = EpisodeEncoder(board_config)
        self.core = CounterexampleGuidedRevisionCore(revision_config, arm)

    def forward(
        self, batch: PluralBatch, *, shuffle_outcomes: bool = False
    ) -> tuple[torch.Tensor, RevisionTrajectory]:
        source, probes, query = self.encoder(batch)
        return self.core(
            source,
            batch.evidence_mask,
            probes,
            batch.outcomes,
            batch.evidence_mask,
            query,
            shuffle_outcomes=shuffle_outcomes,
        )


def final_behavior_loss(
    trajectory: RevisionTrajectory, batch: PluralBatch
) -> torch.Tensor:
    logits = trajectory.steps[-1].behavior_logits
    classes = logits.shape[-1]
    per_item = F.cross_entropy(
        logits.reshape(-1, classes),
        batch.outcomes.reshape(-1),
        reduction="none",
    ).view_as(batch.outcomes)
    mask = batch.evidence_mask.to(per_item.dtype)
    return (per_item * mask).sum() / mask.sum().clamp_min(1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def metrics(
    logits: torch.Tensor, trajectory: RevisionTrajectory, batch: PluralBatch
) -> dict[str, float]:
    prediction = logits.argmax(-1)
    exact = prediction.eq(batch.answer)
    result = {"answer_accuracy": exact.float().mean().item()}
    for family, name in enumerate(FAMILIES):
        selected = batch.family.eq(family)
        result[f"{name}_accuracy"] = (
            exact[selected].float().mean().item() if selected.any() else float("nan")
        )
    mask = batch.evidence_mask.to(logits.dtype)
    first = trajectory.steps[0].contradiction
    last = trajectory.steps[-1].contradiction
    first_mean = (first * mask).sum() / mask.sum().clamp_min(1)
    last_mean = (last * mask).sum() / mask.sum().clamp_min(1)
    result["initial_contradiction"] = first_mean.item()
    result["final_contradiction"] = last_mean.item()
    result["contradiction_reduction"] = (first_mean - last_mean).item()
    result["revision_fraction"] = (
        trajectory.steps[-1].slot_mask.float().mean().item()
    )
    result["correction_rms"] = trajectory.steps[-1].correction_rms.mean().item()
    return result


@torch.inference_mode()
def evaluate(
    model: RevisionReasoner,
    *,
    family: int,
    depth: int,
    count: int,
    seed: int,
    config: BoardConfig,
    device: torch.device,
    shuffle_outcomes: bool = False,
) -> dict[str, Any]:
    model.eval()
    batch = generate_batch(
        count, depth, config, seed=seed, family=family, device=device
    )
    logits, trajectory = model(batch, shuffle_outcomes=shuffle_outcomes)
    return {
        "family": FAMILIES[family],
        "depth": depth,
        "count": count,
        "seed": seed,
        "batch_sha256": batch_sha256(batch),
        "shuffle_outcomes": shuffle_outcomes,
        **metrics(logits, trajectory, batch),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RevisionGateError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RevisionGateError("CUDA is required unless --allow-cpu is explicit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    board_config = BoardConfig(width=args.width)
    cohort_manifest_sha256 = None
    cohort_hashes: dict[tuple[int, int, int, int], str] = {}
    if args.cohort_manifest is not None:
        cohort_bytes = args.cohort_manifest.read_bytes()
        cohort_manifest_sha256 = hashlib.sha256(cohort_bytes).hexdigest()
        cohort_manifest = json.loads(cohort_bytes)
        if cohort_manifest.get("schema") != "shohin-fcpt-plural-cohort-v1":
            raise RevisionGateError("cohort manifest schema differs")
        if cohort_manifest.get("status") != "frozen":
            raise RevisionGateError("cohort manifest is not frozen")
        if cohort_manifest.get("config") != asdict(board_config):
            raise RevisionGateError("cohort manifest config differs")
        cohort_hashes = {
            (
                int(row["family_id"]),
                int(row["depth"]),
                int(row["count"]),
                int(row["seed"]),
            ): str(row["sha256"])
            for row in cohort_manifest.get("batches") or []
            if row.get("split") == "development"
        }
    revision_config = RevisionConfig(
        width=args.width,
        slots=args.slots,
        rounds=args.rounds,
        heads=args.heads,
        ff_multiplier=args.ff_multiplier,
        outcome_classes=board_config.modulus,
        answer_classes=board_config.modulus,
        probes_per_round=args.probes_per_round,
        revision_slots=args.revision_slots,
    )
    model = RevisionReasoner(board_config, revision_config, args.arm).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    started = time.monotonic()
    train_log = []
    model.train()
    for update in range(1, args.updates + 1):
        depth = 2 + ((update * 997 + args.seed) % (args.train_depth_max - 1))
        batch = generate_batch(
            args.batch_size,
            depth,
            board_config,
            seed=args.data_seed + update,
            device=device,
        )
        logits, trajectory = model(batch)
        answer_loss = F.cross_entropy(logits, batch.answer)
        behavior = final_behavior_loss(trajectory, batch)
        loss = answer_loss + args.behavior_weight * behavior
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update == 1 or update % args.log_every == 0 or update == args.updates:
            row = {
                "update": update,
                "depth": depth,
                "loss": loss.item(),
                "answer_loss": answer_loss.item(),
                "behavior_loss": behavior.item(),
                **metrics(logits, trajectory, batch),
            }
            train_log.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    elapsed = time.monotonic() - started
    evaluations = []
    for family in range(len(FAMILIES)):
        for depth in (5, 7):
            seed = args.eval_seed + family * 100 + depth
            evaluations.append(
                evaluate(
                    model,
                    family=family,
                    depth=depth,
                    count=args.eval_count,
                    seed=seed,
                    config=board_config,
                    device=device,
                )
            )
            if args.arm == "guided":
                evaluations.append(
                    evaluate(
                        model,
                        family=family,
                        depth=depth,
                        count=args.eval_count,
                        seed=seed,
                        config=board_config,
                        device=device,
                        shuffle_outcomes=True,
                    )
                )
    if cohort_hashes:
        observed = {
            (
                FAMILIES.index(row["family"]),
                int(row["depth"]),
                int(row["count"]),
                int(row["seed"]),
            ): str(row["batch_sha256"])
            for row in evaluations
            if not row["shuffle_outcomes"]
        }
        if set(observed) != set(cohort_hashes) or any(
            observed[key] != cohort_hashes[key] for key in cohort_hashes
        ):
            raise RevisionGateError("generated development cohort hash differs")
    checkpoint = args.output.with_suffix(".pt")
    if checkpoint.exists():
        raise RevisionGateError(f"refusing existing checkpoint: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": SCHEMA,
            "arm": args.arm,
            "board_config": asdict(board_config),
            "revision_config": asdict(revision_config),
            "model": model.state_dict(),
            "seed": args.seed,
            "data_seed": args.data_seed,
            "updates": args.updates,
        },
        checkpoint,
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return {
        "schema": SCHEMA,
        "status": "complete",
        "arm": args.arm,
        "board_config": asdict(board_config),
        "revision_config": asdict(revision_config),
        "seed": args.seed,
        "data_seed": args.data_seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "train_depth_max": args.train_depth_max,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "behavior_weight": args.behavior_weight,
        "parameters": parameter_count(model),
        "charged_examples": args.updates * args.batch_size,
        "elapsed_seconds": elapsed,
        "examples_per_second": args.updates * args.batch_size / elapsed,
        "train_log": train_log,
        "evaluations": evaluations,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "cohort_manifest": (
            str(args.cohort_manifest.resolve()) if args.cohort_manifest else None
        ),
        "cohort_manifest_sha256": cohort_manifest_sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("guided", "fixed", "dense"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--data-seed", type=int, default=20260870)
    parser.add_argument("--eval-seed", type=int, default=51000)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-count", type=int, default=1024)
    parser.add_argument("--train-depth-max", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-multiplier", type=int, default=2)
    parser.add_argument("--probes-per-round", type=int, default=2)
    parser.add_argument("--revision-slots", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--behavior-weight", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--cohort-manifest", type=Path)
    args = parser.parse_args()
    if not 2 <= args.train_depth_max <= 4:
        parser.error("pilot training depth must be between 2 and 4")
    if min(args.updates, args.batch_size, args.eval_count, args.log_every) <= 0:
        parser.error("training counts must be positive")
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"train_log", "evaluations"}
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
