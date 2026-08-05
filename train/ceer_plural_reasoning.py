#!/usr/bin/env python3
"""Matched plural-family gate for counterfactual energy equilibrium."""

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

from counterexample_guided_revision import RevisionConfig
from counterfactual_energy_equilibrium import (
    CounterfactualEnergyEquilibriumCore,
    EquilibriumArm,
    EquilibriumTrajectory,
)
from fcpt_plural_reasoning import (
    BoardConfig,
    EpisodeEncoder,
    FAMILIES,
    PluralBatch,
    batch_sha256,
    generate_batch,
)


SCHEMA = "shohin-ceer-plural-reasoning-v1"


class EquilibriumGateError(RuntimeError):
    """The energy-equilibrium gate violated its fixed contract."""


class EquilibriumReasoner(nn.Module):
    def __init__(
        self,
        board_config: BoardConfig,
        revision_config: RevisionConfig,
        arm: EquilibriumArm,
        maximum_step: float,
    ):
        super().__init__()
        if board_config.width != revision_config.width:
            raise EquilibriumGateError("board and equilibrium widths differ")
        self.encoder = EpisodeEncoder(board_config)
        self.core = CounterfactualEnergyEquilibriumCore(
            revision_config, arm, maximum_step
        )

    def forward(
        self,
        batch: PluralBatch,
        *,
        shuffle_outcomes: bool = False,
        zero_energy_gradient: bool = False,
    ) -> tuple[torch.Tensor, EquilibriumTrajectory]:
        source, probes, query = self.encoder(batch)
        return self.core(
            source,
            batch.evidence_mask,
            probes,
            batch.outcomes,
            batch.evidence_mask,
            query,
            shuffle_outcomes=shuffle_outcomes,
            zero_energy_gradient=zero_energy_gradient,
        )


def final_behavior_loss(
    trajectory: EquilibriumTrajectory, batch: PluralBatch
) -> torch.Tensor:
    classes = trajectory.final_behavior_logits.shape[-1]
    per_item = F.cross_entropy(
        trajectory.final_behavior_logits.reshape(-1, classes),
        batch.outcomes.reshape(-1),
        reduction="none",
    ).view_as(batch.outcomes)
    mask = batch.evidence_mask.to(per_item.dtype)
    return (per_item * mask).sum() / mask.sum().clamp_min(1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def metrics(
    logits: torch.Tensor,
    trajectory: EquilibriumTrajectory,
    batch: PluralBatch,
) -> dict[str, float]:
    exact = logits.argmax(-1).eq(batch.answer)
    result = {"answer_accuracy": exact.float().mean().item()}
    for family, name in enumerate(FAMILIES):
        selected = batch.family.eq(family)
        result[f"{name}_accuracy"] = (
            exact[selected].float().mean().item() if selected.any() else float("nan")
        )
    initial_energy = trajectory.steps[0].evidence_energy.mean()
    final_energy = trajectory.final_evidence_energy.mean()
    result.update(
        {
            "initial_evidence_energy": initial_energy.item(),
            "final_evidence_energy": final_energy.item(),
            "evidence_energy_reduction": (initial_energy - final_energy).item(),
            "energy_gradient_rms": trajectory.steps[-1]
            .energy_gradient_rms.mean()
            .item(),
            "energy_correction_rms": trajectory.steps[-1]
            .energy_correction_rms.mean()
            .item(),
            "recurrent_correction_rms": trajectory.steps[-1]
            .recurrent_correction_rms.mean()
            .item(),
        }
    )
    return result


def evaluate(
    model: EquilibriumReasoner,
    *,
    family: int,
    depth: int,
    count: int,
    seed: int,
    config: BoardConfig,
    device: torch.device,
    shuffle_outcomes: bool = False,
    zero_energy_gradient: bool = False,
) -> dict[str, Any]:
    model.eval()
    batch = generate_batch(
        count, depth, config, seed=seed, family=family, device=device
    )
    with torch.enable_grad():
        logits, trajectory = model(
            batch,
            shuffle_outcomes=shuffle_outcomes,
            zero_energy_gradient=zero_energy_gradient,
        )
    return {
        "family": FAMILIES[family],
        "depth": depth,
        "count": count,
        "seed": seed,
        "batch_sha256": batch_sha256(batch),
        "shuffle_outcomes": shuffle_outcomes,
        "zero_energy_gradient": zero_energy_gradient,
        **metrics(logits, trajectory, batch),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise EquilibriumGateError(f"refusing existing report: {path}")
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
        raise EquilibriumGateError("CUDA is required unless --allow-cpu is explicit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    board_config = BoardConfig(width=args.width)
    cohort_bytes = args.cohort_manifest.read_bytes()
    cohort_manifest_sha256 = hashlib.sha256(cohort_bytes).hexdigest()
    cohort_manifest = json.loads(cohort_bytes)
    if cohort_manifest.get("schema") != "shohin-fcpt-plural-cohort-v1":
        raise EquilibriumGateError("cohort manifest schema differs")
    if cohort_manifest.get("status") != "frozen":
        raise EquilibriumGateError("cohort manifest is not frozen")
    if cohort_manifest.get("config") != asdict(board_config):
        raise EquilibriumGateError("cohort manifest config differs")
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
    )
    model = EquilibriumReasoner(
        board_config, revision_config, args.arm, args.maximum_step
    ).to(device)
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
            if args.arm == "energy":
                for ablation in ("outcomes", "gradient"):
                    evaluations.append(
                        evaluate(
                            model,
                            family=family,
                            depth=depth,
                            count=args.eval_count,
                            seed=seed,
                            config=board_config,
                            device=device,
                            shuffle_outcomes=ablation == "outcomes",
                            zero_energy_gradient=ablation == "gradient",
                        )
                    )
    observed = {
        (
            FAMILIES.index(row["family"]),
            int(row["depth"]),
            int(row["count"]),
            int(row["seed"]),
        ): str(row["batch_sha256"])
        for row in evaluations
        if not row["shuffle_outcomes"] and not row["zero_energy_gradient"]
    }
    if set(observed) != set(cohort_hashes) or any(
        observed[key] != cohort_hashes[key] for key in cohort_hashes
    ):
        raise EquilibriumGateError("generated development cohort hash differs")
    checkpoint = args.output.with_suffix(".pt")
    if checkpoint.exists():
        raise EquilibriumGateError(f"refusing existing checkpoint: {checkpoint}")
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
        "maximum_step": args.maximum_step,
        "parameters": parameter_count(model),
        "charged_examples": args.updates * args.batch_size,
        "elapsed_seconds": elapsed,
        "examples_per_second": args.updates * args.batch_size / elapsed,
        "train_log": train_log,
        "evaluations": evaluations,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "cohort_manifest": str(args.cohort_manifest.resolve()),
        "cohort_manifest_sha256": cohort_manifest_sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("energy", "recurrent"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--data-seed", type=int, default=20260890)
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
    parser.add_argument("--maximum-step", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--behavior-weight", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.train_depth_max <= 4:
        parser.error("pilot training depth must be between 2 and 4")
    if min(args.updates, args.batch_size, args.eval_count, args.log_every) <= 0:
        parser.error("training counts must be positive")
    if not 0.0 < args.maximum_step <= 1.0:
        parser.error("maximum step is outside (0, 1]")
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
