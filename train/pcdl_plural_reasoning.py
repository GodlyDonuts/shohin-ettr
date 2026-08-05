#!/usr/bin/env python3
"""Matched plural-family gate for a prompt-conditioned determining law."""

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

from fcpt_plural_reasoning import (
    BoardConfig,
    EpisodeEncoder,
    FAMILIES,
    PluralBatch,
    batch_sha256,
    generate_batch,
)
from prompt_conditioned_determining_law import (
    DeterminingLawConfig,
    DeterminingLawOutput,
    LawArm,
    PromptConditionedDeterminingLaw,
)


SCHEMA = "shohin-pcdl-plural-reasoning-v1"


class DeterminingLawGateError(RuntimeError):
    """The determining-law gate violated its fixed contract."""


class DeterminingLawReasoner(nn.Module):
    def __init__(
        self,
        board_config: BoardConfig,
        law_config: DeterminingLawConfig,
        arm: LawArm,
    ):
        super().__init__()
        if board_config.width != law_config.width:
            raise DeterminingLawGateError("board and law widths differ")
        if board_config.modulus != law_config.outcome_classes:
            raise DeterminingLawGateError("board and law classes differ")
        self.encoder = EpisodeEncoder(board_config)
        self.core = PromptConditionedDeterminingLaw(law_config, arm)

    def forward(
        self,
        batch: PluralBatch,
        *,
        shuffle_outcomes: bool = False,
        destroy_law: bool = False,
    ) -> DeterminingLawOutput:
        _, probes, query = self.encoder(batch)
        return self.core(
            probes,
            batch.outcomes,
            batch.evidence_mask,
            query,
            shuffle_outcomes=shuffle_outcomes,
            destroy_law=destroy_law,
        )


def context_loss(result: DeterminingLawOutput, batch: PluralBatch) -> torch.Tensor:
    classes = result.context_logits.shape[-1]
    per_item = F.cross_entropy(
        result.context_logits.reshape(-1, classes),
        batch.outcomes.reshape(-1),
        reduction="none",
    ).view_as(batch.outcomes)
    mask = batch.evidence_mask.to(per_item.dtype)
    return (per_item * mask).sum() / mask.sum().clamp_min(1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def metrics(
    result: DeterminingLawOutput, batch: PluralBatch
) -> dict[str, float]:
    exact = result.selected_logits.argmax(-1).eq(batch.answer)
    context_exact = result.context_logits.argmax(-1).eq(batch.outcomes)
    mask = batch.evidence_mask
    measured = {
        "answer_accuracy": exact.float().mean().item(),
        "context_accuracy": context_exact[mask].float().mean().item(),
        "basis_rms": result.basis_rms.mean().item(),
        "coefficient_rms": result.coefficient_rms.mean().item(),
    }
    for family, name in enumerate(FAMILIES):
        selected = batch.family.eq(family)
        measured[f"{name}_accuracy"] = (
            exact[selected].float().mean().item()
            if selected.any()
            else float("nan")
        )
    return measured


@torch.inference_mode()
def evaluate(
    model: DeterminingLawReasoner,
    *,
    family: int,
    depth: int,
    count: int,
    seed: int,
    config: BoardConfig,
    device: torch.device,
    shuffle_outcomes: bool = False,
    destroy_law: bool = False,
) -> dict[str, Any]:
    model.eval()
    batch = generate_batch(
        count, depth, config, seed=seed, family=family, device=device
    )
    result = model(
        batch,
        shuffle_outcomes=shuffle_outcomes,
        destroy_law=destroy_law,
    )
    return {
        "family": FAMILIES[family],
        "depth": depth,
        "count": count,
        "seed": seed,
        "batch_sha256": batch_sha256(batch),
        "shuffle_outcomes": shuffle_outcomes,
        "destroy_law": destroy_law,
        **metrics(result, batch),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise DeterminingLawGateError(f"refusing existing report: {path}")
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
        raise DeterminingLawGateError(
            "CUDA is required unless --allow-cpu is explicit"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    board_config = BoardConfig(width=args.width)
    manifest_bytes = args.cohort_manifest.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != "shohin-fcpt-plural-cohort-v1":
        raise DeterminingLawGateError("cohort manifest schema differs")
    if manifest.get("status") != "frozen":
        raise DeterminingLawGateError("cohort manifest is not frozen")
    if manifest.get("config") != asdict(board_config):
        raise DeterminingLawGateError("cohort manifest config differs")
    cohort_hashes = {
        (
            int(row["family_id"]),
            int(row["depth"]),
            int(row["count"]),
            int(row["seed"]),
        ): str(row["sha256"])
        for row in manifest.get("batches") or []
        if row.get("split") == "development"
    }

    law_config = DeterminingLawConfig(
        width=args.width,
        rank=args.rank,
        heads=args.heads,
        ff_multiplier=args.ff_multiplier,
        outcome_classes=board_config.modulus,
        ridge=args.ridge,
    )
    model = DeterminingLawReasoner(board_config, law_config, args.arm).to(device)
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
        result = model(batch)
        answer_loss = F.cross_entropy(result.selected_logits, batch.answer)
        witness_loss = context_loss(result, batch)
        loss = answer_loss + args.context_weight * witness_loss
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
                "context_loss": witness_loss.item(),
                **metrics(result, batch),
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
            if args.arm == "law":
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
                evaluations.append(
                    evaluate(
                        model,
                        family=family,
                        depth=depth,
                        count=args.eval_count,
                        seed=seed,
                        config=board_config,
                        device=device,
                        destroy_law=True,
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
        if not row["shuffle_outcomes"] and not row["destroy_law"]
    }
    if set(observed) != set(cohort_hashes) or any(
        observed[key] != cohort_hashes[key] for key in cohort_hashes
    ):
        raise DeterminingLawGateError("generated development cohort hash differs")

    checkpoint = args.output.with_suffix(".pt")
    if checkpoint.exists():
        raise DeterminingLawGateError(
            f"refusing existing checkpoint: {checkpoint}"
        )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": SCHEMA,
            "arm": args.arm,
            "seed": args.seed,
            "board_config": asdict(board_config),
            "law_config": asdict(law_config),
            "model": model.state_dict(),
        },
        checkpoint,
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "arm": args.arm,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "charged_examples": args.updates * args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "context_weight": args.context_weight,
        "train_depth_max": args.train_depth_max,
        "board_config": asdict(board_config),
        "law_config": asdict(law_config),
        "parameters": parameter_count(model),
        "elapsed_seconds": elapsed,
        "examples_per_second": args.updates * args.batch_size / elapsed,
        "cohort_manifest": str(args.cohort_manifest),
        "cohort_manifest_sha256": manifest_sha256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "train_log": train_log,
        "evaluations": evaluations,
    }
    _atomic_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("law", "dense"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--data-seed", type=int, default=20260896)
    parser.add_argument("--eval-seed", type=int, default=51000)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-count", type=int, default=1024)
    parser.add_argument("--train-depth-max", type=int, default=4)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-multiplier", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--context-weight", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    completed = run(parse_args())
    print(
        json.dumps(
            {
                "status": completed["status"],
                "arm": completed["arm"],
                "parameters": completed["parameters"],
                "examples_per_second": completed["examples_per_second"],
            },
            sort_keys=True,
        )
    )
