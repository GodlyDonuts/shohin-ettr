#!/usr/bin/env python3
"""Evaluate deferred whole-presentation closure on a frozen compiler."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from learned_pspa_language_reasoning import (
    LanguageConfig,
    LearnedPSPAGate,
    execute_word,
    greedy_permutation,
    render_source,
)
from prompt_selected_presented_algebra import PresentedAlgebraConfig
from pspa_presented_reasoning import FAMILIES, batch_sha256, generate_batch


SCHEMA = "shohin-deferred-whole-presentation-closure-v1"


def close_presentation(
    row_probabilities: torch.Tensor,
    generator_mask: torch.Tensor,
) -> torch.Tensor:
    tables = greedy_permutation(row_probabilities)
    identity = torch.eye(
        tables.shape[-1], device=tables.device, dtype=tables.dtype
    )
    return torch.where(
        generator_mask[..., None, None], tables, identity[None, None]
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@torch.inference_mode()
def evaluate_cohort(
    model: LearnedPSPAGate,
    *,
    family: int,
    length: int,
    count: int,
    seed: int,
    renderer_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    cpu_batch = generate_batch(count, length, model.algebra, seed=seed, family=family)
    source = render_source(cpu_batch, model.algebra, seed=renderer_seed)
    shuffled_batch = replace(
        cpu_batch, challenge_outcome=cpu_batch.challenge_outcome.roll(1, 0)
    )
    shuffled_source = render_source(
        shuffled_batch, model.algebra, seed=renderer_seed
    )
    batch = cpu_batch.to(device)
    source = source.to(device)
    shuffled_source = shuffled_source.to(device)
    _, row_probabilities = model.row_soft(source, batch.generator_mask, hard=False)
    _, shuffled_probabilities = model.row_soft(
        shuffled_source, batch.generator_mask, hard=False
    )
    closed = close_presentation(row_probabilities, batch.generator_mask)
    shuffled = close_presentation(shuffled_probabilities, batch.generator_mask)
    _, jointly_projected = model.presented(source, batch.generator_mask, hard=True)
    _, row_hard = model.row_soft(source, batch.generator_mask, hard=True)

    def answer(tables: torch.Tensor) -> torch.Tensor:
        return execute_word(
            tables, batch.query_start, batch.query_word, batch.query_word_mask
        ).argmax(-1)

    challenge = execute_word(
        closed,
        batch.challenge_start,
        batch.challenge_word,
        batch.challenge_word_mask,
    ).argmax(-1)
    table_exact = closed.argmax(-1).eq(batch.true_tables.long()).all((-1, -2))
    challenge_correct = challenge.eq(batch.challenge_outcome) & batch.challenge_mask
    return {
        "family": FAMILIES[family],
        "length": length,
        "count": count,
        "seed": seed,
        "renderer_seed": renderer_seed,
        "batch_sha256": batch_sha256(cpu_batch),
        "deferred_accuracy": answer(closed).eq(batch.answer).float().mean().item(),
        "row_soft_accuracy": answer(row_hard).eq(batch.answer).float().mean().item(),
        "joint_projection_accuracy": answer(jointly_projected)
        .eq(batch.answer)
        .float()
        .mean()
        .item(),
        "shuffle_challenge_accuracy": answer(shuffled)
        .eq(batch.answer)
        .float()
        .mean()
        .item(),
        "lineage_swap_accuracy": answer(closed.roll(1, 0))
        .eq(batch.answer)
        .float()
        .mean()
        .item(),
        "challenge_exact": challenge_correct.sum()
        .div(batch.challenge_mask.sum())
        .item(),
        "selected_table_exact": table_exact.float().mean().item(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is required unless --allow-cpu is explicit")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_bytes = args.checkpoint.read_bytes()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    algebra = PresentedAlgebraConfig(**checkpoint["algebra_config"])
    language = LanguageConfig(**checkpoint["language_config"])
    model = LearnedPSPAGate(algebra, language).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    evaluations = []
    for family in range(len(FAMILIES)):
        for length in (8, 12):
            evaluations.append(
                evaluate_cohort(
                    model,
                    family=family,
                    length=length,
                    count=args.eval_count,
                    seed=args.eval_seed + family * 100 + length,
                    renderer_seed=args.renderer_seed + family * 100 + length,
                    device=device,
                )
            )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "eval_count": args.eval_count,
        "eval_seed": args.eval_seed,
        "renderer_seed": args.renderer_seed,
        "evaluations": evaluations,
    }
    _atomic_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-count", type=int, default=1024)
    parser.add_argument("--eval-seed", type=int, default=72000)
    parser.add_argument("--renderer-seed", type=int, default=73000)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    completed = run(parse_args())
    print(json.dumps({"status": completed["status"]}, sort_keys=True))

