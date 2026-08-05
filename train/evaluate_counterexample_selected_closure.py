#!/usr/bin/env python3
"""Select a deferred whole presentation with source counterexamples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from evaluate_deferred_pspa_closure import close_presentation
from learned_pspa_language_reasoning import (
    LanguageConfig,
    LearnedPSPAGate,
    execute_word,
    render_source,
)
from prompt_selected_presented_algebra import PresentedAlgebraConfig
from pspa_presented_reasoning import FAMILIES, batch_sha256, generate_batch


SCHEMA = "shohin-counterexample-selected-deferred-closure-v1"


def binary_completion_candidates(
    row_probabilities: torch.Tensor,
    generator_mask: torch.Tensor,
) -> torch.Tensor:
    """Construct whole presentations from two uncertain rows per generator."""

    batch, generators, carrier, outputs = row_probabilities.shape
    if carrier != outputs:
        raise RuntimeError("generator action must be square")
    confidence = row_probabilities.max(-1).values
    row_order = confidence.argsort(-1, descending=True)
    base = torch.zeros_like(row_probabilities)
    available = torch.ones(
        batch, generators, carrier, dtype=torch.bool, device=row_probabilities.device
    )
    batch_index = torch.arange(batch, device=row_probabilities.device)[:, None]
    generator_index = torch.arange(generators, device=row_probabilities.device)[None]
    for rank in range(carrier - 2):
        row = row_order[..., rank]
        scores = row_probabilities[batch_index, generator_index, row]
        column = scores.masked_fill(~available, -torch.inf).argmax(-1)
        base[batch_index, generator_index, row, column] = 1
        available.scatter_(-1, column[..., None], False)

    uncertain_rows = row_order[..., -2:]
    remaining_columns = available.to(torch.int64).topk(2, -1).indices
    per_generator = base[:, :, None].expand(-1, -1, 2, -1, -1).clone()
    for option in range(2):
        first_column = remaining_columns[..., option]
        second_column = remaining_columns[..., 1 - option]
        per_generator[
            batch_index,
            generator_index,
            option,
            uncertain_rows[..., 0],
            first_column,
        ] = 1
        per_generator[
            batch_index,
            generator_index,
            option,
            uncertain_rows[..., 1],
            second_column,
        ] = 1

    identity = torch.eye(
        carrier, dtype=row_probabilities.dtype, device=row_probabilities.device
    )
    per_generator = torch.where(
        generator_mask[..., None, None, None],
        per_generator,
        identity[None, None, None],
    )
    presentations = []
    for candidate in range(1 << generators):
        tables = []
        for generator in range(generators):
            option = (candidate >> generator) & 1
            tables.append(per_generator[:, generator, option])
        presentations.append(torch.stack(tables, 1))
    return torch.stack(presentations, 1)


def select_with_challenges(
    candidates: torch.Tensor,
    batch_start: torch.Tensor,
    batch_word: torch.Tensor,
    batch_word_mask: torch.Tensor,
    challenge_outcome: torch.Tensor,
    challenge_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, count, generators, carrier, _ = candidates.shape
    flat = candidates.reshape(batch * count, generators, carrier, carrier)

    def expand(value: torch.Tensor) -> torch.Tensor:
        return value[:, None].expand(batch, count, *value.shape[1:]).reshape(
            batch * count, *value.shape[1:]
        )

    predictions = execute_word(
        flat,
        expand(batch_start),
        expand(batch_word),
        expand(batch_word_mask),
    ).argmax(-1)
    predictions = predictions.reshape(batch, count, -1)
    mismatches = (
        predictions.ne(challenge_outcome[:, None]) & challenge_mask[:, None]
    ).sum(-1)
    selected_index = mismatches.argmin(-1)
    batch_index = torch.arange(batch, device=candidates.device)
    selected = candidates[batch_index, selected_index]
    return selected, selected_index, mismatches


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
    batch = cpu_batch.to(device)
    source = source.to(device)
    _, row_probabilities = model.row_soft(source, batch.generator_mask, hard=False)
    candidates = binary_completion_candidates(row_probabilities, batch.generator_mask)
    selected, _, mismatches = select_with_challenges(
        candidates,
        batch.challenge_start,
        batch.challenge_word,
        batch.challenge_word_mask,
        batch.challenge_outcome,
        batch.challenge_mask,
    )
    shuffled_outcomes = batch.challenge_outcome.roll(1, 0)
    shuffled, _, _ = select_with_challenges(
        candidates,
        batch.challenge_start,
        batch.challenge_word,
        batch.challenge_word_mask,
        shuffled_outcomes,
        batch.challenge_mask,
    )
    deferred = close_presentation(row_probabilities, batch.generator_mask)
    _, row_hard = model.row_soft(source, batch.generator_mask, hard=True)

    def answer(tables: torch.Tensor) -> torch.Tensor:
        return execute_word(
            tables, batch.query_start, batch.query_word, batch.query_word_mask
        ).argmax(-1)

    challenge = execute_word(
        selected,
        batch.challenge_start,
        batch.challenge_word,
        batch.challenge_word_mask,
    ).argmax(-1)
    table_exact = selected.argmax(-1).eq(batch.true_tables.long()).all((-1, -2))
    challenge_correct = challenge.eq(batch.challenge_outcome) & batch.challenge_mask
    return {
        "family": FAMILIES[family],
        "length": length,
        "count": count,
        "seed": seed,
        "renderer_seed": renderer_seed,
        "batch_sha256": batch_sha256(cpu_batch),
        "counterexample_accuracy": answer(selected)
        .eq(batch.answer)
        .float()
        .mean()
        .item(),
        "deferred_accuracy": answer(deferred).eq(batch.answer).float().mean().item(),
        "row_soft_accuracy": answer(row_hard).eq(batch.answer).float().mean().item(),
        "shuffle_challenge_accuracy": answer(shuffled)
        .eq(batch.answer)
        .float()
        .mean()
        .item(),
        "lineage_swap_accuracy": answer(selected.roll(1, 0))
        .eq(batch.answer)
        .float()
        .mean()
        .item(),
        "challenge_exact": challenge_correct.sum()
        .div(batch.challenge_mask.sum())
        .item(),
        "selected_table_exact": table_exact.float().mean().item(),
        "candidate_contains_truth": candidates.argmax(-1)
        .eq(batch.true_tables[:, None].long())
        .all((-1, -2))
        .any(-1)
        .float()
        .mean()
        .item(),
        "selection_margin": (mismatches.sort(-1).values[:, 1] - mismatches.min(-1).values)
        .float()
        .mean()
        .item(),
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
