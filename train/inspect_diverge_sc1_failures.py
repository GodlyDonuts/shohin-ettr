#!/usr/bin/env python3
"""Read-only component localization for a failed DIVERGE-SC1 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Sequence

import torch
from tokenizers import Tokenizer

from diverge_sc1_neural_compiler import (
    RawSourceCompiler,
    _support_recalled,
    encode_source,
    gold_boundaries,
    gold_pairs,
    gold_role_targets,
    output_to_scores,
)
from diverge_sc1_source_compiler import (
    CompilerScores,
    OTHER,
    calibrated_scores,
    decode_joint,
    exact,
    generate_episode,
)
from frozen_pointer_backbone import load_frozen_pointer_backbone


SCHEMA = "shohin-diverge-sc1-read-only-failure-audit-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _hybrid(
    learned: CompilerScores,
    oracle: CompilerScores,
    components: frozenset[str],
) -> CompilerScores:
    return CompilerScores(
        oracle.role if "role" in components else learned.role,
        oracle.boundary if "boundary" in components else learned.boundary,
        oracle.pair if "pair" in components else learned.pair,
    )


def _new_arm() -> dict[str, int]:
    return {
        "episodes": 0,
        "exact": 0,
        "support_recalled": 0,
        "overflow": 0,
        "candidate_options": 0,
        "candidate_records": 0,
    }


def _confusion_update(
    target: Sequence[int],
    predicted: Sequence[int],
    confusion: list[list[int]],
) -> None:
    if len(target) != len(predicted):
        raise ValueError("role target and prediction lengths differ")
    for truth, guess in zip(target, predicted, strict=True):
        confusion[truth][guess] += 1


def _quantiles(values: Iterable[float]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0}

    def select(fraction: float) -> float:
        return ordered[round(fraction * (len(ordered) - 1))]

    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "p10": select(0.10),
        "median": select(0.50),
        "p90": select(0.90),
        "maximum": ordered[-1],
    }


def _rates(row: dict[str, int]) -> dict[str, float]:
    episodes = row["episodes"]
    return {
        "exact": row["exact"] / episodes,
        "support_recalled": row["support_recalled"] / episodes,
        "overflow": row["overflow"] / episodes,
        "mean_candidate_options": row["candidate_options"] / episodes,
        "mean_candidate_records": row["candidate_records"] / episodes,
    }


def inspect(
    model: RawSourceCompiler,
    *,
    count: int,
    seed: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    arms = {
        "learned": frozenset(),
        "oracle_role": frozenset(("role",)),
        "oracle_boundary": frozenset(("boundary",)),
        "oracle_pair": frozenset(("pair",)),
        "oracle_role_boundary": frozenset(("role", "boundary")),
        "oracle_role_pair": frozenset(("role", "pair")),
        "oracle_boundary_pair": frozenset(("boundary", "pair")),
        "oracle_all": frozenset(("role", "boundary", "pair")),
    }
    cohorts: dict[str, dict[str, dict[str, int]]] = {}
    role_confusion = [[0 for _ in range(model.role_head.out_features)] for _ in range(model.role_head.out_features)]
    role_semantic = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    boundary_confusion = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    pair_confusion = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    positive_pair_scores: list[float] = []
    negative_pair_scores: list[float] = []
    predicted_active_positions = 0
    gold_active_positions = 0

    model.eval()
    with torch.no_grad():
        for cohort_index, cohort in enumerate(
            ("train", "lexical_shift", "renderer_shift", "composition_shift")
        ):
            by_arm = {name: _new_arm() for name in arms}
            cohorts[cohort] = by_arm
            cohort_seed = seed + cohort_index * 100_000
            for start in range(0, count, batch_size):
                episodes = [
                    generate_episode(seed=cohort_seed + index, cohort=cohort)
                    for index in range(start, min(count, start + batch_size))
                ]
                output = model(
                    [encode_source(model.tokenizer, episode.tokens) for episode in episodes],
                    device,
                )
                for row, episode in enumerate(episodes):
                    learned = output_to_scores(output, row, len(episode.tokens))
                    oracle = calibrated_scores(
                        episode,
                        seed=cohort_seed * 31 + start + row,
                    )

                    target_roles = gold_role_targets(episode)
                    predicted_roles = tuple(
                        max(range(len(values)), key=values.__getitem__)
                        for values in learned.role
                    )
                    _confusion_update(target_roles, predicted_roles, role_confusion)
                    for truth, guess in zip(target_roles, predicted_roles, strict=True):
                        truth_active = truth != OTHER
                        guess_active = guess != OTHER
                        key = (
                            "tp" if truth_active and guess_active else
                            "fp" if not truth_active and guess_active else
                            "fn" if truth_active and not guess_active else
                            "tn"
                        )
                        role_semantic[key] += 1

                    target_boundaries = gold_boundaries(episode)
                    for truth, score in zip(target_boundaries, learned.boundary, strict=True):
                        guess = score > 0
                        key = (
                            "tp" if truth and guess else
                            "fp" if not truth and guess else
                            "fn" if truth and not guess else
                            "tn"
                        )
                        boundary_confusion[key] += 1

                    positive_pairs, gold_active = gold_pairs(episode)
                    margin_active = {
                        position
                        for position, values in enumerate(learned.role)
                        if max(values[1:]) > values[OTHER]
                    }
                    predicted_active_positions += len(margin_active)
                    gold_active_positions += len(gold_active)
                    pair_map = learned.pair_map()
                    universe = sorted(margin_active | gold_active)
                    for offset, left in enumerate(universe):
                        for right in universe[offset + 1 :]:
                            key_pair = tuple(sorted((left, right)))
                            truth = key_pair in positive_pairs
                            score = float(pair_map.get((left, right), 0.0))
                            guess = left in margin_active and right in margin_active and score > 0
                            key = (
                                "tp" if truth and guess else
                                "fp" if not truth and guess else
                                "fn" if truth and not guess else
                                "tn"
                            )
                            pair_confusion[key] += 1
                            (positive_pair_scores if truth else negative_pair_scores).append(score)

                    for name, components in arms.items():
                        receipt = decode_joint(
                            episode.tokens,
                            _hybrid(learned, oracle, components),
                        )
                        target = by_arm[name]
                        target["episodes"] += 1
                        target["exact"] += int(exact(episode, receipt))
                        target["support_recalled"] += int(
                            _support_recalled(episode, receipt)
                        )
                        target["overflow"] += int(receipt.overflow)
                        target["candidate_options"] += receipt.candidate_options
                        target["candidate_records"] += receipt.candidate_records

    total_episodes = count * len(cohorts)
    return {
        "count_per_cohort": count,
        "episodes": total_episodes,
        "cohorts": {
            cohort: {name: _rates(row) for name, row in rows.items()}
            for cohort, rows in cohorts.items()
        },
        "role_confusion": role_confusion,
        "role_semantic": role_semantic,
        "boundary_confusion": boundary_confusion,
        "pair_confusion": pair_confusion,
        "positive_pair_scores": _quantiles(positive_pair_scores),
        "negative_pair_scores": _quantiles(negative_pair_scores),
        "mean_predicted_active_positions": predicted_active_positions / total_episodes,
        "mean_gold_active_positions": gold_active_positions / total_episodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=202608056400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--pair-width", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    if arguments.count <= 0 or arguments.batch_size <= 0:
        raise ValueError("audit count and batch size must be positive")

    torch.set_num_threads(arguments.threads)
    device = torch.device(arguments.device)
    backbone, _, receipt = load_frozen_pointer_backbone(arguments.base, device=device)
    tokenizer = Tokenizer.from_file(str(arguments.tokenizer))
    model = RawSourceCompiler(
        backbone,
        tokenizer,
        layer=arguments.layer,
        width=arguments.width,
        pair_width=arguments.pair_width,
    ).to(device)
    checkpoint = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if unexpected or any(not name.startswith("backbone.") for name in missing):
        raise ValueError("SC1 checkpoint does not match the frozen architecture")

    report = {
        "schema": SCHEMA,
        "inputs": {
            "base_sha256": sha256_file(arguments.base),
            "tokenizer_sha256": sha256_file(arguments.tokenizer),
            "checkpoint_sha256": sha256_file(arguments.checkpoint),
            "base_step": receipt.base_step,
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
        },
        "audit": inspect(
            model,
            count=arguments.count,
            seed=arguments.seed,
            batch_size=arguments.batch_size,
            device=device,
        ),
    }
    _atomic_json(arguments.output, report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
