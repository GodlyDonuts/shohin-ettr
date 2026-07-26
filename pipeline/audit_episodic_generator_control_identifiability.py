"""Audit identifiability of the deranged-support causal control."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Iterable

from source_deleted_episodic_generator_law_board import (
    GeneratedEpisode,
    build_frozen_board,
)


def _compose(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(right[value] for value in left)


def _programs(
    supports: tuple[tuple[int, ...], tuple[int, ...]],
    *,
    maximum_depth: int,
) -> tuple[tuple[int, ...], ...]:
    identity = tuple(range(len(supports[0])))
    programs = [identity]
    frontier = [identity]
    for _depth in range(maximum_depth):
        frontier = [
            _compose(program, support)
            for program in frontier
            for support in supports
        ]
        programs.extend(frontier)
    return tuple(programs)


def _derange_first_support(
    supports: tuple[tuple[int, ...], tuple[int, ...]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    first = tuple(
        1 if target == 0 else 0 if target == 1 else target
        for target in supports[0]
    )
    return first, supports[1]


def _matching_programs(
    programs: Iterable[tuple[int, ...]],
    *,
    target: tuple[int, ...],
    visible_inputs: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        program
        for program in programs
        if all(
            program[source] == target[source]
            for source in visible_inputs
        )
    )


def _classify_episode(
    row: GeneratedEpisode,
    *,
    maximum_depth: int,
) -> dict[str, object]:
    programs = _programs(
        _derange_first_support(
            row.supervisor.support_transition
        ),
        maximum_depth=maximum_depth,
    )
    targets: list[dict[str, object]] = []
    for target, visible_inputs in zip(
        row.supervisor.target_transition,
        row.supervisor.target_visible_inputs,
        strict=True,
    ):
        matching = _matching_programs(
            programs,
            target=target,
            visible_inputs=visible_inputs,
        )
        unique_maps = set(matching)
        targets.append(
            {
                "matching_programs": len(matching),
                "matching_unique_maps": len(unique_maps),
                "true_map_survives": target in unique_maps,
            }
        )
    unique_counts = [
        int(target["matching_unique_maps"])
        for target in targets
    ]
    if any(count == 0 for count in unique_counts):
        classification = "contradictory"
    elif all(count == 1 for count in unique_counts):
        classification = (
            "identified_correct"
            if all(
                bool(target["true_map_survives"])
                for target in targets
            )
            else "identified_wrong"
        )
    else:
        classification = "ambiguous"
    return {
        "cardinality": row.supervisor.cardinality,
        "cell": row.supervisor.cell,
        "classification": classification,
        "family": row.supervisor.family,
        "targets": targets,
    }


def audit_deranged_support_identifiability(
    *,
    seed: int,
    maximum_depth: int = 6,
) -> dict[str, object]:
    development = tuple(
        row
        for row in build_frozen_board(seed=seed)
        if row.supervisor.split == "development"
    )
    episodes = [
        _classify_episode(
            row,
            maximum_depth=maximum_depth,
        )
        for row in development
    ]
    classifications = Counter(
        str(episode["classification"])
        for episode in episodes
    )
    payload = {
        "abstract_interface_zero_seal_possible": (
            classifications["identified_wrong"] == 0
        ),
        "classification_counts": dict(
            sorted(classifications.items())
        ),
        "development_rows": len(development),
        "episodes": episodes,
        "maximum_depth": maximum_depth,
        "raw_source_redundancy_available": True,
        "seed": seed,
        "syntactic_programs": sum(
            2**depth
            for depth in range(maximum_depth + 1)
        ),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return {
        **payload,
        "receipt_sha256": sha256(canonical).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--maximum-depth", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_deranged_support_identifiability(
        seed=args.seed,
        maximum_depth=args.maximum_depth,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="ascii")
    print(payload, end="")


if __name__ == "__main__":
    main()
