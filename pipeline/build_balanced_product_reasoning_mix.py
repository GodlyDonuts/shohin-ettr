#!/usr/bin/env python3
"""Build a deterministic, unique, group-balanced product-reasoning subset."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any


SCHEMA = "shohin-balanced-product-reasoning-mix-v1"


class BalancedMixError(RuntimeError):
    """Raised when the requested balanced mix cannot be built exactly."""


def parse_weights(value: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in value.split(","):
        if "=" not in item:
            raise argparse.ArgumentTypeError("weights must use group=fraction entries")
        group, raw_weight = item.split("=", 1)
        group = group.strip()
        if not group or group in weights:
            raise argparse.ArgumentTypeError("weight groups must be unique and nonempty")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("weight fractions must be numeric") from exc
        if not 0.0 < weight <= 1.0:
            raise argparse.ArgumentTypeError("weight fractions must be in (0, 1]")
        weights[group] = weight
    if not weights or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise argparse.ArgumentTypeError("weight fractions must sum to one")
    return weights


def _question_identity(row: dict[str, Any]) -> str:
    question = row.get("question") or row.get("problem") or row.get("prompt")
    if not question:
        raise BalancedMixError("selected row has no question")
    normalized = re.sub(r"\s+", " ", str(question)).strip().casefold()
    if not normalized:
        raise BalancedMixError("selected row has an empty normalized question")
    return hashlib.sha256(normalized.encode()).hexdigest()


def _source_counts(path: Path, groups: set[str]) -> tuple[Counter[str], str]:
    counts: Counter[str] = Counter()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            try:
                row = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise BalancedMixError("source contains a malformed JSONL row") from exc
            group = str(row.get("training_group", ""))
            if group in groups:
                counts[group] += 1
    missing = sorted(groups - set(counts))
    if missing:
        raise BalancedMixError(f"requested groups are absent: {missing}")
    return counts, digest.hexdigest()


def _quotas(
    counts: Counter[str], weights: dict[str, float], max_rows: int
) -> dict[str, int]:
    capacity = min(int(counts[group] / weight) for group, weight in weights.items())
    total = min(max_rows, capacity)
    if total <= 0:
        raise BalancedMixError("source has no balanced capacity")
    quotas = {group: int(total * weight) for group, weight in weights.items()}
    remainder = total - sum(quotas.values())
    order = sorted(
        weights,
        key=lambda group: (-(total * weights[group] - quotas[group]), group),
    )
    for group in order[:remainder]:
        quotas[group] += 1
    if any(quotas[group] > counts[group] for group in quotas):
        raise BalancedMixError("computed quota exceeds a source group")
    return quotas


def build_balanced_mix(
    source: Path,
    output: Path,
    report_path: Path,
    weights: dict[str, float],
    max_rows: int,
    seed: int,
) -> dict[str, Any]:
    if max_rows <= 0:
        raise BalancedMixError("max rows must be positive")
    if output.exists() or report_path.exists():
        raise BalancedMixError("refusing to replace an existing output")
    counts, source_sha256 = _source_counts(source, set(weights))
    quotas = _quotas(counts, weights, max_rows)
    selected: dict[str, list[dict[str, Any]]] = {group: [] for group in weights}
    seen: Counter[str] = Counter()
    generators = {
        group: random.Random(f"{seed}\0{group}")
        for group in weights
    }
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            group = str(row.get("training_group", ""))
            if group not in selected:
                continue
            seen[group] += 1
            reservoir = selected[group]
            quota = quotas[group]
            if len(reservoir) < quota:
                reservoir.append(row)
            else:
                position = generators[group].randrange(seen[group])
                if position < quota:
                    reservoir[position] = row
    rows = [row for group in sorted(selected) for row in selected[group]]
    if len(rows) != sum(quotas.values()):
        raise BalancedMixError("reservoir did not fill every group quota")
    identities = [_question_identity(row) for row in rows]
    if len(set(identities)) != len(identities):
        raise BalancedMixError("balanced subset contains duplicate questions")
    random.Random(seed).shuffle(rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
    os.replace(temporary, output)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "source": str(source.resolve()),
        "source_sha256": source_sha256,
        "source_group_counts": dict(sorted(counts.items())),
        "output": str(output.resolve()),
        "output_sha256": digest.hexdigest(),
        "selected_group_counts": dict(sorted(quotas.items())),
        "selected_rows": len(rows),
        "weights": dict(sorted(weights.items())),
        "seed": seed,
        "duplicate_questions": 0,
        "replayed_rows": 0,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_tmp = report_path.with_name(f".{report_path.name}.tmp.{os.getpid()}")
    report_tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(report_tmp, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=parse_weights)
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    report = build_balanced_mix(
        args.source,
        args.output,
        args.report,
        args.weights,
        args.max_rows,
        args.seed,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
