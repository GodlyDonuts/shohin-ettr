#!/usr/bin/env python3
"""Audit whether natural wrong/correct trajectories admit useful pointer edits.

The audit is deliberately model-free.  It considers only pairs with exactly one
correct candidate and measures how much of the correct trajectory could be
copied from the wrong trajectory.  Holdout rows are counted but never scored.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


SCHEMA = "shohin-natural-edit-locality-audit-v1"
DEFAULT_ALLOWED_SPLITS = ("train", "development")


class EditLocalityAuditError(ValueError):
    """Raised when the pair corpus violates the audit contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)) if values else 0.0,
        "p50": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "p99": _quantile(values, 0.99),
        "max": float(max(values)) if values else 0.0,
    }


def _common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _common_suffix_length(left: str, right: str, prefix: int) -> int:
    limit = min(len(left), len(right)) - prefix
    index = 0
    while index < limit and left[-1 - index] == right[-1 - index]:
        index += 1
    return index


def edit_locality(wrong: str, correct: str) -> dict[str, float | int]:
    if not isinstance(wrong, str) or not isinstance(correct, str):
        raise EditLocalityAuditError("candidate completions must be strings")
    prefix = _common_prefix_length(wrong, correct)
    suffix = _common_suffix_length(wrong, correct, prefix)
    target_length = len(correct)
    single_splice_copied = prefix + suffix

    matcher = difflib.SequenceMatcher(a=wrong, b=correct, autojunk=False)
    matching_blocks = [block for block in matcher.get_matching_blocks() if block.size]
    multi_span_copied = sum(block.size for block in matching_blocks)

    return {
        "wrong_characters": len(wrong),
        "correct_characters": target_length,
        "common_prefix_characters": prefix,
        "common_suffix_characters": suffix,
        "single_splice_replacement_characters": target_length - single_splice_copied,
        "single_splice_copy_fraction": (
            single_splice_copied / target_length if target_length else 1.0
        ),
        "multi_span_copy_characters": multi_span_copied,
        "multi_span_copy_fraction": (
            multi_span_copied / target_length if target_length else 1.0
        ),
        "multi_span_copy_runs": len(matching_blocks),
    }


def _iter_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise EditLocalityAuditError(
                    f"invalid JSON on line {line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise EditLocalityAuditError(f"line {line_number} is not an object")
            yield row


def audit_pairs(
    path: Path,
    *,
    allowed_splits: Sequence[str] = DEFAULT_ALLOWED_SPLITS,
) -> dict[str, object]:
    allowed = frozenset(allowed_splits)
    if not allowed or "holdout" in allowed:
        raise EditLocalityAuditError("holdout scoring is forbidden")

    scored: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    domains: dict[str, Counter[str]] = defaultdict(Counter)
    row_counts: Counter[str] = Counter()
    ignored_outcomes: Counter[str] = Counter()

    for row in _iter_jsonl(path):
        split = row.get("split")
        if not isinstance(split, str):
            raise EditLocalityAuditError("row has no string split")
        row_counts[split] += 1
        if split not in allowed:
            continue

        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 2:
            raise EditLocalityAuditError("each scored row must have two candidates")
        correct = [candidate for candidate in candidates if candidate.get("correct") is True]
        wrong = [candidate for candidate in candidates if candidate.get("correct") is False]
        if len(correct) != 1 or len(wrong) != 1:
            ignored_outcomes[str(row.get("outcome_class", "unknown"))] += 1
            continue

        metric = edit_locality(
            str(wrong[0].get("completion", "")),
            str(correct[0].get("completion", "")),
        )
        scored[split].append(metric)
        domains[split][str(row.get("task", "unknown"))] += 1

    metric_names = (
        "wrong_characters",
        "correct_characters",
        "common_prefix_characters",
        "common_suffix_characters",
        "single_splice_replacement_characters",
        "single_splice_copy_fraction",
        "multi_span_copy_characters",
        "multi_span_copy_fraction",
        "multi_span_copy_runs",
    )
    summaries: dict[str, object] = {}
    for split in allowed_splits:
        rows = scored.get(split, [])
        summaries[split] = {
            "scored_exactly_one_correct_pairs": len(rows),
            "domain_counts": dict(sorted(domains[split].items())),
            "metrics": {
                name: _summary([float(row[name]) for row in rows])
                for name in metric_names
            },
        }

    return {
        "schema": SCHEMA,
        "input": str(path),
        "input_sha256": sha256_file(path),
        "allowed_splits": list(allowed_splits),
        "holdout_scored": False,
        "row_counts_metadata_only": dict(sorted(row_counts.items())),
        "ignored_scored_split_outcomes": dict(sorted(ignored_outcomes.items())),
        "splits": summaries,
        "interpretation_contract": {
            "single_splice_copy_fraction": (
                "optimistic target-character fraction retained by one prefix/suffix edit"
            ),
            "multi_span_copy_fraction": (
                "optimistic target-character fraction retained by an unrestricted ordered "
                "multi-span diff; not a learned pointer score"
            ),
        },
    }


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allowed-split",
        action="append",
        dest="allowed_splits",
        choices=("train", "development"),
    )
    args = parser.parse_args()

    report = audit_pairs(
        args.pairs,
        allowed_splits=args.allowed_splits or DEFAULT_ALLOWED_SPLITS,
    )
    payload = canonical_json_bytes(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "sha256": hashlib.sha256(payload).hexdigest()}))


if __name__ == "__main__":
    main()
