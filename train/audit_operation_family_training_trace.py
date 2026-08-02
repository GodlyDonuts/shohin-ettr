#!/usr/bin/env python3
"""Measure class-regime drift in a logged operation-family training trace."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Mapping, Sequence

from train_ettr_component_island import _canonical_bytes, _write_no_replace


SCHEMA = "shohin-ettr-operation-family-stream-drift-audit-v1"
FAMILIES = ("0", "1", "2")


class OperationFamilyTraceAuditError(ValueError):
    """A trace row or operation-family count differs from the frozen schema."""


def _histogram(row: Mapping[str, object]) -> tuple[int, int, int]:
    raw = row.get("atomic_action_counts")
    if not isinstance(raw, Mapping):
        raise OperationFamilyTraceAuditError("atomic action counts differ")
    counts = Counter({family: 0 for family in FAMILIES})
    observed = False
    for name, value in raw.items():
        if not isinstance(name, str) or not name.endswith(".effect_family"):
            continue
        if not isinstance(value, Mapping) or set(value) != set(FAMILIES):
            raise OperationFamilyTraceAuditError("effect-family counts differ")
        observed = True
        for family in FAMILIES:
            count = value[family]
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise OperationFamilyTraceAuditError("effect-family count differs")
            counts[family] += count
    if not observed or sum(counts.values()) == 0:
        raise OperationFamilyTraceAuditError("effect-family trace is empty")
    return tuple(counts[family] for family in FAMILIES)


def _distribution(histogram: Sequence[int]) -> dict[str, float]:
    total = sum(histogram)
    if total <= 0:
        raise OperationFamilyTraceAuditError("family histogram is empty")
    return {
        family: histogram[index] / total
        for index, family in enumerate(FAMILIES)
    }


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise OperationFamilyTraceAuditError("training trace is empty")
    positions = []
    losses = []
    histograms = []
    for row in rows:
        position = row.get("position")
        loss = row.get("loss")
        if (
            not isinstance(position, int)
            or isinstance(position, bool)
            or not isinstance(loss, (int, float))
            or isinstance(loss, bool)
        ):
            raise OperationFamilyTraceAuditError("training trace scalar differs")
        positions.append(position)
        losses.append(float(loss))
        histograms.append(_histogram(row))
    if any(right <= left for left, right in zip(positions, positions[1:])):
        raise OperationFamilyTraceAuditError("training trace positions differ")

    dominant = [
        max(range(len(FAMILIES)), key=lambda index: (histogram[index], -index))
        for histogram in histograms
    ]
    longest = 1
    run = 1
    for left, right in zip(dominant, dominant[1:]):
        run = run + 1 if left == right else 1
        longest = max(longest, run)
    total = tuple(sum(value[index] for value in histograms) for index in range(3))
    quarter = max(1, len(histograms) // 4)
    first = tuple(
        sum(value[index] for value in histograms[:quarter]) for index in range(3)
    )
    last = tuple(
        sum(value[index] for value in histograms[-quarter:]) for index in range(3)
    )
    presence = {
        family: sum(histogram[index] > 0 for histogram in histograms) / len(histograms)
        for index, family in enumerate(FAMILIES)
    }
    dominant_losses = {
        family: fmean(
            loss for loss, winner in zip(losses, dominant, strict=True) if winner == index
        )
        for index, family in enumerate(FAMILIES)
        if any(winner == index for winner in dominant)
    }
    incomplete = sum(sum(count > 0 for count in value) < 3 for value in histograms)
    single = sum(sum(count > 0 for count in value) == 1 for value in histograms)
    return {
        "class_presence_rate": presence,
        "dominant_class_loss_mean": dominant_losses,
        "dominant_run_max_logged_updates": longest,
        "family_distribution": _distribution(total),
        "final_logged_batch": {
            "distribution": _distribution(histograms[-1]),
            "histogram": dict(zip(FAMILIES, histograms[-1], strict=True)),
            "loss": losses[-1],
            "position": positions[-1],
        },
        "first_quarter_distribution": _distribution(first),
        "last_quarter_distribution": _distribution(last),
        "logged_updates": len(rows),
        "missing_at_least_one_family_updates": incomplete,
        "missing_at_least_one_family_rate": incomplete / len(rows),
        "single_family_updates": single,
        "single_family_rate": single / len(rows),
        "total_target_counts": dict(zip(FAMILIES, total, strict=True)),
    }


def audit_trace(path: Path, *, source_label: str) -> dict[str, object]:
    payload = path.read_bytes()
    try:
        rows = tuple(
            json.loads(line)
            for line in payload.decode("ascii").splitlines()
            if line
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperationFamilyTraceAuditError("training trace is malformed") from exc
    if not all(isinstance(row, Mapping) for row in rows):
        raise OperationFamilyTraceAuditError("training trace row differs")
    summary = summarize_rows(rows)
    return {
        "interpretation": {
            "claim_boundary": (
                "logged checkpoints only; this audit does not infer unlogged updates"
            ),
            "release_gate": (
                "fail" if summary["missing_at_least_one_family_updates"] else "pass"
            ),
            "required_repair": (
                "deterministic replay-balanced optimizer windows before capability-floor fits"
            ),
        },
        "schema": SCHEMA,
        "source": {
            "bytes": len(payload),
            "label": source_label,
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "status": "pass-diagnostic-complete",
        "summary": summary,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit_trace(args.trace, source_label=args.source_label)
    _write_no_replace(args.output, _canonical_bytes(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
