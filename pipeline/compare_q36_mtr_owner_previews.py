#!/usr/bin/env python3
"""Compare two Q36 owner trajectories on their exact shared identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

INPUT_SCHEMA = "shohin-q36-mtr-draft-preview-v1"
OUTPUT_SCHEMA = "shohin-q36-mtr-owner-paired-comparison-v1"
TASKS = ("math500", "bbh_logic", "mbpp")


class Q36MTROwnerPreviewComparisonError(RuntimeError):
    """Raised when owner preview outcomes cannot be paired exactly."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_reports(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], list[dict]]:
    if not paths:
        raise Q36MTROwnerPreviewComparisonError("owner preview reports are absent")
    resolved = [path.resolve(strict=True) for path in paths]
    if len(resolved) != len(set(resolved)):
        raise Q36MTROwnerPreviewComparisonError("duplicate owner preview report")
    outcomes: dict[str, dict[str, Any]] = {}
    receipts = []
    for path in resolved:
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise Q36MTROwnerPreviewComparisonError(
                f"unreadable owner preview report: {path}"
            ) from error
        rows = report.get("outcomes") if isinstance(report, dict) else None
        if (
            report.get("schema") != INPUT_SCHEMA
            or report.get("status") != "complete"
            or report.get("split") != "development"
            or not isinstance(rows, list)
            or report.get("rows") != len(rows)
        ):
            raise Q36MTROwnerPreviewComparisonError("owner preview report differs")
        for row in rows:
            identity = row.get("identity_sha256") if isinstance(row, dict) else None
            if (
                not isinstance(identity, str)
                or len(identity) != 64
                or identity in outcomes
                or row.get("task") not in TASKS
                or not isinstance(row.get("correct"), bool)
            ):
                raise Q36MTROwnerPreviewComparisonError("owner preview outcome differs")
            outcomes[identity] = row
        receipts.append(
            {"path": str(path), "sha256": sha256_file(path), "rows": len(rows)}
        )
    return outcomes, receipts


def _mcnemar_two_sided(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if not discordant:
        return 1.0
    lower = min(first_only, second_only)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def compare(
    first_paths: list[Path],
    second_paths: list[Path],
    *,
    first_label: str,
    second_label: str,
) -> dict[str, Any]:
    if not first_label or not second_label or first_label == second_label:
        raise Q36MTROwnerPreviewComparisonError("owner labels differ")
    first, first_receipts = _load_reports(first_paths)
    second, second_receipts = _load_reports(second_paths)
    shared = sorted(set(first) & set(second))
    if not shared:
        raise Q36MTROwnerPreviewComparisonError("owner previews do not overlap")
    if any(first[identity]["task"] != second[identity]["task"] for identity in shared):
        raise Q36MTROwnerPreviewComparisonError("paired owner task differs")

    first_only = sum(
        first[identity]["correct"] and not second[identity]["correct"]
        for identity in shared
    )
    second_only = sum(
        second[identity]["correct"] and not first[identity]["correct"]
        for identity in shared
    )
    both_correct = sum(
        first[identity]["correct"] and second[identity]["correct"]
        for identity in shared
    )
    both_wrong = len(shared) - first_only - second_only - both_correct
    first_correct = first_only + both_correct
    second_correct = second_only + both_correct
    oracle = first_only + second_only + both_correct
    domains = {}
    for task in TASKS:
        identities = [
            identity for identity in shared if first[identity]["task"] == task
        ]
        domains[task] = {
            "rows": len(identities),
            "first_correct": sum(first[identity]["correct"] for identity in identities),
            "second_correct": sum(
                second[identity]["correct"] for identity in identities
            ),
            "oracle_correct": sum(
                first[identity]["correct"] or second[identity]["correct"]
                for identity in identities
            ),
        }
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "complete",
        "interpretation": "engineering_owner_complementarity_not_matched_gate",
        "first_label": first_label,
        "second_label": second_label,
        "first_total_identities": len(first),
        "second_total_identities": len(second),
        "shared_identities": len(shared),
        "first_correct": first_correct,
        "second_correct": second_correct,
        "first_accuracy": first_correct / len(shared),
        "second_accuracy": second_correct / len(shared),
        "paired_cells": {
            "both_wrong": both_wrong,
            "first_only_correct": first_only,
            "second_only_correct": second_only,
            "both_correct": both_correct,
        },
        "discordant_identities": first_only + second_only,
        "mcnemar_exact_two_sided_p": _mcnemar_two_sided(first_only, second_only),
        "oracle_correct": oracle,
        "oracle_accuracy": oracle / len(shared),
        "oracle_gain_over_best_count": oracle - max(first_correct, second_correct),
        "oracle_gain_over_best_points": 100.0
        * (oracle - max(first_correct, second_correct))
        / len(shared),
        "domains": domains,
        "inputs": {"first": first_receipts, "second": second_receipts},
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTROwnerPreviewComparisonError("comparison output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-report", action="append", type=Path, required=True)
    parser.add_argument("--second-report", action="append", type=Path, required=True)
    parser.add_argument("--first-label", required=True)
    parser.add_argument("--second-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = compare(
        args.first_report,
        args.second_report,
        first_label=args.first_label,
        second_label=args.second_label,
    )
    _atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
