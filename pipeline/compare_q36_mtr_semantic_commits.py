#!/usr/bin/env python3
"""Compare Q36 endpoints and learned semantic commits on exact shared identities."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

INPUT_SCHEMA = "shohin-q36-mtr-draft-preview-v1"
OUTPUT_SCHEMA = "shohin-q36-mtr-semantic-commit-comparison-v1"
TASKS = ("math500", "bbh_logic", "mbpp")
EXPECTED_ROWS = 1_289


class Q36MTRSemanticComparisonError(RuntimeError):
    """Semantic commit reports are incomplete, malformed, or not paired."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    if not paths:
        raise Q36MTRSemanticComparisonError("semantic comparison group is empty")
    outcomes: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    resolved = [path.resolve(strict=True) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise Q36MTRSemanticComparisonError("semantic comparison report is duplicated")
    for path in resolved:
        if path.is_symlink() or not path.is_file():
            raise Q36MTRSemanticComparisonError("semantic comparison input is linked")
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise Q36MTRSemanticComparisonError(
                "semantic comparison input is unreadable"
            ) from error
        rows = report.get("outcomes") if isinstance(report, dict) else None
        if (
            report.get("schema") != INPUT_SCHEMA
            or report.get("status") != "complete"
            or report.get("split") != "development"
            or not isinstance(rows, list)
            or report.get("rows") != len(rows)
            or report.get("correct")
            != sum(int(row.get("correct") is True) for row in rows)
        ):
            raise Q36MTRSemanticComparisonError("semantic comparison report differs")
        for row in rows:
            identity = row.get("identity_sha256") if isinstance(row, dict) else None
            if (
                not isinstance(identity, str)
                or len(identity) != 64
                or identity in outcomes
                or row.get("task") not in TASKS
                or not isinstance(row.get("correct"), bool)
            ):
                raise Q36MTRSemanticComparisonError(
                    "semantic comparison outcome differs"
                )
            outcomes[identity] = row
        receipts.append(
            {"path": str(path), "sha256": sha256_file(path), "rows": len(rows)}
        )
    if len(outcomes) != EXPECTED_ROWS:
        raise Q36MTRSemanticComparisonError("semantic comparison coverage differs")
    return outcomes, receipts


def _mcnemar(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if not discordant:
        return 1.0
    lower = min(first_only, second_only)
    tail = sum(math.comb(discordant, value) for value in range(lower + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def compare(groups: dict[str, list[Path]]) -> dict[str, Any]:
    if len(groups) < 2 or any(not label.strip() for label in groups):
        raise Q36MTRSemanticComparisonError("semantic comparison labels differ")
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    receipts: dict[str, list[dict[str, Any]]] = {}
    for label in sorted(groups):
        loaded[label], receipts[label] = _load(groups[label])
    identities = set(next(iter(loaded.values())))
    if any(set(rows) != identities for rows in loaded.values()):
        raise Q36MTRSemanticComparisonError("semantic comparison identities differ")
    if any(
        len({rows[identity]["task"] for rows in loaded.values()}) != 1
        for identity in identities
    ):
        raise Q36MTRSemanticComparisonError("semantic comparison tasks differ")
    ordered = sorted(identities)
    summaries: dict[str, dict[str, Any]] = {}
    for label, rows in loaded.items():
        correct = sum(int(rows[identity]["correct"]) for identity in ordered)
        summaries[label] = {
            "correct": correct,
            "accuracy": correct / len(ordered),
            "domains": {
                task: {
                    "rows": sum(rows[identity]["task"] == task for identity in ordered),
                    "correct": sum(
                        rows[identity]["task"] == task and rows[identity]["correct"]
                        for identity in ordered
                    ),
                }
                for task in TASKS
            },
        }
    pairwise: dict[str, dict[str, Any]] = {}
    labels = sorted(loaded)
    for first_index, first_label in enumerate(labels):
        for second_label in labels[first_index + 1 :]:
            first = loaded[first_label]
            second = loaded[second_label]
            first_only = sum(
                first[identity]["correct"] and not second[identity]["correct"]
                for identity in ordered
            )
            second_only = sum(
                second[identity]["correct"] and not first[identity]["correct"]
                for identity in ordered
            )
            both = sum(
                first[identity]["correct"] and second[identity]["correct"]
                for identity in ordered
            )
            pairwise[f"{first_label}__vs__{second_label}"] = {
                "first_label": first_label,
                "second_label": second_label,
                "both_correct": both,
                "first_only_correct": first_only,
                "second_only_correct": second_only,
                "both_wrong": len(ordered) - both - first_only - second_only,
                "first_minus_second_correct": first_only - second_only,
                "mcnemar_exact_two_sided_p": _mcnemar(first_only, second_only),
            }
    oracle = sum(
        any(rows[identity]["correct"] for rows in loaded.values())
        for identity in ordered
    )
    best_label = max(labels, key=lambda label: (summaries[label]["correct"], label))
    return {
        "schema": OUTPUT_SCHEMA,
        "status": "complete",
        "interpretation": "engineering_semantic_commit_matrix",
        "rows": len(ordered),
        "variants": summaries,
        "best_variant": best_label,
        "best_correct": summaries[best_label]["correct"],
        "oracle_correct": oracle,
        "oracle_accuracy": oracle / len(ordered),
        "oracle_gain_over_best_count": oracle - summaries[best_label]["correct"],
        "oracle_gain_over_best_points": 100.0
        * (oracle - summaries[best_label]["correct"])
        / len(ordered),
        "pairwise": pairwise,
        "inputs": receipts,
    }


def _group(values: list[str]) -> dict[str, list[Path]]:
    groups: defaultdict[str, list[Path]] = defaultdict(list)
    for value in values:
        label, separator, path = value.partition(":")
        if not separator or not label or not path:
            raise Q36MTRSemanticComparisonError(
                "group must have the form LABEL:/absolute/report.json"
            )
        groups[label].append(Path(path))
    return dict(groups)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRSemanticComparisonError("semantic comparison output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compare(_group(args.group))
    _atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
