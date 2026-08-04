#!/usr/bin/env python3
"""Select the highest semantic-verifier score from each candidate group."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-product-verified-candidate-selection-v1"


class VerifiedCandidateSelectionError(RuntimeError):
    """Scored candidate shards cannot support exact selection."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label_map(path: Path | None) -> dict[tuple[str, int], bool] | None:
    if path is None:
        return None
    labels: dict[tuple[str, int], bool] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("identity_sha256") or ""), int(row["sample_index"]))
            if not key[0] or key in labels:
                raise VerifiedCandidateSelectionError("label candidate identity differs")
            labels[key] = bool(row["correct"])
    if not labels:
        raise VerifiedCandidateSelectionError("label candidate source is empty")
    return labels


def select(
    paths: list[Path], label_candidates: Path | None = None
) -> dict[str, Any]:
    labels = _label_map(label_candidates)
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                identity = str(row.get("identity_sha256") or "")
                score = row.get("verifier_score")
                if not identity or not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise VerifiedCandidateSelectionError("candidate identity or score differs")
                if labels is not None:
                    key = (identity, int(row["sample_index"]))
                    if key not in labels:
                        raise VerifiedCandidateSelectionError("candidate label is missing")
                    row["correct"] = labels[key]
                grouped.setdefault(identity, []).append(row)
    if not grouped:
        raise VerifiedCandidateSelectionError("scored candidate source is empty")
    if labels is not None:
        observed = {
            (identity, int(row["sample_index"]))
            for identity, rows in grouped.items()
            for row in rows
        }
        if observed != set(labels):
            raise VerifiedCandidateSelectionError("label and scored candidate sets differ")
    total = first = oracle = selected = 0
    results: list[dict[str, Any]] = []
    for identity, rows in grouped.items():
        rows.sort(key=lambda row: int(row["sample_index"]))
        if [int(row["sample_index"]) for row in rows] != list(range(len(rows))):
            raise VerifiedCandidateSelectionError("candidate sample indices differ")
        winner = max(rows, key=lambda row: float(row["verifier_score"]))
        total += 1
        first += int(bool(rows[0]["correct"]))
        oracle += int(any(bool(row["correct"]) for row in rows))
        selected += int(bool(winner["correct"]))
        results.append(
            {
                "identity_sha256": identity,
                "task": winner["task"],
                "selected_sample_index": int(winner["sample_index"]),
                "selected_score": float(winner["verifier_score"]),
                "selected_prediction": winner.get("prediction"),
                "selected_completion": winner.get("completion"),
                "selected_correct": bool(winner["correct"]),
            }
        )
    return {
        "schema": SCHEMA,
        "selector": "counterbalanced_frozen_semantic_verifier_v1",
        "selector_reads_gold": False,
        "total": total,
        "first_correct": first,
        "first_accuracy": first / total,
        "oracle_correct": oracle,
        "oracle_accuracy": oracle / total,
        "selected_correct": selected,
        "selected_accuracy": selected / total,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-candidates", type=Path, action="append", required=True)
    parser.add_argument("--label-candidates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = select(args.scored_candidates, args.label_candidates)
    report["scored_candidate_paths"] = [str(path.resolve()) for path in args.scored_candidates]
    report["scored_candidate_sha256"] = [_sha256(path) for path in args.scored_candidates]
    if args.label_candidates is not None:
        report["label_candidates"] = str(args.label_candidates.resolve())
        report["label_candidates_sha256"] = _sha256(args.label_candidates)
    if args.output.exists():
        raise VerifiedCandidateSelectionError(f"refusing existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(
        f"[verified-selection] selected={report['selected_correct']}/"
        f"{report['total']} first={report['first_correct']} "
        f"oracle={report['oracle_correct']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
