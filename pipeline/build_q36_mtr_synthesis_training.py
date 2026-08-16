#!/usr/bin/env python3
"""Build calibration-only multi-trajectory synthesis training presentations."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from hf_q36_mtr_synthesize_trajectories import synthesis_prompt
import train_apply_q36_mtr_sparse_router as sparse

SCHEMA = "shohin-q36-mtr-synthesis-training-v1"
REPORT_SCHEMA = "shohin-q36-mtr-synthesis-training-report-v1"
PRESENTATIONS = 9_655


class Q36MTRSynthesisTrainingError(RuntimeError):
    """Synthesis-training source, target, or presentation geometry differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _target(row: dict[str, Any]) -> tuple[str, str, int]:
    correct = [
        (lineage, candidate)
        for lineage, candidate in zip(sparse.LINEAGES, row["candidates"], strict=True)
        if candidate["correct"]
    ]
    if not correct:
        raise Q36MTRSynthesisTrainingError("synthesis row has no verified target")
    lineage, candidate = min(
        correct,
        key=lambda item: (
            item[1]["generated_tokens"],
            len(item[1]["completion"]),
            item[0],
        ),
    )
    return str(candidate["completion"]), lineage, len(correct)


def build_presentations(
    rows: list[dict[str, Any]], *, total: int = PRESENTATIONS
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if total <= 0:
        raise Q36MTRSynthesisTrainingError("synthesis presentation target differs")
    weighted: list[dict[str, Any]] = []
    identities: set[str] = set()
    target_lineages: Counter[str] = Counter()
    correctness_counts: Counter[int] = Counter()
    for row in sorted(rows, key=lambda value: value["identity_sha256"]):
        if not any(candidate["correct"] for candidate in row["candidates"]):
            continue
        identity = row["identity_sha256"]
        if identity in identities:
            raise Q36MTRSynthesisTrainingError("synthesis identity duplicated")
        identities.add(identity)
        target, target_lineage, correct_count = _target(row)
        question, attempt_order = synthesis_prompt(
            row["question"], identity, row["candidates"]
        )
        weight = 4 if correct_count == 1 else 2 if correct_count == 2 else 1
        base = {
            "schema": SCHEMA,
            "identity_sha256": identity,
            "task": row["task"],
            "question": question,
            "response": target,
            "target_lineage": target_lineage,
            "correct_candidates": correct_count,
            "attempt_order": attempt_order,
            "development_labels_read": 0,
        }
        weighted.extend(
            dict(base, within_identity_copy=index) for index in range(weight)
        )
        target_lineages[target_lineage] += 1
        correctness_counts[correct_count] += 1
    if not weighted or len(identities) < 100:
        raise Q36MTRSynthesisTrainingError("synthesis eligible geometry differs")
    presentations = [
        {
            **weighted[index % len(weighted)],
            "presentation_index": index,
            "presentation_cycle": index // len(weighted),
        }
        for index in range(total)
    ]
    return presentations, {
        "eligible_identities": len(identities),
        "weighted_cycle_presentations": len(weighted),
        "target_lineages": dict(sorted(target_lineages.items())),
        "correct_candidate_counts": {
            str(key): value for key, value in sorted(correctness_counts.items())
        },
        "presentations": len(presentations),
        "complete_cycles": len(presentations) // len(weighted),
        "partial_cycle_presentations": len(presentations) % len(weighted),
    }


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise Q36MTRSynthesisTrainingError("synthesis training output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRSynthesisTrainingError("synthesis training report exists")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = sparse.load_training_rows(args.training_rows)
    presentations, geometry = build_presentations(rows)
    output_sha256 = _atomic_lines(args.output, presentations)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "interpretation": "calibration_only_multi_trajectory_synthesis_training",
        "training_rows": str(args.training_rows.resolve()),
        "training_rows_sha256": sha256_file(args.training_rows),
        "source_rows": len(rows),
        "geometry": geometry,
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
        "development_labels_read": 0,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(run(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
