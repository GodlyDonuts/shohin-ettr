#!/usr/bin/env python3
"""Build and hash the frozen DIVERGE-TOL1 train/dev/OOD boards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from diverge_tol1_data import generate_split, split_report


SCHEMA = "shohin-diverge-tol1-build-report-v1"
RESERVED_BIGRAMS = (("GUARD", "SWAP"), ("SWAP", "MULTIPLY"))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    return sha256_path(path)


def _body_operations(row: dict[str, object]) -> tuple[str, ...]:
    clauses = row["clauses"]
    depth = int(row["body_depth"])
    body = clauses[4 : 4 + depth]
    if len(body) != depth:
        raise RuntimeError("TOL1 body slice differs")
    return tuple(str(value["operation"]) for value in body)


def _reserved_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts = {"->".join(pair): 0 for pair in RESERVED_BIGRAMS}
    for row in rows:
        operations = _body_operations(row)
        pairs = set(zip(operations, operations[1:], strict=False))
        for pair in RESERVED_BIGRAMS:
            counts["->".join(pair)] += int(pair in pairs)
    return counts


def build(
    output_dir: Path,
    *,
    train_count: int,
    development_count: int,
    ood_count: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise RuntimeError(f"refusing existing TOL1 output: {output_dir}")
    if min(train_count, development_count, ood_count) <= 0:
        raise RuntimeError("TOL1 split sizes must be positive")
    output_dir.mkdir(parents=True)
    seeds = {"train": 2026080501, "development": 2026080502, "ood": 2026080503}
    counts = {
        "train": train_count,
        "development": development_count,
        "ood": ood_count,
    }
    boards = {
        split: generate_split(split, counts[split], seeds[split])
        for split in ("train", "development", "ood")
    }
    identity_sets = {
        split: {str(row["identity_sha256"]) for row in rows}
        for split, rows in boards.items()
    }
    overlaps = {
        "train_development": len(identity_sets["train"] & identity_sets["development"]),
        "train_ood": len(identity_sets["train"] & identity_sets["ood"]),
        "development_ood": len(identity_sets["development"] & identity_sets["ood"]),
    }
    if any(overlaps.values()):
        raise RuntimeError("TOL1 split identity overlap")
    reserved = {split: _reserved_counts(rows) for split, rows in boards.items()}
    if any(reserved["train"].values()) or not all(
        value == ood_count for value in reserved["ood"].values()
    ):
        raise RuntimeError("TOL1 reserved-composition contract differs")
    paths = {split: output_dir / f"{split}.jsonl" for split in boards}
    hashes = {
        split: _write_jsonl(paths[split], boards[split]) for split in boards
    }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "seeds": seeds,
        "counts": counts,
        "hashes": hashes,
        "paths": {split: str(path) for split, path in paths.items()},
        "split_reports": {
            split: split_report(rows) for split, rows in boards.items()
        },
        "identity_overlaps": overlaps,
        "reserved_bigram_counts": reserved,
    }
    _atomic_json(output_dir / "report.json", report)
    report["report_sha256"] = sha256_path(output_dir / "report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=24_000)
    parser.add_argument("--development-count", type=int, default=512)
    parser.add_argument("--ood-count", type=int, default=1_024)
    args = parser.parse_args()
    report = build(
        args.output_dir,
        train_count=args.train_count,
        development_count=args.development_count,
        ood_count=args.ood_count,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
