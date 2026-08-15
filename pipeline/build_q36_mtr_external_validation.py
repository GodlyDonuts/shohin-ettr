#!/usr/bin/env python3
"""Materialize the unused Q36 source partition for external validation.

The Q36 development work used the frozen train/development partitions only.
This builder replays the same hash split and publishes the previously unused
1,279-row partition as a label-free model view plus a separate assessor board.
It also publishes a deterministic 256-row screen selected without labels.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_pcf1_data import (
    FROZEN_CUSTODY,
    PAIR_SCHEMA,
    SPLIT_SEED,
    _assessor,
    _is_sha256,
    _iter_jsonl,
    _source_prompt,
    _validate_source_schema,
    assigned_split,
    sha256_file,
)

ROWS = 1_279
SCREEN_ROWS = 256
VALIDATION_ROWS = ROWS - SCREEN_ROWS
SCREEN_SEED = 2026081435
SOURCE_SCHEMA = "shohin-q36-mtr-external-validation-source-v1"
ASSESSOR_SCHEMA = "shohin-q36-mtr-external-validation-assessor-v1"
REPORT_SCHEMA = "shohin-q36-mtr-external-validation-report-v1"


class Q36MTRExternalValidationError(RuntimeError):
    """The external source universe, split, or output differs."""


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise Q36MTRExternalValidationError(f"refusing existing output: {path}")
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
        raise Q36MTRExternalValidationError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _screen_rank(identity: str) -> bytes:
    return hashlib.sha256(f"{SCREEN_SEED}\0{identity}".encode()).digest()


def build(
    pairs_path: Path,
    bank_paths: list[Path],
    output_root: Path,
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    if (
        not pairs_path.is_file()
        or not bank_paths
        or any(not path.is_file() for path in bank_paths)
    ):
        raise Q36MTRExternalValidationError("external validation inputs are missing")
    if output_root.exists() or output_root.is_symlink():
        raise Q36MTRExternalValidationError("external validation output exists")
    if verify_hashes:
        if sha256_file(pairs_path) != FROZEN_CUSTODY.pairs_sha256:
            raise Q36MTRExternalValidationError("pair bank SHA-256 differs")
        bank_hashes = [sha256_file(path) for path in bank_paths]
        if frozenset(bank_hashes) != FROZEN_CUSTODY.bank_sha256s:
            raise Q36MTRExternalValidationError("source bank SHA-256 set differs")
    else:
        bank_hashes = [sha256_file(path) for path in bank_paths]

    pairs: dict[str, str] = {}
    for row in _iter_jsonl(pairs_path, "Q36 external pair bank"):
        identity = row.get("identity_sha256")
        task = row.get("task")
        if (
            row.get("schema") != PAIR_SCHEMA
            or not _is_sha256(identity)
            or not isinstance(task, str)
            or identity in pairs
        ):
            raise Q36MTRExternalValidationError("external pair row differs")
        pairs[identity] = task

    source_rows: list[dict[str, Any]] = []
    assessor_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bank_path, bank_sha256 in zip(bank_paths, bank_hashes, strict=True):
        for source in _iter_jsonl(bank_path, "Q36 external source bank"):
            _validate_source_schema(source.get("schema"), bank_sha256)
            identity = source.get("identity_sha256")
            task = source.get("task")
            if (
                not _is_sha256(identity)
                or identity in seen
                or pairs.get(identity) != task
            ):
                raise Q36MTRExternalValidationError("external source binding differs")
            seen.add(identity)
            if assigned_split(identity, SPLIT_SEED) != "holdout":
                continue
            source_rows.append(
                {
                    "schema": SOURCE_SCHEMA,
                    "identity_sha256": identity,
                    "split": "external_validation",
                    "task": task,
                    "source_prompt": _source_prompt(source),
                    "runtime_fields": ["source_prompt"],
                    "supervisor_only_fields": ["task"],
                }
            )
            assessor_rows.append(
                {
                    "schema": ASSESSOR_SCHEMA,
                    "identity_sha256": identity,
                    "split": "external_validation",
                    "task": task,
                    "assessor": _assessor(source),
                }
            )
    if seen != set(pairs):
        raise Q36MTRExternalValidationError("external source universe differs")
    source_rows.sort(key=lambda row: row["identity_sha256"])
    assessor_rows.sort(key=lambda row: row["identity_sha256"])
    if (
        len(source_rows) != ROWS
        or len(assessor_rows) != ROWS
        or [row["identity_sha256"] for row in source_rows]
        != [row["identity_sha256"] for row in assessor_rows]
    ):
        raise Q36MTRExternalValidationError("external partition geometry differs")

    screen_identities = {
        row["identity_sha256"]
        for row in sorted(
            source_rows, key=lambda row: _screen_rank(row["identity_sha256"])
        )[:SCREEN_ROWS]
    }
    screen_sources = [
        row for row in source_rows if row["identity_sha256"] in screen_identities
    ]
    screen_assessors = [
        row for row in assessor_rows if row["identity_sha256"] in screen_identities
    ]
    if len(screen_sources) != SCREEN_ROWS or len(screen_assessors) != SCREEN_ROWS:
        raise Q36MTRExternalValidationError("external screen geometry differs")
    validation_sources = [
        row for row in source_rows if row["identity_sha256"] not in screen_identities
    ]
    validation_assessors = [
        row for row in assessor_rows if row["identity_sha256"] not in screen_identities
    ]
    if (
        len(validation_sources) != VALIDATION_ROWS
        or len(validation_assessors) != VALIDATION_ROWS
    ):
        raise Q36MTRExternalValidationError("external validation geometry differs")

    output_root.mkdir(parents=True)
    outputs = {
        "full_sources": (output_root / "external_sources.jsonl", source_rows),
        "full_assessors": (output_root / "external_assessors.jsonl", assessor_rows),
        "screen_sources": (output_root / "screen_sources.jsonl", screen_sources),
        "screen_assessors": (output_root / "screen_assessors.jsonl", screen_assessors),
        "validation_sources": (
            output_root / "validation_sources.jsonl",
            validation_sources,
        ),
        "validation_assessors": (
            output_root / "validation_assessors.jsonl",
            validation_assessors,
        ),
    }
    receipts: dict[str, dict[str, Any]] = {}
    for name, (path, rows) in outputs.items():
        receipts[name] = {
            "path": str(path.resolve()),
            "rows": len(rows),
            "sha256": _atomic_lines(path, rows),
        }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "split_seed": SPLIT_SEED,
        "screen_seed": SCREEN_SEED,
        "source_disjoint_from_q36_development": True,
        "development_identity_overlap": 0,
        "full_rows": ROWS,
        "screen_rows": SCREEN_ROWS,
        "validation_rows": VALIDATION_ROWS,
        "full_task_counts": dict(Counter(row["task"] for row in source_rows)),
        "screen_task_counts": dict(Counter(row["task"] for row in screen_sources)),
        "validation_task_counts": dict(
            Counter(row["task"] for row in validation_sources)
        ),
        "inputs": {
            "pairs_sha256": sha256_file(pairs_path),
            "source_bank_sha256s": sorted(bank_hashes),
        },
        "outputs": receipts,
    }
    _atomic_json(output_root / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--bank", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build(args.pairs, args.bank, args.output_root)


if __name__ == "__main__":
    main()
