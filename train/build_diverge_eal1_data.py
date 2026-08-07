#!/usr/bin/env python3
"""Build immutable training and source-disjoint development data for EAL1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from diverge_eal1_data import (
    ASSESSOR_SCHEMA,
    DEVELOPMENT_EPISODES,
    DEVELOPMENT_SEED,
    PUBLIC_SCHEMA,
    REPORT_SCHEMA,
    TRAIN_ROWS,
    TRAIN_SCHEMA,
    TRAIN_SEED,
    build_development_episode,
    build_training_record,
    canonical_sha256,
    overlap_report,
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _serialized_sha256(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing EAL1 data output: {args.output}")
    args.output.mkdir(parents=True)

    training = [build_training_record(index) for index in range(TRAIN_ROWS)]
    paired = [build_development_episode(index) for index in range(DEVELOPMENT_EPISODES)]
    public = [item[0] for item in paired]
    assessor = [item[1] for item in paired]
    paths = {
        "training": args.output / "training.jsonl",
        "development_public": args.output / "development_public.jsonl",
        "development_assessor": args.output / "development_assessor.jsonl",
    }
    for name, rows in (
        ("training", training),
        ("development_public", public),
        ("development_assessor", assessor),
    ):
        _atomic_jsonl(paths[name], rows)

    audit = overlap_report(training, public)
    if any(audit[key] for key in ("source_overlap", "name_overlap", "matrix_overlap")):
        raise SystemExit("EAL1 split-overlap audit failed")
    if not audit["training_source_unique"] or not audit["development_source_unique"]:
        raise SystemExit("EAL1 source uniqueness audit failed")
    repeated_training = [build_training_record(index) for index in range(TRAIN_ROWS)]
    repeated_paired = [
        build_development_episode(index) for index in range(DEVELOPMENT_EPISODES)
    ]
    repeated_public = [item[0] for item in repeated_paired]
    repeated_assessor = [item[1] for item in repeated_paired]
    reproducible = {
        "training": _serialized_sha256(repeated_training)
        == sha256_path(paths["training"]),
        "development_public": _serialized_sha256(repeated_public)
        == sha256_path(paths["development_public"]),
        "development_assessor": _serialized_sha256(repeated_assessor)
        == sha256_path(paths["development_assessor"]),
    }
    if not all(reproducible.values()):
        raise SystemExit("EAL1 deterministic regeneration failed")
    report = {
        "schema": REPORT_SCHEMA,
        "schemas": {
            "training": TRAIN_SCHEMA,
            "development_public": PUBLIC_SCHEMA,
            "development_assessor": ASSESSOR_SCHEMA,
        },
        "seeds": {"training": TRAIN_SEED, "development": DEVELOPMENT_SEED},
        "rows": {
            "training": len(training),
            "development_public": len(public),
            "development_assessor": len(assessor),
        },
        "overlap_audit": audit,
        "deterministic_regeneration": reproducible,
        "files": {
            name: {
                "path": str(path),
                "sha256": sha256_path(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
        },
    }
    report["identity_sha256"] = canonical_sha256(report)
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "report_sha256": sha256_path(report_path),
                "files": report["files"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
