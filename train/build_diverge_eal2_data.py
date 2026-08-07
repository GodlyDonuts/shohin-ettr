#!/usr/bin/env python3
"""Build immutable composition-held-out data for DIVERGE-EAL2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from diverge_eal2_data import (
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
    overlap_report,
)
from diverge_eal1_data import canonical_sha256


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


def _renderer_audit(
    training: Sequence[dict[str, Any]], public: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    train_pairs = {tuple(row["renderer"][:2]) for row in training}
    development_pairs = {
        tuple(item["renderer"][:2])
        for episode in public
        for item in episode["evidence"]
    }
    return {
        "training_pairs": [list(value) for value in sorted(train_pairs)],
        "development_pairs": [list(value) for value in sorted(development_pairs)],
        "pair_overlap": len(train_pairs & development_pairs),
        "training_before_primitives": sorted({value[0] for value in train_pairs}),
        "development_before_primitives": sorted(
            {value[0] for value in development_pairs}
        ),
        "training_after_primitives": sorted({value[1] for value in train_pairs}),
        "development_after_primitives": sorted(
            {value[1] for value in development_pairs}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing EAL2 data output: {args.output}")
    args.output.mkdir(parents=True)
    training = [build_training_record(index) for index in range(TRAIN_ROWS)]
    paired = [build_development_episode(index) for index in range(DEVELOPMENT_EPISODES)]
    public = [value[0] for value in paired]
    assessor = [value[1] for value in paired]
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
    overlap = overlap_report(training, public)
    renderer = _renderer_audit(training, public)
    if any(
        overlap[key] for key in ("source_overlap", "name_overlap", "matrix_overlap")
    ):
        raise SystemExit("EAL2 split-overlap audit failed")
    if renderer["pair_overlap"] != 0 or any(
        renderer[key] != list(range(4))
        for key in (
            "training_before_primitives",
            "development_before_primitives",
            "training_after_primitives",
            "development_after_primitives",
        )
    ):
        raise SystemExit("EAL2 renderer composition audit failed")
    repeated_training = [build_training_record(index) for index in range(TRAIN_ROWS)]
    repeated = [
        build_development_episode(index) for index in range(DEVELOPMENT_EPISODES)
    ]
    reproducible = {
        "training": _serialized_sha256(repeated_training)
        == sha256_path(paths["training"]),
        "development_public": _serialized_sha256([value[0] for value in repeated])
        == sha256_path(paths["development_public"]),
        "development_assessor": _serialized_sha256([value[1] for value in repeated])
        == sha256_path(paths["development_assessor"]),
    }
    if not all(reproducible.values()):
        raise SystemExit("EAL2 deterministic regeneration failed")
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
        "overlap_audit": overlap,
        "renderer_audit": renderer,
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
