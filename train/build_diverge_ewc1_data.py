#!/usr/bin/env python3
"""Build immutable train/development/confirmation data for DIVERGE-EWC1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from diverge_ewc1_data import (
    BOARD_ROWS,
    CONFIRMATION_SEED,
    DEVELOPMENT_SEED,
    TRAIN_ROWS,
    TRAIN_SEED,
    generate_records,
    overlap_report,
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing EWC1 data output: {args.output}")
    args.output.mkdir(parents=True)
    train = generate_records(split="train", seed=TRAIN_SEED, count=TRAIN_ROWS)
    development = generate_records(
        split="development", seed=DEVELOPMENT_SEED, count=BOARD_ROWS
    )
    confirmation = generate_records(
        split="confirmation", seed=CONFIRMATION_SEED, count=BOARD_ROWS
    )
    paths = {
        "train": args.output / "train.jsonl",
        "development": args.output / "development.jsonl",
        "confirmation": args.output / "confirmation.jsonl",
    }
    for name, rows in (
        ("train", train),
        ("development", development),
        ("confirmation", confirmation),
    ):
        _atomic_jsonl(paths[name], rows)
    report = overlap_report(train, development, confirmation)
    report["files"] = {
        name: {
            "path": str(path),
            "sha256": sha256_path(path),
            "bytes": path.stat().st_size,
        }
        for name, path in paths.items()
    }
    if not report["all_zero"]:
        raise SystemExit("EWC1 split overlap audit failed")
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
