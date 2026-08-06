#!/usr/bin/env python3
"""Build the frozen DIVERGE-NPW1 narrative WORLD training corpus."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from diverge_npw1_data import TRAIN_SEED, training_record_from_tol1


SCHEMA = "shohin-diverge-npw1-training-report-v1"
TRAIN_ROWS = 20_000
SOURCE_SHA256 = "d8b4af0744d3c4232c1de91989a7b6fd4dd3168f45e35c98f9495add1b52b8ba"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", default=SOURCE_SHA256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=TRAIN_ROWS)
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing NPW1 training output: {args.output}")
    if (
        args.rows != TRAIN_ROWS
        or args.seed != TRAIN_SEED
        or args.source_sha256 != SOURCE_SHA256
        or sha256_path(args.source) != SOURCE_SHA256
    ):
        raise SystemExit("NPW1 frozen training input or geometry differs")

    selected: list[dict[str, Any]] = []
    with args.source.open(encoding="utf-8") as handle:
        for line in handle:
            if len(selected) == args.rows:
                break
            selected.append(json.loads(line))
    if len(selected) != args.rows:
        raise SystemExit("NPW1 source has too few rows")
    rows = [
        training_record_from_tol1(row, index=index, seed=args.seed)
        for index, row in enumerate(selected)
    ]
    identities = [str(row["npw1_identity_sha256"]) for row in rows]
    sources = [str(row["natural_world"]["source_text"]) for row in rows]
    if len(set(identities)) != args.rows or len(set(sources)) != args.rows:
        raise SystemExit("NPW1 training identities or sources are not unique")

    args.output.mkdir(parents=True)
    training_path = args.output / "training.jsonl"
    _atomic_jsonl(training_path, rows)
    forms = Counter(
        str(event["form"])
        for row in rows
        for event in row["natural_world"]["events"]
    )
    report: dict[str, object] = {
        "schema": SCHEMA,
        "seed": args.seed,
        "rows": len(rows),
        "source": str(args.source),
        "source_sha256": SOURCE_SHA256,
        "training": str(training_path),
        "training_sha256": sha256_path(training_path),
        "source_bytes": sum(len(value.encode("ascii")) for value in sources),
        "events": sum(forms.values()),
        "event_forms": dict(sorted(forms.items())),
        "identity_count": len(set(identities)),
        "source_count": len(set(sources)),
        "model_score_used_for_selection": False,
    }
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "training": str(training_path),
                "training_sha256": report["training_sha256"],
                "report": str(report_path),
                "report_sha256": sha256_path(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
