#!/usr/bin/env python3
"""Build the one frozen DIVERGE-NVE1 training set and confirmation board."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

from diverge_nve1_data import (
    BOARD_ROWS,
    BOARD_SEED,
    TRAIN_ROWS,
    TRAIN_SEED,
    augment_confirmation_board,
    generate_training_records,
    validate_board_row,
    validate_training_record,
)
from diverge_tfs1_data import FAULT_LINES, WORLDS, generate_board


SCHEMA = "shohin-diverge-nve1-data-report-v1"


class NVE1BuildError(RuntimeError):
    """The frozen NVE1 data cannot be materialized exactly."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    training = generate_training_records(TRAIN_ROWS, TRAIN_SEED)
    for row in training:
        validate_training_record(row)
    typed = generate_board(BOARD_ROWS, BOARD_SEED)
    confirmation = augment_confirmation_board(typed, seed=BOARD_SEED)
    for row in confirmation:
        validate_board_row(row)

    training_texts = {str(row["source_text"]) for row in training}
    confirmation_texts = {
        str(item["source_text"])
        for row in confirmation
        for item in row["natural_evidence"]
    }
    overlap = training_texts & confirmation_texts
    if overlap:
        raise NVE1BuildError("NVE1 training and confirmation sentences overlap")
    if len(training_texts) != TRAIN_ROWS:
        raise NVE1BuildError("NVE1 training statements are not unique")
    if len(confirmation_texts) != BOARD_ROWS * FAULT_LINES:
        raise NVE1BuildError("NVE1 confirmation statements are not unique")
    if len({str(row["identity_sha256"]) for row in confirmation}) != BOARD_ROWS:
        raise NVE1BuildError("NVE1 confirmation episode identities are not unique")

    train_renderers = Counter(int(row["renderer"]) for row in training)
    confirm_renderers = Counter(
        int(item["renderer"])
        for row in confirmation
        for item in row["natural_evidence"]
    )
    if set(confirm_renderers.values()) != {1024}:
        raise NVE1BuildError("NVE1 confirmation renderers are not balanced")
    report = {
        "schema": SCHEMA,
        "training_seed": TRAIN_SEED,
        "confirmation_seed": BOARD_SEED,
        "training_rows": len(training),
        "confirmation_rows": len(confirmation),
        "confirmation_evidence_items": BOARD_ROWS * FAULT_LINES,
        "fault_lines_per_episode": FAULT_LINES,
        "worlds_per_episode": WORLDS,
        "total_represented_worlds": BOARD_ROWS * WORLDS,
        "training_renderers": dict(sorted(train_renderers.items())),
        "confirmation_renderers": dict(sorted(confirm_renderers.items())),
        "training_source_bytes": sum(
            len(str(row["source_text"]).encode("ascii")) for row in training
        ),
        "confirmation_source_bytes": sum(
            len(str(item["source_text"]).encode("ascii"))
            for row in confirmation
            for item in row["natural_evidence"]
        ),
        "exact_sentence_overlap": len(overlap),
        "model_score_used_for_selection": False,
        "typed_runtime_changed": False,
    }
    return training, confirmation, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing existing NVE1 data directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    training, confirmation, report = build()
    training_path = args.output_dir / "evidence_train.jsonl"
    confirmation_path = args.output_dir / "confirmation_board.jsonl"
    report_path = args.output_dir / "report.json"
    _atomic_jsonl(training_path, training)
    _atomic_jsonl(confirmation_path, confirmation)
    report.update(
        {
            "training_path": str(training_path),
            "training_sha256": sha256_path(training_path),
            "confirmation_path": str(confirmation_path),
            "confirmation_sha256": sha256_path(confirmation_path),
        }
    )
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "training_sha256": report["training_sha256"],
                "confirmation_sha256": report["confirmation_sha256"],
                "report_sha256": sha256_path(report_path),
                "training_rows": len(training),
                "confirmation_rows": len(confirmation),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
