#!/usr/bin/env python3
"""Materialize the one frozen DIVERGE-IEM1 query and confirmation board."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from diverge_iem1_data import (
    BOARD_ROWS,
    BOARD_SEED,
    QUERY_TRAIN_ROWS,
    TRAIN_SEED,
    augment_confirmation_board,
    generate_query_training_records,
    validate_board_row,
    validate_query_training_record,
)
from diverge_nve1_data import validate_training_record
from diverge_tfs1_data import generate_board


SCHEMA = "shohin-diverge-iem1-data-report-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError(f"input hash differs: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            validate_training_record(row)
            rows.append(row)
    return rows


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-training", type=Path, required=True)
    parser.add_argument("--evidence-training-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--board-seed", type=int, default=BOARD_SEED)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing IEM1 output: {args.output}")
    if args.train_seed != TRAIN_SEED or args.board_seed != BOARD_SEED:
        raise SystemExit("IEM1 frozen data seeds differ")

    evidence_training = _load_jsonl(
        args.evidence_training,
        args.evidence_training_sha256,
    )
    queries = generate_query_training_records()
    typed_board = generate_board(BOARD_ROWS, BOARD_SEED)
    board = augment_confirmation_board(typed_board)
    for row in queries:
        validate_query_training_record(row)
    for row in board:
        validate_board_row(row)

    evidence_train_text = {str(row["source_text"]) for row in evidence_training}
    query_train_text = {str(row["source_text"]) for row in queries}
    evidence_confirmation = [
        str(item["source_text"]) for row in board for item in row["natural_evidence"]
    ]
    query_confirmation = [
        str(item["source_text"])
        for row in board
        for item in row["natural_queries"].values()
    ]
    evidence_overlap = len(evidence_train_text & set(evidence_confirmation))
    query_overlap = len(query_train_text & set(query_confirmation))
    identities = [str(row["identity_sha256"]) for row in board]
    if evidence_overlap or query_overlap or len(set(identities)) != BOARD_ROWS:
        raise SystemExit("IEM1 split integrity failed")

    args.output.mkdir(parents=True)
    query_path = args.output / "query_train.jsonl"
    board_path = args.output / "confirmation_board.jsonl"
    report_path = args.output / "report.json"
    _atomic_jsonl(query_path, queries)
    _atomic_jsonl(board_path, board)
    report = {
        "schema": SCHEMA,
        "training_seed": TRAIN_SEED,
        "confirmation_seed": BOARD_SEED,
        "query_training_rows": QUERY_TRAIN_ROWS,
        "query_training_renderers": dict(
            sorted(Counter(int(row["renderer"]) for row in queries).items())
        ),
        "confirmation_rows": BOARD_ROWS,
        "confirmation_evidence_items": BOARD_ROWS * 12,
        "confirmation_query_items": BOARD_ROWS * 3,
        "fault_lines_per_episode": 12,
        "worlds_per_episode": 4096,
        "total_represented_worlds": BOARD_ROWS * 4096,
        "evidence_training": str(args.evidence_training),
        "evidence_training_sha256": args.evidence_training_sha256,
        "query_training_path": str(query_path),
        "query_training_sha256": sha256_path(query_path),
        "confirmation_path": str(board_path),
        "confirmation_sha256": sha256_path(board_path),
        "evidence_exact_sentence_overlap": evidence_overlap,
        "query_exact_sentence_overlap": query_overlap,
        "confirmation_unique_identities": len(set(identities)),
        "model_score_used_for_selection": False,
        "typed_runtime_changed": False,
    }
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "query_training": str(query_path),
                "query_training_sha256": report["query_training_sha256"],
                "confirmation": str(board_path),
                "confirmation_sha256": report["confirmation_sha256"],
                "report": str(report_path),
                "report_sha256": sha256_path(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
