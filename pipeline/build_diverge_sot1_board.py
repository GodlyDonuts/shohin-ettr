#!/usr/bin/env python3
"""Materialize the fresh DIVERGE-SOT1 confirmation board and overlap audit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from diverge_iem1_data import validate_query_training_record
from diverge_nve1_data import validate_training_record
from diverge_sot1_data import (
    SOT1_BOARD_ROWS,
    SOT1_BOARD_SEED,
    augment_sot1_board,
    validate_sot1_board_row,
)
from diverge_tfs1_data import generate_board


SCHEMA = "shohin-diverge-sot1-data-report-v1"


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
            rows.append(json.loads(line))
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


def _texts(rows: list[dict[str, Any]], field: str) -> set[str]:
    return {str(row[field]) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-training", type=Path, required=True)
    parser.add_argument("--evidence-training-sha256", required=True)
    parser.add_argument("--query-training", type=Path, required=True)
    parser.add_argument("--query-training-sha256", required=True)
    parser.add_argument("--prior-confirmation", type=Path, required=True)
    parser.add_argument("--prior-confirmation-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--board-seed", type=int, default=SOT1_BOARD_SEED)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing SOT1 output: {args.output}")
    if args.board_seed != SOT1_BOARD_SEED:
        raise SystemExit("SOT1 frozen board seed differs")

    evidence_training = _load_jsonl(
        args.evidence_training, args.evidence_training_sha256
    )
    query_training = _load_jsonl(args.query_training, args.query_training_sha256)
    prior = _load_jsonl(args.prior_confirmation, args.prior_confirmation_sha256)
    for row in evidence_training:
        validate_training_record(row)
    for row in query_training:
        validate_query_training_record(row)

    board = augment_sot1_board(
        generate_board(SOT1_BOARD_ROWS, SOT1_BOARD_SEED),
        seed=SOT1_BOARD_SEED,
    )
    for row in board:
        validate_sot1_board_row(row)

    evidence_train_text = _texts(evidence_training, "source_text")
    query_train_text = _texts(query_training, "source_text")
    prior_sources = {str(row["tfs1"]["source"]) for row in prior}
    prior_evidence = {
        str(item["source_text"]) for row in prior for item in row["natural_evidence"]
    }
    prior_queries = {
        str(item["source_text"])
        for row in prior
        for item in row["natural_queries"].values()
    }
    prior_identities = {str(row["identity_sha256"]) for row in prior}
    board_sources = {str(row["tfs1"]["source"]) for row in board}
    board_evidence = [
        str(item["source_text"]) for row in board for item in row["natural_evidence"]
    ]
    board_queries = [
        str(item["source_text"])
        for row in board
        for item in row["natural_queries"].values()
    ]
    board_identities = [str(row["identity_sha256"]) for row in board]
    overlaps = {
        "source_with_prior": len(board_sources & prior_sources),
        "evidence_with_training": len(set(board_evidence) & evidence_train_text),
        "evidence_with_prior": len(set(board_evidence) & prior_evidence),
        "query_with_training": len(set(board_queries) & query_train_text),
        "query_with_prior": len(set(board_queries) & prior_queries),
        "identity_with_prior": len(set(board_identities) & prior_identities),
    }
    fatal = (
        overlaps["source_with_prior"]
        or overlaps["query_with_training"]
        or overlaps["query_with_prior"]
        or overlaps["identity_with_prior"]
        or len(set(board_identities)) != SOT1_BOARD_ROWS
    )
    if fatal:
        raise SystemExit(f"SOT1 split integrity failed: {overlaps}")

    args.output.mkdir(parents=True)
    board_path = args.output / "confirmation_board.jsonl"
    report_path = args.output / "report.json"
    _atomic_jsonl(board_path, board)
    report = {
        "schema": SCHEMA,
        "confirmation_seed": SOT1_BOARD_SEED,
        "confirmation_rows": SOT1_BOARD_ROWS,
        "confirmation_evidence_items": SOT1_BOARD_ROWS * 12,
        "confirmation_query_items": SOT1_BOARD_ROWS * 3,
        "fault_lines_per_episode": 12,
        "worlds_per_episode": 4096,
        "total_represented_worlds": SOT1_BOARD_ROWS * 4096,
        "query_renderers": dict(
            sorted(
                Counter(
                    int(item["renderer"])
                    for row in board
                    for item in row["natural_queries"].values()
                ).items()
            )
        ),
        "evidence_training": str(args.evidence_training),
        "evidence_training_sha256": args.evidence_training_sha256,
        "query_training": str(args.query_training),
        "query_training_sha256": args.query_training_sha256,
        "prior_confirmation": str(args.prior_confirmation),
        "prior_confirmation_sha256": args.prior_confirmation_sha256,
        "confirmation": str(board_path),
        "confirmation_sha256": sha256_path(board_path),
        "overlaps": overlaps,
        "confirmation_unique_identities": len(set(board_identities)),
        "confirmation_unique_queries": len(set(board_queries)),
        "confirmation_unique_evidence": len(set(board_evidence)),
        "model_score_used_for_selection": False,
        "typed_runtime_changed": False,
    }
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
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
