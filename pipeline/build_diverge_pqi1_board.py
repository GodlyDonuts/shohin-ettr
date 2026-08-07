#!/usr/bin/env python3
"""Materialize the sealed PQI1 confirmation board and overlap receipts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from diverge_pqi1_data import (
    PQI1_BOARD_ROWS,
    PQI1_BOARD_SEED,
    PQI1_NAMES,
    augment_board,
)
from diverge_rrg1_data import validate_training_record
from diverge_tfs1_data import FAULT_LINES, WORLDS, generate_board


SCHEMA = "shohin-diverge-pqi1-data-report-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise RuntimeError(f"input hash differs: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


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
    parser.add_argument("--query-training", type=Path, required=True)
    parser.add_argument("--query-training-sha256", required=True)
    parser.add_argument("--prior-board", type=Path, action="append", required=True)
    parser.add_argument("--prior-board-sha256", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=PQI1_BOARD_SEED)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing PQI1 output: {args.output}")
    if args.seed != PQI1_BOARD_SEED:
        raise SystemExit("PQI1 frozen board seed differs")
    if len(args.prior_board) != len(args.prior_board_sha256):
        raise SystemExit("PQI1 prior board receipts differ")

    training = _load_jsonl(args.query_training, args.query_training_sha256)
    for row in training:
        validate_training_record(row)
        if row["stage"] != "QUERY":
            raise SystemExit("PQI1 query training stage differs")
    prior_rows = []
    for path, expected in zip(args.prior_board, args.prior_board_sha256, strict=True):
        prior_rows.extend(_load_jsonl(path, expected))

    board = augment_board(
        generate_board(PQI1_BOARD_ROWS, PQI1_BOARD_SEED, name_bank=PQI1_NAMES),
        seed=PQI1_BOARD_SEED,
    )
    training_texts = {str(row["source_text"]) for row in training}
    training_symbols = {
        str(symbol) for row in training for symbol in row["symbols"]
    }
    prior_sources = {str(row["tfs1"]["source"]) for row in prior_rows}
    prior_queries = {
        str(item["source_text"])
        for row in prior_rows
        for item in row["natural_queries"].values()
    }
    prior_identities = {str(row["identity_sha256"]) for row in prior_rows}
    prior_symbols = {
        str(symbol) for row in prior_rows for symbol in row["tfs1"]["symbols"]
    }
    sources = {str(row["tfs1"]["source"]) for row in board}
    queries = {
        str(item["source_text"])
        for row in board
        for item in row["natural_queries"].values()
    }
    identities = {str(row["identity_sha256"]) for row in board}
    symbols = {str(symbol) for row in board for symbol in row["tfs1"]["symbols"]}
    overlap = {
        "source_with_prior": len(sources & prior_sources),
        "query_with_training": len(queries & training_texts),
        "query_with_prior": len(queries & prior_queries),
        "identity_with_prior": len(identities & prior_identities),
        "symbols_with_training": len(symbols & training_symbols),
        "symbols_with_prior": len(symbols & prior_symbols),
    }
    if any(overlap.values()) or len(identities) != PQI1_BOARD_ROWS:
        raise SystemExit(f"PQI1 split integrity failed: {overlap}")

    renderer_mode: defaultdict[str, Counter[str]] = defaultdict(Counter)
    role_order = Counter()
    for row in board:
        for mode, item in row["natural_queries"].items():
            renderer_mode[str(int(item["renderer"]))][str(mode)] += 1
            role_order[str(tuple(int(value) for value in item["symbol_role_ids"]))] += 1

    args.output.mkdir(parents=True)
    board_path = args.output / "confirmation_board.jsonl"
    _atomic_jsonl(board_path, board)
    report = {
        "schema": SCHEMA,
        "seed": PQI1_BOARD_SEED,
        "rows": len(board),
        "fault_lines_per_episode": FAULT_LINES,
        "represented_worlds_per_episode": WORLDS,
        "represented_worlds_total": len(board) * WORLDS,
        "evidence_items": len(board) * FAULT_LINES,
        "query_items": len(board) * 3,
        "entity_bank": list(PQI1_NAMES),
        "renderer_mode_counts": {
            key: dict(sorted(value.items()))
            for key, value in sorted(renderer_mode.items())
        },
        "query_role_order_counts": dict(sorted(role_order.items())),
        "overlap": overlap,
        "model_score_used_for_selection": False,
        "confirmation_generated_before_training": True,
        "query_training_sha256": args.query_training_sha256,
        "prior_board_sha256": list(args.prior_board_sha256),
        "board": str(board_path),
        "board_sha256": sha256_path(board_path),
    }
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    print(json.dumps({
        "board": str(board_path),
        "board_sha256": report["board_sha256"],
        "report": str(report_path),
        "report_sha256": sha256_path(report_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
