#!/usr/bin/env python3
"""Materialize the fresh DIVERGE-CCR1 board and overlap receipts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from diverge_ccr1_data import (
    CCR1_BOARD_ROWS,
    CCR1_BOARD_SEED,
    CCR1_NAMES,
    augment_board,
)
from diverge_iem1_data import validate_query_training_record
from diverge_nve1_data import validate_training_record
from diverge_tfs1_data import FAULT_LINES, WORLDS, generate_board


SCHEMA = "shohin-diverge-ccr1-data-report-v1"


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
    parser.add_argument("--evidence-training", type=Path, required=True)
    parser.add_argument("--evidence-training-sha256", required=True)
    parser.add_argument("--query-training", type=Path, required=True)
    parser.add_argument("--query-training-sha256", required=True)
    parser.add_argument("--prior-srp1", type=Path, required=True)
    parser.add_argument("--prior-srp1-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=CCR1_BOARD_SEED)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing CCR1 output: {args.output}")
    if args.seed != CCR1_BOARD_SEED:
        raise SystemExit("CCR1 frozen board seed differs")

    evidence_training = _load_jsonl(args.evidence_training, args.evidence_training_sha256)
    query_training = _load_jsonl(args.query_training, args.query_training_sha256)
    prior = _load_jsonl(args.prior_srp1, args.prior_srp1_sha256)
    for row in evidence_training:
        validate_training_record(row)
    for row in query_training:
        validate_query_training_record(row)

    board = augment_board(
        generate_board(CCR1_BOARD_ROWS, CCR1_BOARD_SEED, name_bank=CCR1_NAMES),
        seed=CCR1_BOARD_SEED,
    )
    evidence_texts = {str(row["source_text"]) for row in evidence_training}
    query_texts = {str(row["source_text"]) for row in query_training}
    training_symbols = {
        str(symbol)
        for row in (*evidence_training, *query_training)
        for symbol in row["symbols"]
    }
    prior_sources = {str(row["tfs1"]["source"]) for row in prior}
    prior_queries = {
        str(item["source_text"])
        for row in prior
        for item in row["natural_queries"].values()
    }
    prior_identities = {str(row["identity_sha256"]) for row in prior}
    prior_symbols = {str(value) for row in prior for value in row["tfs1"]["symbols"]}
    sources = {str(row["tfs1"]["source"]) for row in board}
    evidence = [
        str(item["source_text"]) for row in board for item in row["natural_evidence"]
    ]
    queries = [
        str(item["source_text"])
        for row in board
        for item in row["natural_queries"].values()
    ]
    identities = [str(row["identity_sha256"]) for row in board]
    symbols = {str(value) for row in board for value in row["tfs1"]["symbols"]}
    overlap = {
        "source_with_srp1": len(sources & prior_sources),
        "evidence_with_training": len(set(evidence) & evidence_texts),
        "query_with_training": len(set(queries) & query_texts),
        "query_with_srp1": len(set(queries) & prior_queries),
        "identity_with_srp1": len(set(identities) & prior_identities),
        "symbols_with_srp1": len(symbols & prior_symbols),
        "symbols_with_training": len(symbols & training_symbols),
    }
    if any(overlap.values()) or len(set(identities)) != CCR1_BOARD_ROWS:
        raise SystemExit(f"CCR1 split integrity failed: {overlap}")

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
        "seed": CCR1_BOARD_SEED,
        "rows": len(board),
        "fault_lines_per_episode": FAULT_LINES,
        "represented_worlds_per_episode": WORLDS,
        "represented_worlds_total": len(board) * WORLDS,
        "evidence_items": len(evidence),
        "query_items": len(queries),
        "entity_bank": list(CCR1_NAMES),
        "renderer_mode_counts": {
            key: dict(sorted(value.items()))
            for key, value in sorted(renderer_mode.items())
        },
        "query_role_order_counts": dict(sorted(role_order.items())),
        "overlap": overlap,
        "model_score_used_for_selection": False,
        "evidence_training_sha256": args.evidence_training_sha256,
        "query_training_sha256": args.query_training_sha256,
        "prior_srp1_sha256": args.prior_srp1_sha256,
        "board": str(board_path),
        "board_sha256": sha256_path(board_path),
    }
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "board": str(board_path),
                "board_sha256": report["board_sha256"],
                "report": str(report_path),
                "report_sha256": sha256_path(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
