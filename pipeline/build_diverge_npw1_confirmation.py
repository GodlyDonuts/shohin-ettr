#!/usr/bin/env python3
"""Materialize the source-disjoint DIVERGE-NPW1 confirmation board."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from diverge_npw1_data import (
    CONFIRMATION_NAMES,
    CONFIRMATION_ROWS,
    CONFIRMATION_SEED,
    augment_board,
    validate_augmented_row,
)
from diverge_tfs1_data import FAULT_LINES, WORLDS, generate_board, validate_row


SCHEMA = "shohin-diverge-npw1-confirmation-report-v1"


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
    parser.add_argument("--prior-sot1", type=Path, required=True)
    parser.add_argument("--prior-sot1-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=CONFIRMATION_SEED)
    parser.add_argument("--rows", type=int, default=CONFIRMATION_ROWS)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing NPW1 output: {args.output}")
    if args.seed != CONFIRMATION_SEED or args.rows != CONFIRMATION_ROWS:
        raise SystemExit("NPW1 frozen confirmation geometry differs")

    prior = _load_jsonl(args.prior_sot1, args.prior_sot1_sha256)
    semantic = generate_board(
        args.rows,
        args.seed,
        name_bank=CONFIRMATION_NAMES,
    )
    board = augment_board(
        semantic,
        seed=args.seed,
        confirmation=True,
    )
    for row in board:
        validate_row(row)
        validate_augmented_row(row, confirmation=True)

    prior_tfs = [row["tfs1"] for row in prior]
    prior_semantic = {str(row["identity_sha256"]) for row in prior_tfs}
    prior_typed_sources = {str(row["source"]) for row in prior_tfs}
    prior_symbols = {
        str(symbol) for row in prior_tfs for symbol in row["symbols"]
    }
    semantic_ids = [str(row["identity_sha256"]) for row in board]
    npw1_ids = [str(row["npw1_identity_sha256"]) for row in board]
    narratives = [str(row["natural_world"]["source_text"]) for row in board]
    symbols = {str(symbol) for row in board for symbol in row["symbols"]}
    overlap = {
        "semantic_identity_with_sot1": len(set(semantic_ids) & prior_semantic),
        "typed_source_with_sot1": len(
            {str(row["source"]) for row in board} & prior_typed_sources
        ),
        "narrative_with_sot1_typed_source": len(set(narratives) & prior_typed_sources),
        "symbols_with_sot1": len(symbols & prior_symbols),
    }
    if any(overlap.values()):
        raise SystemExit(f"NPW1 source-disjoint split failed: {overlap}")
    if len(set(semantic_ids)) != args.rows or len(set(npw1_ids)) != args.rows:
        raise SystemExit("NPW1 identities are not unique")

    args.output.mkdir(parents=True)
    board_path = args.output / "confirmation_board.jsonl"
    _atomic_jsonl(board_path, board)
    event_forms = Counter(
        str(event["form"])
        for row in board
        for event in row["natural_world"]["events"]
    )
    source_bytes = sum(len(value.encode("ascii")) for value in narratives)
    report: dict[str, object] = {
        "schema": SCHEMA,
        "seed": args.seed,
        "rows": len(board),
        "fault_lines_per_episode": FAULT_LINES,
        "represented_worlds_per_episode": WORLDS,
        "represented_worlds_total": len(board) * WORLDS,
        "source_bytes": source_bytes,
        "event_count": sum(event_forms.values()),
        "event_forms": dict(sorted(event_forms.items())),
        "semantic_identity_count": len(set(semantic_ids)),
        "npw1_identity_count": len(set(npw1_ids)),
        "narrative_count": len(set(narratives)),
        "symbol_count": len(symbols),
        "confirmation_name_bank": list(CONFIRMATION_NAMES),
        "overlap": overlap,
        "model_score_used_for_selection": False,
        "prior_sot1": str(args.prior_sot1),
        "prior_sot1_sha256": args.prior_sot1_sha256,
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
