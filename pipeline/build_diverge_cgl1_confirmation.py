#!/usr/bin/env python3
"""Build the fresh balanced DIVERGE-CGL1 confirmation board."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from diverge_cgl1_confirmation_data import (
    BOARD_ROWS,
    BOARD_SEED,
    MODES,
    NAMES,
    generate_confirmation_board,
)
from diverge_cgl1_data import validate_public_record
from diverge_tfs1_data import FAULT_LINES, WORLDS


SCHEMA = "shohin-diverge-cgl1-confirmation-report-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cgl1-public", type=Path, required=True)
    parser.add_argument("--cgl1-public-sha256", required=True)
    parser.add_argument("--prior-board", type=Path, action="append", default=[])
    parser.add_argument("--prior-board-sha256", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=BOARD_SEED)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing CGL1 confirmation output: {args.output}")
    if args.seed != BOARD_SEED:
        raise SystemExit("CGL1 confirmation seed differs")
    if len(args.prior_board) != len(args.prior_board_sha256):
        raise SystemExit("CGL1 prior-board receipts differ")
    if sha256_path(args.cgl1_public) != args.cgl1_public_sha256:
        raise SystemExit("CGL1 public training hash differs")

    training_texts = set()
    training_symbols = set()
    with args.cgl1_public.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            validate_public_record(row)
            training_texts.add(str(row["source_text"]))
            training_symbols.update(str(value) for value in row["symbols"])

    prior_sources = set()
    prior_queries = set()
    prior_identities = set()
    prior_symbols = set()
    for path, expected in zip(
        args.prior_board, args.prior_board_sha256, strict=True
    ):
        if sha256_path(path) != expected:
            raise SystemExit(f"CGL1 prior board hash differs: {path}")
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                prior_sources.add(str(row["tfs1"]["source"]))
                prior_queries.update(
                    str(item["source_text"])
                    for item in row["natural_queries"].values()
                )
                prior_identities.add(str(row["identity_sha256"]))
                prior_symbols.update(str(value) for value in row["tfs1"]["symbols"])

    board = generate_confirmation_board(seed=args.seed)
    sources = {str(row["tfs1"]["source"]) for row in board}
    queries = {
        str(item["source_text"])
        for row in board
        for item in row["natural_queries"].values()
    }
    identities = {str(row["identity_sha256"]) for row in board}
    symbols = {str(value) for row in board for value in row["tfs1"]["symbols"]}
    overlap = {
        "source_with_prior": len(sources & prior_sources),
        "query_with_training": len(queries & training_texts),
        "query_with_prior": len(queries & prior_queries),
        "identity_with_prior": len(identities & prior_identities),
        "symbols_with_training": len(symbols & training_symbols),
        "symbols_with_prior": len(symbols & prior_symbols),
    }
    if any(overlap.values()) or len(identities) != BOARD_ROWS:
        raise SystemExit(f"CGL1 confirmation overlap differs: {overlap}")

    renderer_mode: defaultdict[str, Counter[str]] = defaultdict(Counter)
    renderer_order = Counter()
    transaction = Counter()
    for row in board:
        for mode in MODES:
            item = row["natural_queries"][mode]
            renderer = int(item["renderer"])
            renderer_mode[str(renderer)][mode] += 1
            renderer_order[f"{renderer}:{int(item['order'])}"] += 1
            transaction[str(tuple(int(value) for value in item["symbol_role_ids"]))] += 1
    if renderer_order != Counter(
        {f"{renderer}:{order}": 64 for renderer in range(6) for order in (0, 1)}
    ):
        raise SystemExit("CGL1 confirmation renderer/order receipt differs")

    args.output.mkdir(parents=True)
    board_path = args.output / "confirmation_board.jsonl"
    _atomic_jsonl(board_path, board)
    report = {
        "schema": SCHEMA,
        "seed": args.seed,
        "rows": len(board),
        "fault_lines_per_episode": FAULT_LINES,
        "represented_worlds_per_episode": WORLDS,
        "represented_worlds_total": len(board) * WORLDS,
        "query_items": len(board) * len(MODES),
        "entity_bank": list(NAMES),
        "renderer_mode_counts": {
            key: dict(sorted(value.items()))
            for key, value in sorted(renderer_mode.items())
        },
        "renderer_order_counts": dict(sorted(renderer_order.items())),
        "query_transaction_counts": dict(sorted(transaction.items())),
        "overlap": overlap,
        "model_score_used_for_selection": False,
        "generated_before_cgl1_development_result": True,
        "cgl1_public_sha256": args.cgl1_public_sha256,
        "prior_board_sha256": list(args.prior_board_sha256),
        "board": board_path.name,
        "board_sha256": sha256_path(board_path),
    }
    report_path = args.output / "report.json"
    _atomic_json(report_path, report)
    for path in (board_path, report_path):
        os.chmod(path, 0o444)
    os.chmod(args.output, 0o555)
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
