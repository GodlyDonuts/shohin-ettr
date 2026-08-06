#!/usr/bin/env python3
"""Build the one frozen DIVERGE-TFS1 delayed-disambiguation board."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from diverge_tfs1_data import (
    FAULT_LINES,
    OPERATION_PAIRS,
    SCHEMA as BOARD_SCHEMA,
    TFS1_NAMES,
    WORLDS,
    generate_board,
)
from diverge_tol1_product import sha256_path


SCHEMA = "shohin-diverge-tfs1-build-report-v1"
COUNT = 256
SEED = 2026080607


def _atomic_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as destination:
        for row in rows:
            destination.write(json.dumps(row, sort_keys=True) + "\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)
    return sha256_path(path)


def _atomic_json(path: Path, payload: dict[str, object]) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as destination:
        destination.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)
    return sha256_path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=COUNT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing TFS1 output: {args.output}")
    if args.count != COUNT or args.seed != SEED:
        raise SystemExit("TFS1 count or seed differs from the frozen gate")

    rows = generate_board(args.count, args.seed)
    args.output.mkdir(parents=True)
    board_path = args.output / "board.jsonl"
    board_sha256 = _atomic_jsonl(board_path, rows)
    partial_counts = [int(row["partial_survivors"]) for row in rows]
    pair_counts = Counter(
        tuple(option["operation"] for option in step["options"])
        for row in rows
        for step in row["steps"]
        if step["options"] is not None
    )
    report = {
        "schema": SCHEMA,
        "board_schema": BOARD_SCHEMA,
        "board": str(board_path),
        "board_sha256": board_sha256,
        "count": len(rows),
        "seed": args.seed,
        "fault_lines_per_episode": FAULT_LINES,
        "represented_worlds_per_episode": WORLDS,
        "represented_worlds_total": len(rows) * WORLDS,
        "names": list(TFS1_NAMES),
        "operation_pairs": [list(value) for value in OPERATION_PAIRS],
        "operation_pair_counts": {
            "/".join(pair): count for pair, count in sorted(pair_counts.items())
        },
        "partial_survivors": {
            "minimum": min(partial_counts),
            "maximum": max(partial_counts),
            "mean": sum(partial_counts) / len(partial_counts),
        },
        "identity_count": len({str(row["identity_sha256"]) for row in rows}),
        "enumeration_commitment_count": len(
            {str(row["enumeration_sha256"]) for row in rows}
        ),
    }
    report_path = args.output / "report.json"
    report_sha256 = _atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "board": str(board_path),
                "board_sha256": board_sha256,
                "report": str(report_path),
                "report_sha256": report_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
