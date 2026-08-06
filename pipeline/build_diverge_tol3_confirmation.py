#!/usr/bin/env python3
"""Build the one fresh DIVERGE-TOL3 confirmation board."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from diverge_tol1_data import split_report
from diverge_tol1_product import sha256_path
from diverge_tol3_confirmation_data import (
    CONFIRMATION_NAMES,
    generate_confirmation_split,
)


SCHEMA = "shohin-diverge-tol3-confirmation-report-v1"


def _load_identities(paths: list[Path]) -> tuple[set[str], list[dict[str, str]]]:
    identities = set()
    sources = []
    for path in paths:
        with path.open() as source:
            for line in source:
                if line.strip():
                    identities.add(str(json.loads(line)["identity_sha256"]))
        sources.append({"path": str(path), "sha256": sha256_path(path)})
    return identities, sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", required=True)
    parser.add_argument("--count", type=int, default=1_024)
    parser.add_argument("--seed", type=int, default=2026080506)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing confirmation output: {args.output}")
    if args.count != 1_024 or args.seed != 2026080506:
        raise SystemExit("TOL3 confirmation count or seed differs")

    excluded, source_files = _load_identities(args.exclude)
    rows = generate_confirmation_split(args.count, args.seed)
    identities = {str(row["identity_sha256"]) for row in rows}
    overlap = identities & excluded
    if overlap:
        raise SystemExit("TOL3 confirmation overlaps an earlier program identity")

    args.output.mkdir(parents=True)
    board_path = args.output / "confirmation.jsonl"
    temporary = board_path.with_suffix(".jsonl.tmp")
    with temporary.open("w") as destination:
        for row in rows:
            destination.write(json.dumps(row, sort_keys=True) + "\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, board_path)
    board_sha256 = sha256_path(board_path)
    report = {
        "schema": SCHEMA,
        "seed": args.seed,
        "count": args.count,
        "board": str(board_path),
        "board_sha256": board_sha256,
        "excluded_sources": source_files,
        "identity_overlap": 0,
        "names": list(CONFIRMATION_NAMES),
        "body_depth_min": min(int(row["body_depth"]) for row in rows),
        "body_depth_max": max(int(row["body_depth"]) for row in rows),
        "renderer": {
            "set": "into TARGET, set OPERAND",
            "add": "to TARGET, increase by OPERAND",
            "subtract": "from TARGET, decrease by OPERAND",
            "multiply": "multiply TARGET with OPERAND",
            "swap": "with RIGHT, exchange LEFT",
            "query": "with TARGET, return",
            "guard": "otherwise FALSE; if PREDICATE, then TRUE",
        },
        "split": split_report(rows),
    }
    report_path = args.output / "report.json"
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_report, report_path)
    print(
        json.dumps(
            {
                "board": str(board_path),
                "board_sha256": board_sha256,
                "report": str(report_path),
                "report_sha256": sha256_path(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
