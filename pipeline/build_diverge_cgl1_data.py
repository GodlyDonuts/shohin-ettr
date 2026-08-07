#!/usr/bin/env python3
"""Build immutable DIVERGE-CGL1 public/outcome training files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from diverge_cgl1_data import DATA_SEED, derive_outcome_orbits
from diverge_rrg1_data import ROWS_PER_STAGE


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
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DATA_SEED)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing CGL1 output: {args.output}")
    if sha256_path(args.source) != args.source_sha256:
        raise SystemExit("CGL1 source hash differs")
    rows = []
    with args.source.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    if len(rows) != ROWS_PER_STAGE:
        raise SystemExit("CGL1 source count differs")
    public, supervisors, report = derive_outcome_orbits(rows, seed=args.seed)
    args.output.mkdir(parents=True)
    public_path = args.output / "public.jsonl"
    supervisor_path = args.output / "supervisor.jsonl"
    report_path = args.output / "report.json"
    _atomic_jsonl(public_path, public)
    _atomic_jsonl(supervisor_path, supervisors)
    report = {
        **report,
        "source": str(args.source),
        "source_sha256": args.source_sha256,
        "public": str(public_path),
        "public_sha256": sha256_path(public_path),
        "supervisor": str(supervisor_path),
        "supervisor_sha256": sha256_path(supervisor_path),
    }
    _atomic_json(report_path, report)
    for path in (public_path, supervisor_path, report_path):
        os.chmod(path, 0o444)
    os.chmod(args.output, 0o555)
    print(
        json.dumps(
            {
                "public": str(public_path),
                "public_sha256": report["public_sha256"],
                "supervisor": str(supervisor_path),
                "supervisor_sha256": report["supervisor_sha256"],
                "report": str(report_path),
                "report_sha256": sha256_path(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
