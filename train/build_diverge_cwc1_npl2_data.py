#!/usr/bin/env python3
"""Materialize one immutable CWC1-to-NPL2 wrapper split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from diverge_cwc1_npl2_data import audit_wrapper_records, build_wrapper_records


REPORT_SCHEMA = "shohin-diverge-cwc1-npl2-wrapper-report-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_path(path) != expected_sha256:
        raise SystemExit("CWC1/NPL2 public input hash differs")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-data", type=Path, required=True)
    parser.add_argument("--public-data-sha256", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing existing CWC1/NPL2 wrapper output")
    public = _load_jsonl(args.public_data, args.public_data_sha256)
    rows = build_wrapper_records(public, split=args.split)
    audit = audit_wrapper_records(rows)
    if not audit["all_conditions_passed"]:
        raise SystemExit("CWC1/NPL2 wrapper audit failed")
    _atomic_jsonl(args.output, rows)
    output_sha256 = sha256_path(args.output)
    report = {
        "schema": REPORT_SCHEMA,
        "split": args.split,
        "public_data": str(args.public_data),
        "public_data_sha256": args.public_data_sha256,
        "output": str(args.output),
        "output_sha256": output_sha256,
        "audit": audit,
    }
    _atomic_json(args.report, report)
    os.chmod(args.output, 0o444)
    os.chmod(args.report, 0o444)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
