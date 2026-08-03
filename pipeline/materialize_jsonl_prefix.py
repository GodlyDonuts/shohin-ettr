#!/usr/bin/env python3
"""Materialize a provenance-bound prefix from a deterministically ordered JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-jsonl-prefix-v1"


class PrefixMaterializationError(RuntimeError):
    """The requested prefix cannot be materialized safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_prefix(
    source: Path,
    source_report: Path,
    output: Path,
    report_path: Path,
    *,
    rows: int,
    expected_source_sha256: str,
) -> dict[str, Any]:
    if rows <= 0:
        raise PrefixMaterializationError("rows must be positive")
    if output.exists() or report_path.exists():
        raise PrefixMaterializationError("refusing to replace an existing output")
    if not source.is_file() or not source_report.is_file():
        raise PrefixMaterializationError("source and source report must exist")

    parent = json.loads(source_report.read_text(encoding="utf-8"))
    if parent.get("status") != "complete":
        raise PrefixMaterializationError("source report is not complete")
    declared_sha256 = str(parent.get("output_sha256") or "")
    if declared_sha256 != expected_source_sha256:
        raise PrefixMaterializationError("source SHA-256 does not match its report")
    declared_rows = int(parent.get("rows") or 0)
    if declared_rows < rows:
        raise PrefixMaterializationError("source report contains too few rows")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    written = 0
    try:
        with source.open("rb") as source_handle, temporary.open("wb") as output_handle:
            for line_number, line in enumerate(source_handle, start=1):
                if written >= rows:
                    break
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    raise PrefixMaterializationError(
                        f"malformed JSONL at source line {line_number}"
                    ) from exc
                output_handle.write(line)
                digest.update(line)
                written += 1
        if written != rows:
            raise PrefixMaterializationError(
                f"source ended after {written} rows; requested {rows}"
            )
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    report = {
        "schema": SCHEMA,
        "status": "complete",
        "selection": "ordered-prefix",
        "rows": written,
        "source": str(source.resolve()),
        "source_declared_rows": declared_rows,
        "source_declared_sha256": declared_sha256,
        "source_report": str(source_report.resolve()),
        "source_report_sha256": _sha256(source_report),
        "output": str(output.resolve()),
        "output_sha256": digest.hexdigest(),
    }
    temporary_report = report_path.with_name(
        f".{report_path.name}.tmp.{os.getpid()}"
    )
    try:
        temporary_report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_report, report_path)
    except Exception:
        temporary_report.unlink(missing_ok=True)
        output.unlink(missing_ok=True)
        raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()
    report = materialize_prefix(
        args.source,
        args.source_report,
        args.output,
        args.report,
        rows=args.rows,
        expected_source_sha256=args.expected_source_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
