#!/usr/bin/env python3
"""Merge complete, disjoint MPR2 trained-owner draft shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPORT_SCHEMA = "shohin-mpr2-trained-owner-draft-shard-v1"
OUTPUT_SCHEMA = "shohin-mpr2-trained-owner-draft-v1"
MERGED_SCHEMA = "shohin-mpr2-trained-owner-drafts-v1"


class MPR2MergeError(RuntimeError):
    """MPR2 draft shards are incomplete, overlapping, or inconsistent."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise MPR2MergeError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise MPR2MergeError(f"refusing existing report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def merge(args: argparse.Namespace) -> dict[str, Any]:
    reports = [json.loads(path.read_text()) for path in args.shard_report]
    if not reports:
        raise MPR2MergeError("no MPR2 shard reports")
    count = int(reports[0].get("shard_count", 0))
    indices = sorted(int(report.get("shard_index", -1)) for report in reports)
    stable = (
        "source_sha256",
        "source_report_sha256",
        "unique_sources",
        "owner_checkpoint_sha256",
        "owner_architecture",
        "owner_update",
        "owner_draft_control",
        "model_revision",
        "generation_mode",
        "max_new_tokens",
        "seed",
        "shard_count",
    )
    if (
        len(reports) != count
        or indices != list(range(count))
        or any(report.get("schema") != REPORT_SCHEMA or report.get("status") != "complete" for report in reports)
        or any(any(report.get(key) != reports[0].get(key) for key in stable) for report in reports[1:])
    ):
        raise MPR2MergeError("MPR2 shard set differs")
    rows: list[dict[str, Any]] = []
    inputs = []
    for path, report in sorted(zip(args.shard_report, reports), key=lambda item: item[1]["shard_index"]):
        output = Path(str(report["output"]))
        if report["output_sha256"] != sha256_file(output):
            raise MPR2MergeError("MPR2 shard output hash differs")
        shard_rows = [json.loads(line) for line in output.read_text().splitlines() if line]
        if len(shard_rows) != int(report["rows"]) or any(row.get("schema") != OUTPUT_SCHEMA for row in shard_rows):
            raise MPR2MergeError("MPR2 shard row schema/count differs")
        rows.extend(shard_rows)
        inputs.append({"report": str(path.resolve()), "report_sha256": sha256_file(path), "output_sha256": report["output_sha256"]})
    rows.sort(key=lambda row: row["source_identity_sha256"])
    identities = [row["source_identity_sha256"] for row in rows]
    if len(rows) != int(reports[0]["unique_sources"]) or len(identities) != len(set(identities)):
        raise MPR2MergeError("MPR2 merged source coverage differs")
    output_sha256 = atomic_lines(args.output, rows)
    payload = {
        "schema": MERGED_SCHEMA,
        "status": "complete",
        "source_sha256": reports[0]["source_sha256"],
        "source_report_sha256": reports[0]["source_report_sha256"],
        "owner_checkpoint_sha256": reports[0]["owner_checkpoint_sha256"],
        "owner_architecture": reports[0]["owner_architecture"],
        "owner_update": reports[0]["owner_update"],
        "owner_draft_control": reports[0]["owner_draft_control"],
        "model_revision": reports[0]["model_revision"],
        "generation_mode": reports[0]["generation_mode"],
        "max_new_tokens": reports[0]["max_new_tokens"],
        "seed": reports[0]["seed"],
        "shard_count": count,
        "rows": len(rows),
        "generated_tokens": sum(int(report["generated_tokens"]) for report in reports),
        "max_token_exhausted": sum(int(report["max_token_exhausted"]) for report in reports),
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
        "inputs": inputs,
    }
    atomic_json(args.report, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    print(json.dumps(merge(parser.parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

