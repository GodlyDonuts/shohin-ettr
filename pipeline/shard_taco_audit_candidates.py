#!/usr/bin/env python3
"""Split and reunite exact TACO verification candidates without changing rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-taco-audit-shards-v1"
MERGE_SCHEMA = "shohin-taco-audit-shard-merge-v1"


class TacoShardError(RuntimeError):
    """The candidate partition or verified union is not exact."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(errors="replace") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise TacoShardError(f"malformed row {line_number}: {path}") from error
            if not isinstance(row, dict) or row.get("problem_id") is None:
                raise TacoShardError(f"row {line_number} lacks problem_id: {path}")
            identity = str(row["problem_id"])
            if identity in seen:
                raise TacoShardError(f"duplicate problem_id {identity}: {path}")
            seen.add(identity)
            rows.append(row)
    if not rows:
        raise TacoShardError(f"candidate file is empty: {path}")
    return rows


def _atomic_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        raise TacoShardError(f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def split_candidates(
    input_path: Path, output_dir: Path, manifest_path: Path, shards: int
) -> dict[str, Any]:
    if shards <= 1:
        raise TacoShardError("at least two shards are required")
    if output_dir.exists() or manifest_path.exists():
        raise TacoShardError("split outputs must not already exist")
    rows = read_rows(input_path)
    output_dir.mkdir(parents=True)
    records = []
    for shard_index in range(shards):
        selected = rows[shard_index::shards]
        path = output_dir / f"candidates_{shard_index:02d}_of_{shards:02d}.jsonl"
        payload = b"".join(
            (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
            for row in selected
        )
        _atomic_bytes(path, payload)
        records.append(
            {
                "index": shard_index,
                "path": str(path.resolve()),
                "rows": len(selected),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "input": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "input_rows": len(rows),
        "strategy": "source-index-modulo-shard-count",
        "shard_count": shards,
        "shards": records,
    }
    _atomic_bytes(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    return manifest


def merge_verified(
    *,
    input_path: Path,
    manifest_path: Path,
    verified_paths: list[Path],
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise TacoShardError("split manifest schema differs")
    if manifest.get("input_sha256") != sha256_file(input_path):
        raise TacoShardError("candidate input hash differs from split manifest")
    shard_records = manifest.get("shards")
    if not isinstance(shard_records, list) or len(verified_paths) != len(shard_records):
        raise TacoShardError("verified path count differs from split manifest")

    candidates = read_rows(input_path)
    candidate_by_id = {str(row["problem_id"]): row for row in candidates}
    partition_ids: set[str] = set()
    accepted: dict[str, dict[str, Any]] = {}
    per_shard = []
    for record, verified_path in zip(shard_records, verified_paths, strict=True):
        shard_path = Path(record["path"])
        if sha256_file(shard_path) != record["sha256"]:
            raise TacoShardError(f"candidate shard hash differs: {shard_path}")
        shard_rows = read_rows(shard_path)
        shard_ids = {str(row["problem_id"]) for row in shard_rows}
        if partition_ids & shard_ids:
            raise TacoShardError("candidate shards overlap")
        partition_ids.update(shard_ids)

        verified_rows = read_rows(verified_path)
        for row in verified_rows:
            identity = str(row["problem_id"])
            if identity not in shard_ids:
                raise TacoShardError(f"verified row is outside its shard: {identity}")
            if identity in accepted:
                raise TacoShardError(f"verified row is duplicated: {identity}")
            original = candidate_by_id[identity]
            if str(row.get("response") or "").strip() != str(
                original.get("response") or ""
            ).strip():
                raise TacoShardError(f"verified response differs: {identity}")
            if (
                int(row.get("full_verified_cases") or 0) <= 0
                or row.get("training_group") != "code"
                or row.get("verification") != "execution_verified"
            ):
                raise TacoShardError(f"verified admission fields differ: {identity}")
            accepted[identity] = row
        per_shard.append(
            {
                "index": record["index"],
                "candidate_rows": len(shard_rows),
                "verified_rows": len(verified_rows),
                "verified_path": str(verified_path.resolve()),
                "verified_sha256": sha256_file(verified_path),
            }
        )
    if partition_ids != set(candidate_by_id):
        raise TacoShardError("candidate shards do not exactly partition the input")

    output_bytes = b"".join(
        (json.dumps(accepted[identity], ensure_ascii=False, sort_keys=True) + "\n").encode()
        for identity in (str(row["problem_id"]) for row in candidates)
        if identity in accepted
    )
    _atomic_bytes(output_path, output_bytes)
    report = {
        "schema": MERGE_SCHEMA,
        "status": "complete",
        "input": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "candidate_rows": len(candidates),
        "verified_rows": len(accepted),
        "dropped_rows": len(candidates) - len(accepted),
        "shards": per_shard,
        "output": str(output_path.resolve()),
        "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
    }
    _atomic_bytes(
        report_path,
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    split = subparsers.add_parser("split")
    split.add_argument("--input", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--shards", type=int, required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--input", type=Path, required=True)
    merge.add_argument("--manifest", type=Path, required=True)
    merge.add_argument("--verified", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "split":
        report = split_candidates(args.input, args.output_dir, args.manifest, args.shards)
    else:
        report = merge_verified(
            input_path=args.input,
            manifest_path=args.manifest,
            verified_paths=args.verified,
            output_path=args.output,
            report_path=args.report,
        )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
