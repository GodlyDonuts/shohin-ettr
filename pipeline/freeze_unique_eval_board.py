#!/usr/bin/env python3
"""Freeze an evaluation JSONL with one deterministic row per prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-unique-eval-board-v1"


class BoardFreezeError(RuntimeError):
    """The source board cannot be frozen without ambiguity."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_row(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise BoardFreezeError(f"row {line_number} is not an object")
            rows.append(value)
    if not rows:
        raise BoardFreezeError("source board is empty")
    return rows


def freeze_rows(
    rows: list[dict[str, Any]], identity_field: str, id_field: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for source_index, row in enumerate(rows):
        identity = row.get(identity_field)
        if not isinstance(identity, str) or not identity:
            raise BoardFreezeError(
                f"row {source_index + 1} lacks nonempty {identity_field!r}"
            )
        groups.setdefault(identity, []).append((source_index, row))

    keep_indices: set[int] = set()
    duplicate_groups: list[dict[str, Any]] = []
    for identity, members in groups.items():
        ranked = sorted(
            members,
            key=lambda item: (
                str(item[1].get(id_field, "")),
                _canonical_row(item[1]),
                item[0],
            ),
        )
        kept_index, kept_row = ranked[0]
        keep_indices.add(kept_index)
        if len(ranked) > 1:
            duplicate_groups.append(
                {
                    "identity_sha256": hashlib.sha256(identity.encode()).hexdigest(),
                    "kept_id": kept_row.get(id_field),
                    "dropped_ids": [row.get(id_field) for _, row in ranked[1:]],
                    "source_count": len(ranked),
                }
            )

    frozen = [row for index, row in enumerate(rows) if index in keep_indices]
    duplicate_groups.sort(key=lambda item: item["identity_sha256"])
    return frozen, duplicate_groups


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise BoardFreezeError(f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise BoardFreezeError(f"refusing to replace report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--identity-field", required=True)
    parser.add_argument("--id-field", default="task_id")
    args = parser.parse_args()

    if not args.input.is_file():
        raise BoardFreezeError(f"missing source board: {args.input}")
    source_sha256 = _sha256(args.input)
    rows = _load_rows(args.input)
    frozen, duplicate_groups = freeze_rows(rows, args.identity_field, args.id_field)
    _write_jsonl_atomic(args.output, frozen)
    report = {
        "dropped_rows": len(rows) - len(frozen),
        "duplicate_groups": duplicate_groups,
        "id_field": args.id_field,
        "identity_field": args.identity_field,
        "input": str(args.input.resolve()),
        "input_rows": len(rows),
        "input_sha256": source_sha256,
        "output": str(args.output.resolve()),
        "output_rows": len(frozen),
        "output_sha256": _sha256(args.output),
        "schema": SCHEMA,
        "status": "complete",
    }
    _write_json_atomic(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
