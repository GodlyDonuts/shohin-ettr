#!/usr/bin/env python3
"""Materialize development trajectories selected by a learned owner commit head."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from q36_mtr_roles import MODEL_REVISION

CANDIDATE_SCHEMA = "shohin-q36-mtr-model-draft-v1"
SELECTION_SCHEMA = "shohin-q36-mtr-commit-selection-v1"
REPORT_SCHEMA = "shohin-q36-mtr-owner-commit-materialization-v1"
DEVELOPMENT_ROWS = 1_289
TASKS = ("math500", "bbh_logic", "mbpp")


class Q36MTROwnerCommitApplyError(RuntimeError):
    """Owner candidate or learned-selection evidence differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise Q36MTROwnerCommitApplyError(f"missing or linked input: {path}")
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise Q36MTROwnerCommitApplyError(f"unreadable input: {path}") from error
    if not rows:
        raise Q36MTROwnerCommitApplyError(f"empty input: {path}")
    return rows


def _candidates(paths: list[Path]) -> dict[str, dict[str, Any]]:
    if len(paths) != 16:
        raise Q36MTROwnerCommitApplyError("owner candidate shard count differs")
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _jsonl(path):
            if row.get("split") != "development":
                continue
            identity = row.get("identity_sha256")
            if (
                row.get("schema") != CANDIDATE_SCHEMA
                or not isinstance(identity, str)
                or len(identity) != 64
                or identity in result
                or row.get("task") not in TASKS
                or not isinstance(row.get("completion"), str)
                or not row["completion"].strip()
                or not isinstance(row.get("max_token_exhausted"), bool)
            ):
                raise Q36MTROwnerCommitApplyError("owner development candidate differs")
            result[identity] = row
    if len(result) != DEVELOPMENT_ROWS:
        raise Q36MTROwnerCommitApplyError("owner development coverage differs")
    return result


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise Q36MTROwnerCommitApplyError(f"refusing existing output: {path}")
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTROwnerCommitApplyError(f"refusing existing report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def apply(args: argparse.Namespace) -> dict[str, Any]:
    first = _candidates(args.first_candidates)
    second = _candidates(args.second_candidates)
    if set(first) != set(second):
        raise Q36MTROwnerCommitApplyError("owner candidate identities differ")
    selections: dict[str, dict[str, Any]] = {}
    for row in _jsonl(args.selections):
        identity = row.get("identity_sha256")
        selected = row.get("selected_index")
        margin = row.get("margin")
        if (
            row.get("schema") != SELECTION_SCHEMA
            or not isinstance(identity, str)
            or identity not in first
            or identity in selections
            or row.get("task") != first[identity]["task"]
            or second[identity]["task"] != first[identity]["task"]
            or isinstance(selected, bool)
            or selected not in (0, 1)
            or row.get("selected_lineage") != ("revision", "unchanged")[selected]
            or not isinstance(row.get("order_consistent"), bool)
            or isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not math.isfinite(float(margin))
        ):
            raise Q36MTROwnerCommitApplyError("learned owner selection differs")
        selections[identity] = row
    if set(selections) != set(first) or len(selections) != DEVELOPMENT_ROWS:
        raise Q36MTROwnerCommitApplyError("learned owner selection coverage differs")

    selected_rows: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    for identity in sorted(selections):
        selection = selections[identity]
        index = int(selection["selected_index"])
        selected_counts[("first", "second")[index]] += 1
        selected_rows.append((first, second)[index][identity])
    output_sha256 = _atomic_lines(args.output, selected_rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "model_revision": MODEL_REVISION,
        "rows": len(selected_rows),
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
        "selections_sha256": sha256_file(args.selections),
        "selected": dict(sorted(selected_counts.items())),
        "order_consistent": sum(
            int(row["order_consistent"]) for row in selections.values()
        ),
        "label_or_assessor_reads": 0,
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-candidates", type=Path, action="append", required=True)
    parser.add_argument(
        "--second-candidates", type=Path, action="append", required=True
    )
    parser.add_argument("--selections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(apply(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
