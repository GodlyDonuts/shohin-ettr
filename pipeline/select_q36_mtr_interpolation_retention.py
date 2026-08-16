#!/usr/bin/env python3
"""Select between hierarchical and interpolated Q36 answers without labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from hf_q36_mtr_hierarchical_synthesis import ROWS, load_candidate_group

OUTPUT_SCHEMA = "shohin-q36-mtr-interpolation-retention-selection-v1"
REPORT_SCHEMA = "shohin-q36-mtr-interpolation-retention-report-v1"


class Q36MTRInterpolationRetentionError(RuntimeError):
    """Raised when the label-free retention selection differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def choose(hierarchy: dict[str, Any], interpolation: dict[str, Any]) -> str:
    if hierarchy.get("identity_sha256") != interpolation.get(
        "identity_sha256"
    ) or hierarchy.get("task") != interpolation.get("task"):
        raise Q36MTRInterpolationRetentionError("retention identity differs")
    task = hierarchy["task"]
    if task == "bbh_logic":
        return "hierarchy"
    if task == "mbpp":
        return "interpolation"
    if task == "math500":
        return (
            "interpolation"
            if interpolation["generated_tokens"] <= hierarchy["generated_tokens"]
            else "hierarchy"
        )
    raise Q36MTRInterpolationRetentionError("retention task differs")


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise Q36MTRInterpolationRetentionError("retention output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded.decode())
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRInterpolationRetentionError("retention report exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(
    hierarchy_paths: list[Path],
    interpolation_paths: list[Path],
    output: Path,
    report_path: Path,
) -> dict[str, Any]:
    hierarchy = load_candidate_group(hierarchy_paths, expected_paths=16)
    interpolation = load_candidate_group(interpolation_paths, expected_paths=16)
    if set(hierarchy) != set(interpolation) or len(hierarchy) != ROWS:
        raise Q36MTRInterpolationRetentionError("retention coverage differs")
    counts = {
        task: {"hierarchy": 0, "interpolation": 0}
        for task in ("bbh_logic", "math500", "mbpp")
    }
    rows = []
    for identity in sorted(hierarchy):
        selected = choose(hierarchy[identity], interpolation[identity])
        source = (
            hierarchy[identity] if selected == "hierarchy" else interpolation[identity]
        )
        row = dict(source)
        row["retention_selection"] = {
            "schema": OUTPUT_SCHEMA,
            "selected": selected,
            "rule": "task_and_generated_token_retention_v1",
            "development_labels_read": 0,
        }
        rows.append(row)
        counts[row["task"]][selected] += 1
    output_sha256 = _atomic_lines(output, rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "rows": len(rows),
        "selection_counts": counts,
        "hierarchy_sha256": [sha256_file(path) for path in hierarchy_paths],
        "interpolation_sha256": [sha256_file(path) for path in interpolation_paths],
        "output": str(output.resolve()),
        "output_sha256": output_sha256,
        "development_labels_read": 0,
    }
    _atomic_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchy", action="append", type=Path, required=True)
    parser.add_argument("--interpolation", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(
        json.dumps(
            run(args.hierarchy, args.interpolation, args.output, args.report),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
