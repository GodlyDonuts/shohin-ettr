#!/usr/bin/env python3
"""Apply the matched DIVERGE-GTI1 development promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "shohin-diverge-gti1-assessment-v1"


class GTI1AssessmentError(RuntimeError):
    """The matched GTI1 arm contract differs."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "shohin-diverge-gti1-evaluation-v1":
        raise GTI1AssessmentError(f"GTI1 evaluation schema differs: {path}")
    if payload.get("board_type") != "development":
        raise GTI1AssessmentError(f"GTI1 assessment received non-development result: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shohin", type=Path, required=True)
    parser.add_argument("--smollm2", type=Path, required=True)
    parser.add_argument("--shuffled", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing GTI1 assessment: {args.output}")
    reports = {
        "shohin": _load(args.shohin),
        "smollm2": _load(args.smollm2),
        "shuffled": _load(args.shuffled),
    }
    counts = {
        name: int(report["normal"]["overall"]["exact"])
        for name, report in reports.items()
    }
    conditions = {
        "all_same_board": len({report["data_sha256"] for report in reports.values()}) == 1,
        "real_arms_not_shuffled": (
            reports["shohin"]["shuffle_supervision"] is False
            and reports["smollm2"]["shuffle_supervision"] is False
        ),
        "control_is_shuffled": reports["shuffled"]["shuffle_supervision"] is True,
        "smollm2_absolute_gate": reports["smollm2"]["promotion_gate"]["passed"] is True,
        "smollm2_beats_shohin_by_32": counts["smollm2"] - counts["shohin"] >= 32,
        "shuffled_at_most_430": counts["shuffled"] <= 430,
    }
    report = {
        "schema": SCHEMA,
        "status": "pass" if all(conditions.values()) else "fail",
        "counts": counts,
        "delta_smollm2_minus_shohin": counts["smollm2"] - counts["shohin"],
        "promotion_gate": {"conditions": conditions, "passed": all(conditions.values())},
        "inputs": {
            name: {"path": str(path), "sha256": sha256_path(path)}
            for name, path in (
                ("shohin", args.shohin),
                ("smollm2", args.smollm2),
                ("shuffled", args.shuffled),
            )
        },
    }
    _atomic_json(args.output, report)
    os.chmod(args.output, 0o444)
    print(json.dumps({
        "output": str(args.output),
        "output_sha256": sha256_path(args.output),
        **report,
    }, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
