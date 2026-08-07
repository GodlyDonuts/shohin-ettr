#!/usr/bin/env python3
"""Assess the three matched PQI1 development arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


SCHEMA = "shohin-diverge-pqi1-assessment-v1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
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
        raise SystemExit(f"refusing existing PQI1 assessment: {args.output}")
    reports = {name: _load(path) for name, path in (
        ("shohin", args.shohin), ("smollm2", args.smollm2), ("shuffled", args.shuffled)
    )}
    if any(report.get("schema") != "shohin-diverge-pqi1-evaluation-v1" for report in reports.values()):
        raise SystemExit("PQI1 assessment schema differs")
    counts = {
        name: int(report["normal"]["overall"].get("exact", 0))
        for name, report in reports.items()
    }
    conditions = {
        "smollm2_absolute_gate": reports["smollm2"]["promotion_gate"]["passed"] is True,
        "smollm2_beats_shohin_by_64": counts["smollm2"] - counts["shohin"] >= 64,
        "shuffled_at_most_430": counts["shuffled"] <= 430,
        "all_same_board": len({report["data_sha256"] for report in reports.values()}) == 1,
        "real_arms_not_shuffled": not reports["shohin"]["shuffle_supervision"] and not reports["smollm2"]["shuffle_supervision"],
        "control_is_shuffled": reports["shuffled"]["shuffle_supervision"] is True,
    }
    report = {
        "schema": SCHEMA,
        "status": "pass" if all(conditions.values()) else "fail",
        "promotion_gate": {"conditions": conditions, "passed": all(conditions.values())},
        "counts": counts,
        "delta_smollm2_minus_shohin": counts["smollm2"] - counts["shohin"],
        "inputs": {
            name: {"path": str(path), "sha256": sha256_path(path)}
            for name, path in (
                ("shohin", args.shohin), ("smollm2", args.smollm2), ("shuffled", args.shuffled)
            )
        },
    }
    _atomic_json(args.output, report)
    os.chmod(args.output, 0o444)
    print(json.dumps({"output": str(args.output), "output_sha256": sha256_path(args.output), **report}, sort_keys=True))
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
