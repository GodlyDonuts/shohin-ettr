#!/usr/bin/env python3
"""Aggregate immutable per-seed DIVERGE-v0 matched-gate reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from diverge_v0_matched_gate import ARM_NAMES, SCHEMA


AGGREGATE_SCHEMA = "shohin-diverge-v0-matched-a-g-aggregate-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def aggregate(paths: list[Path]) -> dict[str, object]:
    if len(paths) != 5:
        raise ValueError("the frozen aggregate requires exactly five reports")
    reports = []
    seeds = set()
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("schema") != SCHEMA:
            raise ValueError(f"unexpected schema in {path}")
        seed = int(report["arguments"]["seed"])
        if seed in seeds:
            raise ValueError("duplicate matched-gate seed")
        seeds.add(seed)
        reports.append((path, report))
    board_records = [report["board"] for _, report in reports]
    if any(record != board_records[0] for record in board_records[1:]):
        raise ValueError("per-seed boards differ")
    winning = sum(bool(report["gate"]["pass_single_seed"]) for _, report in reports)
    arms = {}
    for arm in ARM_NAMES:
        values = [float(report["arms"][arm]["exact"]) for _, report in reports]
        arms[arm] = {
            "mean_exact": sum(values) / len(values),
            "minimum_exact": min(values),
            "maximum_exact": max(values),
            "per_seed": values,
        }
    intervention_drops = {}
    for name in reports[0][1]["interventions"]:
        values = [
            float(report["interventions"][name]["drop_points_from_G"])
            for _, report in reports
        ]
        intervention_drops[name] = {
            "minimum_drop_points": min(values),
            "mean_drop_points": sum(values) / len(values),
        }
    return {
        "schema": AGGREGATE_SCHEMA,
        "inputs": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "seed": int(report["arguments"]["seed"]),
                "checkpoint_sha256": report["inputs"]["checkpoint_sha256"],
            }
            for path, report in reports
        ],
        "board": board_records[0],
        "totals": {
            "seeds": len(reports),
            "episodes": sum(int(report["board"]["episodes"]) for _, report in reports),
            "queries": sum(int(report["board"]["queries"]) for _, report in reports),
        },
        "compiler_minima": {
            key: min(float(report["compiler"][key]) for _, report in reports)
            for key in ("packet_exact", "gold_support_recall", "valid_support_preserved")
        },
        "arms": arms,
        "intervention_drops": intervention_drops,
        "sharing": reports[0][1]["sharing"],
        "gate": {
            "winning_seeds": winning,
            "minimum_required_winning_seeds": 4,
            "pass": winning >= 4,
        },
        "claim_boundary": (
            "Five-seed synthetic source-sealed delayed-disambiguation mechanism result; "
            "not unrestricted language reasoning or public benchmark evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = aggregate(args.input)
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(args.output),
                "gate": report["gate"],
            },
            sort_keys=True,
        )
    )
    if not report["gate"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
