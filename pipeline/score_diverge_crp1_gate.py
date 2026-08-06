#!/usr/bin/env python3
"""Apply the frozen complete-trace causal-revision promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPORT_SCHEMA = "shohin-diverge-crp1-evaluation-v1"
GATE_SCHEMA = "shohin-diverge-crp1-gate-v1"
ARMS = {
    "plain_wrong": ("plain", "normal", "wrong"),
    "guarded_wrong": ("guarded", "normal", "wrong"),
    "unguarded_wrong": ("unguarded", "normal", "wrong"),
    "reset_wrong": ("guarded", "reset", "wrong"),
    "shift_wrong": ("guarded", "shift", "wrong"),
    "packet_swap_wrong": ("guarded", "packet_swap", "wrong"),
    "plain_correct": ("plain", "normal", "correct"),
    "guarded_correct": ("guarded", "normal", "correct"),
    "unguarded_correct": ("unguarded", "normal", "correct"),
}


class CRP1GateError(RuntimeError):
    """The CRP1 result bundle violates the frozen scoring contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CRP1GateError(f"report is not an object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise CRP1GateError(f"refusing to replace gate report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def score(args: argparse.Namespace) -> dict[str, Any]:
    paths = {name: getattr(args, name) for name in ARMS}
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    reports: dict[str, dict[str, Any]] = {}
    shared_data_sha256 = None
    for name, expected in ARMS.items():
        report = _load(paths[name])
        arm, ablation, variant = expected
        if (
            report.get("schema") != REPORT_SCHEMA
            or report.get("status") != "complete"
            or report.get("arm") != arm
            or report.get("ablation") != ablation
            or report.get("variant") != variant
            or report.get("input_rows") != 480
            or report.get("evaluated_rows") != 480
            or report.get("skipped_length") != 0
            or len(report.get("results", [])) != 480
        ):
            raise CRP1GateError(f"{name} report envelope differs")
        if shared_data_sha256 is None:
            shared_data_sha256 = report.get("data_sha256")
        elif report.get("data_sha256") != shared_data_sha256:
            raise CRP1GateError("CRP1 arms did not use one board")
        reports[name] = report

    guarded = reports["guarded_wrong"]
    unguarded = reports["unguarded_wrong"]
    exact = {name: int(report["exact_answers"]) for name, report in reports.items()}
    packet = {
        name: report.get("packet_localizations") for name, report in reports.items()
    }
    family_advantages = {
        family: int(guarded["family_metrics"][family]["exact_answers"])
        - int(unguarded["family_metrics"][family]["exact_answers"])
        for family in ("register", "scalar", "symbolic")
    }
    ablation_drops = {
        name: exact["guarded_wrong"] - exact[name]
        for name in ("reset_wrong", "shift_wrong", "packet_swap_wrong")
    }
    checks = {
        "guarded_wrong_exact_at_least_half": exact["guarded_wrong"] >= 240,
        "guarded_beats_plain_by_ten_points": (
            exact["guarded_wrong"] - exact["plain_wrong"] >= 48
        ),
        "guarded_beats_unguarded_by_five_points": (
            exact["guarded_wrong"] - exact["unguarded_wrong"] >= 24
        ),
        "guarded_packet_localization_at_least_75_percent": (
            int(packet["guarded_wrong"] or 0) >= 360
        ),
        "guarded_joint_revision_at_least_40_percent": (
            int(guarded["joint_correct"]) >= 192
        ),
        "guarded_preserves_valid_traces_at_90_percent": (
            exact["guarded_correct"] >= 432
        ),
        "guarded_no_error_packet_at_90_percent": (
            int(packet["guarded_correct"] or 0) >= 432
        ),
        "valid_trace_regression_bounded": (
            exact["guarded_correct"] >= exact["plain_correct"] - 12
        ),
        "guard_advantage_in_every_family": all(
            value >= 5 for value in family_advantages.values()
        ),
        "all_causal_interventions_drop_five_points": all(
            value >= 24 for value in ablation_drops.values()
        ),
        "two_causal_interventions_drop_ten_points": sum(
            value >= 48 for value in ablation_drops.values()
        )
        >= 2,
        "no_generation_exhaustion": all(
            int(report["exhausted"]) == 0 for report in reports.values()
        ),
    }
    payload = {
        "schema": GATE_SCHEMA,
        "status": "pass" if all(checks.values()) else "fail",
        "gate_pass": all(checks.values()),
        "checks": checks,
        "exact_answers": exact,
        "packet_localizations": packet,
        "guarded_joint_correct": int(guarded["joint_correct"]),
        "family_guarded_minus_unguarded": family_advantages,
        "ablation_drops": ablation_drops,
        "data_sha256": shared_data_sha256,
        "receipts": {
            name: f"{paths[name].resolve()}#{hashes[name]}" for name in paths
        },
    }
    _atomic_json(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ARMS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    report = score(parse_args())
    print(
        f"[crp1-gate] status={report['status']} "
        f"guarded={report['exact_answers']['guarded_wrong']}/480",
        flush=True,
    )
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
