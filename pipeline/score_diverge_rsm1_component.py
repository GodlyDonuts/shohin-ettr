#!/usr/bin/env python3
"""Apply the frozen oracle-selection RSM1 component gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


TRAIN_SCHEMA = "shohin-diverge-rsm1-training-report-v1"
EVAL_SCHEMA = "shohin-diverge-rsm1-evaluation-v1"
GATE_SCHEMA = "shohin-diverge-rsm1-component-gate-v1"


class RSM1GateError(RuntimeError):
    """The RSM1 component result bundle violates its frozen contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RSM1GateError(f"report is not an object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise RSM1GateError(f"refusing to replace gate report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def score(args: argparse.Namespace) -> dict[str, Any]:
    train = _load(args.train_report)
    evaluation = _load(args.evaluation)
    if (
        train.get("schema") != TRAIN_SCHEMA
        or train.get("status") != "complete"
        or train.get("architecture") != "diverge-rsm1"
        or train.get("packet_arm") != "guarded"
        or train.get("updates") != 1600
        or train.get("identities_per_update") != 8
        or train.get("frozen_crp_unchanged") is not True
        or train.get("replay_changed") is not True
    ):
        raise RSM1GateError("RSM1 training report envelope differs")
    if (
        evaluation.get("schema") != EVAL_SCHEMA
        or evaluation.get("status") != "complete"
        or evaluation.get("packet_arm") != "guarded"
        or evaluation.get("trace_kind") != "wrong"
        or evaluation.get("selection_mode") != "forced"
        or evaluation.get("ablation") != "normal"
        or evaluation.get("checkpoint_update") != 1600
        or evaluation.get("model_unchanged") is not True
        or len(evaluation.get("results", [])) != 480
        or evaluation.get("overall", {}).get("rows") != 480
    ):
        raise RSM1GateError("RSM1 evaluation report envelope differs")
    if train.get("crp_checkpoint_sha256") != evaluation.get("crp_checkpoint_sha256"):
        raise RSM1GateError("RSM1 frozen packet receipt differs")
    if train.get("data_sha256") == evaluation.get("data_sha256"):
        raise RSM1GateError("RSM1 training and OOD evaluation boards coincide")
    families = evaluation.get("families")
    if not isinstance(families, dict) or set(families) != {
        "scalar",
        "register",
        "symbolic",
    }:
        raise RSM1GateError("RSM1 family accounting differs")
    if any(int(families[family].get("rows", -1)) != 160 for family in families):
        raise RSM1GateError("RSM1 family row counts differ")
    overall = evaluation["overall"]
    terminal_by_family = {
        family: int(values["terminal_correct"])
        for family, values in families.items()
    }
    trajectory_by_family = {
        family: int(values["full_trajectory_correct"])
        for family, values in families.items()
    }
    checks = {
        "terminal_at_least_90_percent": int(overall["terminal_correct"]) >= 432,
        "terminal_at_least_85_percent_each_family": all(
            value >= 136 for value in terminal_by_family.values()
        ),
        "full_trajectory_at_least_80_percent_each_family": all(
            value >= 128 for value in trajectory_by_family.values()
        ),
        "zero_invalid_terminal_states": int(overall["invalid_terminal"]) == 0,
        "zero_runtime_semantic_calls": evaluation.get("runtime_semantic_calls") == 0,
        "frozen_state_unchanged": evaluation.get("model_unchanged") is True,
        "finite_training_and_evaluation": _all_finite(train)
        and _all_finite(evaluation),
    }
    payload = {
        "schema": GATE_SCHEMA,
        "status": "pass" if all(checks.values()) else "fail",
        "gate_pass": all(checks.values()),
        "checks": checks,
        "terminal_correct": int(overall["terminal_correct"]),
        "terminal_by_family": terminal_by_family,
        "full_trajectory_correct": int(overall["full_trajectory_correct"]),
        "full_trajectory_by_family": trajectory_by_family,
        "packet_correct": int(overall["packet_correct"]),
        "invalid_terminal": int(overall["invalid_terminal"]),
        "train_report": (
            f"{args.train_report.resolve()}#{_sha256_file(args.train_report)}"
        ),
        "evaluation": (
            f"{args.evaluation.resolve()}#{_sha256_file(args.evaluation)}"
        ),
    }
    _atomic_json(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    report = score(parse_args())
    print(
        f"[rsm1-component] status={report['status']} "
        f"terminal={report['terminal_correct']}/480 "
        f"trajectory={report['full_trajectory_correct']}/480",
        flush=True,
    )
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
