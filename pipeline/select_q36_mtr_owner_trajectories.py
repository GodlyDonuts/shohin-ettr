#!/usr/bin/env python3
"""Select between two complete Q36 owner trajectories without answer labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from hf_product_reasoning_eval import has_explicit_final_answer

INPUT_SCHEMA = "shohin-q36-mtr-model-draft-v1"
REPORT_SCHEMA = "shohin-q36-mtr-owner-trajectory-selection-report-v1"
RULE = "explicit_final_then_nonexhausted_then_first_v1"
TASKS = ("math500", "bbh_logic", "mbpp")


class Q36MTROwnerTrajectorySelectionError(RuntimeError):
    """Raised when two owner trajectory files cannot be selected exactly."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, split: str | None = None) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise Q36MTROwnerTrajectorySelectionError(
            f"unreadable owner candidate file: {path}"
        ) from error
    identities = []
    for row in rows:
        identity = row.get("identity_sha256") if isinstance(row, dict) else None
        if (
            row.get("schema") != INPUT_SCHEMA
            or not isinstance(identity, str)
            or len(identity) != 64
            or row.get("task") not in TASKS
            or row.get("split") not in {"train", "development"}
            or not isinstance(row.get("completion"), str)
            or not row["completion"].strip()
            or not isinstance(row.get("max_token_exhausted"), bool)
            or not isinstance(row.get("generated_tokens"), int)
            or row["generated_tokens"] <= 0
        ):
            raise Q36MTROwnerTrajectorySelectionError("owner candidate row differs")
        identities.append(identity)
    if not rows or len(identities) != len(set(identities)):
        raise Q36MTROwnerTrajectorySelectionError("owner candidate identities differ")
    selected = rows if split is None else [row for row in rows if row["split"] == split]
    if not selected:
        raise Q36MTROwnerTrajectorySelectionError("owner candidate split is empty")
    return selected


def _explicit(row: dict[str, Any]) -> bool:
    return row["task"] == "mbpp" or has_explicit_final_answer(row["completion"])


def _choose(first: dict[str, Any], second: dict[str, Any]) -> tuple[str, str]:
    first_explicit = _explicit(first)
    second_explicit = _explicit(second)
    if second_explicit and not first_explicit:
        return "second", "explicit_final_answer"
    if (
        second_explicit == first_explicit
        and first["max_token_exhausted"]
        and not second["max_token_exhausted"]
    ):
        return "second", "nonexhausted"
    return "first", "retained_first"


def select(
    first_path: Path, second_path: Path, split: str | None = None
) -> tuple[list[dict], dict[str, Any]]:
    if split not in (None, "train", "development"):
        raise Q36MTROwnerTrajectorySelectionError("owner selection split differs")
    first = _load(first_path, split)
    second = _load(second_path, split)
    if len(first) != len(second):
        raise Q36MTROwnerTrajectorySelectionError("owner candidate counts differ")
    selected = []
    reasons = {"explicit_final_answer": 0, "nonexhausted": 0, "retained_first": 0}
    adaptive_second_generation = 0
    for left, right in zip(first, second, strict=True):
        adaptive_second_generation += int(
            left["max_token_exhausted"] or not _explicit(left)
        )
        for field in (
            "identity_sha256",
            "task",
            "split",
            "prompt_sha256",
            "model_revision",
        ):
            if left.get(field) != right.get(field):
                raise Q36MTROwnerTrajectorySelectionError(
                    "owner candidate alignment differs"
                )
        choice, reason = _choose(left, right)
        source = left if choice == "first" else right
        row = dict(source)
        row["trajectory_selection"] = {
            "schema": "shohin-q36-mtr-owner-trajectory-selection-v1",
            "rule": RULE,
            "choice": choice,
            "reason": reason,
            "first_owner_checkpoint_sha256": left.get("owner_checkpoint_sha256"),
            "second_owner_checkpoint_sha256": right.get("owner_checkpoint_sha256"),
        }
        selected.append(row)
        reasons[reason] += 1
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "interpretation": "candidate_only_owner_trajectory_commit",
        "rule": RULE,
        "selected_split": split or "all",
        "rows": len(selected),
        "first_candidates": str(first_path.resolve()),
        "first_candidates_sha256": sha256_file(first_path),
        "second_candidates": str(second_path.resolve()),
        "second_candidates_sha256": sha256_file(second_path),
        "selection_counts": {
            "first": sum(
                row["trajectory_selection"]["choice"] == "first" for row in selected
            ),
            "second": sum(
                row["trajectory_selection"]["choice"] == "second" for row in selected
            ),
        },
        "reason_counts": reasons,
        "adaptive_generation": {
            "policy": "generate_second_only_if_first_exhausted_or_lacks_explicit_final_v1",
            "first_trajectory_calls": len(selected),
            "second_trajectory_calls": adaptive_second_generation,
            "trajectory_calls": len(selected) + adaptive_second_generation,
            "trajectory_calls_per_identity": (
                (len(selected) + adaptive_second_generation) / len(selected)
            ),
            "second_generation_fraction": adaptive_second_generation / len(selected),
            "second_generation_saved_vs_two_full": 1.0
            - adaptive_second_generation / len(selected),
        },
        "answer_labels_read": 0,
        "assessor_fields_read": 0,
    }
    return selected, report


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTROwnerTrajectorySelectionError("selection output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTROwnerTrajectorySelectionError("selection report exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-candidates", type=Path, required=True)
    parser.add_argument("--second-candidates", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "development"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, report = select(args.first_candidates, args.second_candidates, args.split)
    _atomic_jsonl(args.output, rows)
    report["output"] = str(args.output.resolve())
    report["output_sha256"] = sha256_file(args.output)
    _atomic_json(args.report, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
