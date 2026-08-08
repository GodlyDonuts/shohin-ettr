#!/usr/bin/env python3
"""Build source-disjoint AQC1 pairs from frozen IDR1 trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

PAIR_SCHEMA = "shohin-aqc1-whole-trajectory-pair-v1"
REPORT_SCHEMA = "shohin-aqc1-pair-report-v1"
TASKS = ("math500", "bbh_logic", "mbpp")
OUTCOMES = ("both_correct", "idr1_only", "both_wrong", "control_only")


class AQC1BuildError(RuntimeError):
    """Frozen AQC1 source or output custody differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assigned_development_split(identity: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}\0{identity}".encode()).digest()
    return "train" if int.from_bytes(digest[:8], "big") % 10_000 < 8_000 else "development"


def load_candidates(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = row.get("identity_sha256")
            if not isinstance(identity, str) or len(identity) != 64 or identity in rows:
                raise AQC1BuildError(f"invalid or duplicate candidate identity: {path}")
            if row.get("task") not in TASKS or not isinstance(row.get("completion"), str):
                raise AQC1BuildError(f"invalid candidate row: {path}")
            if not isinstance(row.get("correct"), bool):
                raise AQC1BuildError(f"candidate correctness is absent: {path}")
            rows[identity] = row
    if not rows:
        raise AQC1BuildError(f"empty candidate file: {path}")
    return rows


def load_questions(path: Path, expected_split: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = row.get("identity_sha256")
            if row.get("split") != expected_split or not isinstance(identity, str):
                raise AQC1BuildError("IDR1 question split or identity differs")
            if identity in rows or not isinstance(row.get("question"), str):
                raise AQC1BuildError("IDR1 question coverage differs")
            rows[identity] = row
    return rows


def outcome(idr1_correct: bool, control_correct: bool) -> str:
    if idr1_correct and control_correct:
        return "both_correct"
    if idr1_correct:
        return "idr1_only"
    if control_correct:
        return "control_only"
    return "both_wrong"


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise AQC1BuildError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return sha256_file(path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise AQC1BuildError(f"refusing existing output: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_specs = {
        "development": (args.development_data, args.idr1_development, args.control_development),
        "holdout": (args.holdout_data, args.idr1_holdout, args.control_holdout),
    }
    output_rows: list[dict[str, Any]] = []
    source_receipts: dict[str, Any] = {}
    for source_split, (data_path, idr1_path, control_path) in source_specs.items():
        questions = load_questions(data_path, source_split)
        idr1 = load_candidates(idr1_path)
        control = load_candidates(control_path)
        if set(questions) != set(idr1) or set(questions) != set(control):
            raise AQC1BuildError(f"{source_split} identity coverage differs")
        for identity in sorted(questions):
            question = questions[identity]
            left, right = idr1[identity], control[identity]
            if left["task"] != question["task"] or right["task"] != question["task"]:
                raise AQC1BuildError("AQC1 task binding differs")
            split = (
                assigned_development_split(identity, args.seed)
                if source_split == "development"
                else "holdout"
            )
            output_rows.append(
                {
                    "schema": PAIR_SCHEMA,
                    "identity_sha256": identity,
                    "source_split": source_split,
                    "split": split,
                    "task": question["task"],
                    "question": question["question"],
                    "outcome_class": outcome(left["correct"], right["correct"]),
                    "candidates": [
                        {
                            "lineage": "idr1",
                            "completion": left["completion"],
                            "correct": left["correct"],
                            "generated_tokens": left["generated_tokens"],
                            "max_token_exhausted": left["max_token_exhausted"],
                        },
                        {
                            "lineage": "control",
                            "completion": right["completion"],
                            "correct": right["correct"],
                            "generated_tokens": right["generated_tokens"],
                            "max_token_exhausted": right["max_token_exhausted"],
                        },
                    ],
                }
            )
        source_receipts[source_split] = {
            "data": str(data_path.resolve()),
            "data_sha256": sha256_file(data_path),
            "idr1_candidates": str(idr1_path.resolve()),
            "idr1_candidates_sha256": sha256_file(idr1_path),
            "control_candidates": str(control_path.resolve()),
            "control_candidates_sha256": sha256_file(control_path),
            "rows": len(questions),
        }
    split_counts = Counter(row["split"] for row in output_rows)
    outcome_counts = {
        split: Counter(
            row["outcome_class"] for row in output_rows if row["split"] == split
        )
        for split in ("train", "development", "holdout")
    }
    task_counts = {
        split: Counter(row["task"] for row in output_rows if row["split"] == split)
        for split in ("train", "development", "holdout")
    }
    if set(split_counts) != {"train", "development", "holdout"}:
        raise AQC1BuildError("AQC1 split coverage differs")
    if any(set(outcome_counts[split]) != set(OUTCOMES) for split in outcome_counts):
        raise AQC1BuildError("AQC1 outcome coverage differs")
    output_sha256 = atomic_lines(args.output, output_rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "seed": args.seed,
        "split_rule": "development sha256(seed\\0identity) 80/20; original holdout preserved",
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
        "rows": len(output_rows),
        "split_counts": dict(split_counts),
        "outcome_counts": {key: dict(value) for key, value in outcome_counts.items()},
        "task_counts": {key: dict(value) for key, value in task_counts.items()},
        "source_receipts": source_receipts,
        "inference_fields": ["question", "candidate_a", "candidate_b"],
        "supervisor_only_fields": ["correct", "outcome_class", "task", "lineage"],
    }
    atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--holdout-data", type=Path, required=True)
    parser.add_argument("--idr1-development", type=Path, required=True)
    parser.add_argument("--idr1-holdout", type=Path, required=True)
    parser.add_argument("--control-development", type=Path, required=True)
    parser.add_argument("--control-holdout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026080820)
    args = parser.parse_args()
    report = build(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
