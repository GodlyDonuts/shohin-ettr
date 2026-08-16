#!/usr/bin/env python3
"""Select label-free Q36 trajectories between two complete candidate groups."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import select_q36_mtr_owner_trajectories as base

REPORT_SCHEMA = "shohin-q36-mtr-candidate-group-selection-report-v1"
EXPECTED_ROWS = 1_289


class Q36MTRCandidateGroupError(RuntimeError):
    """Candidate-group inputs, alignment, or output coverage differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _group(paths: list[Path], split: str) -> dict[str, dict[str, Any]]:
    if not paths:
        raise Q36MTRCandidateGroupError("candidate group is empty")
    result: dict[str, dict[str, Any]] = {}
    resolved: set[Path] = set()
    for path in paths:
        canonical = path.resolve(strict=True)
        if canonical in resolved or canonical.is_symlink() or not canonical.is_file():
            raise Q36MTRCandidateGroupError("candidate group path differs")
        resolved.add(canonical)
        for row in base._load(canonical, split):
            identity = row["identity_sha256"]
            if identity in result:
                raise Q36MTRCandidateGroupError("candidate group identity overlaps")
            result[identity] = row
    if len(result) != EXPECTED_ROWS:
        raise Q36MTRCandidateGroupError("candidate group coverage differs")
    return result


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise Q36MTRCandidateGroupError("candidate group output exists")
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
        raise Q36MTRCandidateGroupError("candidate group report exists")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def select(args: argparse.Namespace) -> dict[str, Any]:
    first = _group(args.first_candidates, args.split)
    second = _group(args.second_candidates, args.split)
    if set(first) != set(second):
        raise Q36MTRCandidateGroupError("candidate group identities differ")
    selected: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    choices: Counter[str] = Counter()
    for identity in sorted(first):
        left, right = first[identity], second[identity]
        for field in ("identity_sha256", "task", "split", "prompt_sha256"):
            if left.get(field) != right.get(field):
                raise Q36MTRCandidateGroupError("candidate group alignment differs")
        choice, reason = base._choose(left, right)
        chosen = dict(left if choice == "first" else right)
        chosen["candidate_group_selection"] = {
            "schema": "shohin-q36-mtr-candidate-group-selection-v1",
            "rule": base.RULE,
            "choice": choice,
            "reason": reason,
            "first_group": args.first_label,
            "second_group": args.second_label,
        }
        selected.append(chosen)
        reasons[reason] += 1
        choices[choice] += 1
    output_sha = _atomic_lines(args.output, selected)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "interpretation": "label_free_complete_trajectory_group_commit",
        "split": args.split,
        "rows": len(selected),
        "rule": base.RULE,
        "first_label": args.first_label,
        "second_label": args.second_label,
        "selection_counts": dict(sorted(choices.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "first_candidate_sha256": [sha256_file(path) for path in args.first_candidates],
        "second_candidate_sha256": [
            sha256_file(path) for path in args.second_candidates
        ],
        "output": str(args.output.resolve()),
        "output_sha256": output_sha,
        "answer_labels_read": 0,
        "assessor_fields_read": 0,
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-candidates", type=Path, action="append", required=True)
    parser.add_argument(
        "--second-candidates", type=Path, action="append", required=True
    )
    parser.add_argument("--first-label", required=True)
    parser.add_argument("--second-label", required=True)
    parser.add_argument("--split", choices=("train", "development"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    payload = select(parse_args())
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
