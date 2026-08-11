#!/usr/bin/env python3
"""Attribute NDR1's aligned-control delta to immutable draft states."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-ndr1-state-failure-attribution-v1"


class NDR1AttributionError(RuntimeError):
    """NDR1 candidate, data, or identity custody differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_lines(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            identity = str(row.get("identity_sha256", ""))
            if len(identity) != 64 or identity in rows:
                raise NDR1AttributionError(f"identity coverage differs: {path}")
            rows[identity] = row
    if not rows:
        raise NDR1AttributionError(f"empty input: {path}")
    return rows


def pair_outcome(aligned: bool, shuffled: bool) -> str:
    if aligned and shuffled:
        return "both_correct"
    if aligned:
        return "aligned_only"
    if shuffled:
        return "shuffled_only"
    return "neither_correct"


def draft_state(draft: dict[str, Any]) -> str:
    correctness = "correct" if draft.get("correct") is True else "wrong"
    completion = "exhausted" if draft.get("max_token_exhausted") is True else "complete"
    return f"draft_{correctness}_{completion}"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise NDR1AttributionError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    development = load_lines(args.development_data)
    aligned = load_lines(args.aligned_candidates)
    shuffled = load_lines(args.shuffled_candidates)
    identities = set(development)
    if set(aligned) != identities or set(shuffled) != identities:
        raise NDR1AttributionError("arm identity coverage differs")
    if args.expected_rows is not None and len(identities) != args.expected_rows:
        raise NDR1AttributionError("development row count differs")

    training_report = json.loads(args.training_data_report.read_text(encoding="utf-8"))
    if (
        training_report.get("schema") != "shohin-ndr1-natural-revision-data-report-v1"
        or training_report.get("status") != "complete"
        or training_report.get("natural_drafts_only") is not True
    ):
        raise NDR1AttributionError("training data report differs")
    train_rows = int(training_report.get("admitted_rows_per_arm", 0))
    train_exhausted = int(training_report.get("draft_exhausted_rows_per_arm", -1))
    if train_rows <= 0 or not 0 <= train_exhausted <= train_rows:
        raise NDR1AttributionError("training draft-state counts differ")

    outcomes: Counter[str] = Counter()
    by_state: dict[str, Counter[str]] = defaultdict(Counter)
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    arm_stats: dict[str, Counter[str]] = {
        "aligned": Counter(),
        "shuffled": Counter(),
    }
    for identity in sorted(identities):
        row = development[identity]
        left = aligned[identity]
        right = shuffled[identity]
        if left.get("task") != row.get("task") or right.get("task") != row.get("task"):
            raise NDR1AttributionError("task binding differs")
        draft = row.get("internal_draft")
        if not isinstance(draft, dict) or draft.get("identity_sha256") != identity:
            raise NDR1AttributionError("internal draft binding differs")
        state = draft_state(draft)
        outcome = pair_outcome(bool(left.get("correct")), bool(right.get("correct")))
        outcomes[outcome] += 1
        by_state[state][outcome] += 1
        by_state[state]["rows"] += 1
        by_task[str(row.get("task"))][outcome] += 1
        for arm_name, candidate in (("aligned", left), ("shuffled", right)):
            stats = arm_stats[arm_name]
            stats["correct"] += int(candidate.get("correct") is True)
            stats["generated_tokens"] += int(candidate.get("generated_tokens", 0))
            stats["exhausted"] += int(candidate.get("max_token_exhausted") is True)
            stats["exact_draft_completion"] += int(
                str(candidate.get("completion", "")).strip()
                == str(draft.get("completion", "")).strip()
            )
            stats["prediction_equals_draft_prediction"] += int(
                candidate.get("prediction") is not None
                and candidate.get("prediction") == draft.get("prediction")
            )

    eval_exhausted = sum(
        counts["rows"] for state, counts in by_state.items() if state.endswith("_exhausted")
    )
    wrong_exhausted = by_state.get("draft_wrong_exhausted", Counter())
    total_delta = arm_stats["aligned"]["correct"] - arm_stats["shuffled"]["correct"]
    wrong_exhausted_delta = (
        wrong_exhausted["aligned_only"] - wrong_exhausted["shuffled_only"]
    )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "inputs": {
            "development_data_sha256": sha256_file(args.development_data),
            "aligned_candidates_sha256": sha256_file(args.aligned_candidates),
            "shuffled_candidates_sha256": sha256_file(args.shuffled_candidates),
            "training_data_report_sha256": sha256_file(args.training_data_report),
        },
        "rows": len(identities),
        "training_draft_state": {
            "rows": train_rows,
            "exhausted": train_exhausted,
            "exhausted_fraction": train_exhausted / train_rows,
        },
        "evaluation_draft_state": {
            "rows": len(identities),
            "exhausted": eval_exhausted,
            "exhausted_fraction": eval_exhausted / len(identities),
        },
        "pair_outcomes": dict(outcomes),
        "by_draft_state": {state: dict(counts) for state, counts in sorted(by_state.items())},
        "by_task": {task: dict(counts) for task, counts in sorted(by_task.items())},
        "arms": {name: dict(stats) for name, stats in arm_stats.items()},
        "aligned_minus_shuffled_answers": total_delta,
        "wrong_exhausted_aligned_minus_shuffled_answers": wrong_exhausted_delta,
        "fraction_of_total_deficit_from_wrong_exhausted": (
            wrong_exhausted_delta / total_delta if total_delta else None
        ),
        "interpretation": (
            "The matched training corpus underrepresents exhausted drafts relative to "
            "development, and most of the aligned deficit occurs on wrong exhausted drafts. "
            "This is read-only attribution and does not reopen exact NDR1."
        ),
    }
    atomic_json(args.output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-data", type=Path, required=True)
    parser.add_argument("--aligned-candidates", type=Path, required=True)
    parser.add_argument("--shuffled-candidates", type=Path, required=True)
    parser.add_argument("--training-data-report", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=1289)
    parser.add_argument("--output", type=Path, required=True)
    result = analyze(parser.parse_args())
    print(json.dumps({
        "aligned_minus_shuffled_answers": result["aligned_minus_shuffled_answers"],
        "wrong_exhausted_delta": result["wrong_exhausted_aligned_minus_shuffled_answers"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
