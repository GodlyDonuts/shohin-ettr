#!/usr/bin/env python3
"""Select complete WTV1 trajectories using sealed counterbalanced scores."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-wtv1-comparison-v1"


class WTV1ComparisonError(RuntimeError):
    """WTV1 candidate or score custody differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidates = load_jsonl(args.candidates)
    by_key = {
        (str(row["identity_sha256"]), int(row["sample_index"])): row
        for row in candidates
    }
    if len(by_key) != len(candidates):
        raise WTV1ComparisonError("candidate keys repeat")
    scores: dict[tuple[str, int], dict[str, Any]] = {}
    reports = []
    for path in args.scores:
        for row in load_jsonl(path):
            key = (str(row["identity_sha256"]), int(row["sample_index"]))
            if key in scores:
                raise WTV1ComparisonError("score keys overlap")
            scores[key] = row
        reports.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    if set(scores) != set(by_key):
        raise WTV1ComparisonError("score and candidate keys differ")

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for key, candidate in by_key.items():
        grouped[key[0]].append((candidate, scores[key]))
    selected = []
    per_family: Counter[str] = Counter()
    per_member: Counter[str] = Counter()
    correct = 0
    truncation = 0
    for identity, rows in grouped.items():
        rows.sort(key=lambda pair: int(pair[0]["sample_index"]))
        winner = max(rows, key=lambda pair: float(pair[1]["verifier_score"]))
        candidate, score = winner
        is_correct = bool(candidate["correct"])
        correct += int(is_correct)
        per_family[f"{candidate['corruption_family']}:total"] += 1
        per_family[f"{candidate['corruption_family']}:correct"] += int(is_correct)
        per_member[f"{candidate['pair_member']}:total"] += 1
        per_member[f"{candidate['pair_member']}:correct"] += int(is_correct)
        truncation += sum(int(bool(item[1].get("prompt_truncated"))) for item in rows)
        selected.append(
            {
                "identity_sha256": identity,
                "sample_index": int(candidate["sample_index"]),
                "candidate_origins": candidate["candidate_origins"],
                "correct": is_correct,
                "verifier_score": float(score["verifier_score"]),
            }
        )

    overall = 1769 + correct
    choice_correct = per_family["choice_final:correct"]
    numeric_correct = per_family["numeric_final:correct"]
    clean_correct = per_member["clean:correct"]
    fault_correct = per_member["fault:correct"]
    gate = {
        "disagreement_selected_at_least_105": correct >= 105,
        "overall_at_least_1874": overall >= 1874,
        "choice_total_at_least_220": 125 + choice_correct >= 220,
        "zero_prompt_truncation": truncation == 0,
    }
    payload = {
        "schema": SCHEMA,
        "status": "pass" if all(gate.values()) else "fail",
        "candidates": {
            "path": str(args.candidates.resolve()),
            "sha256": sha256_file(args.candidates),
        },
        "score_sources": reports,
        "disagreement_groups": len(grouped),
        "disagreement_selected_correct": correct,
        "overall_correct_with_unanimous_rows": overall,
        "overall_total": 1908,
        "choice_correct_with_unanimous_rows": 125 + choice_correct,
        "choice_total": 256,
        "numeric_correct_with_unanimous_rows": 1644 + numeric_correct,
        "numeric_total": 1652,
        "clean_correct_with_unanimous_rows": 900 + clean_correct,
        "clean_total": 954,
        "fault_correct_with_unanimous_rows": 869 + fault_correct,
        "fault_total": 954,
        "prompt_truncated_candidates": truncation,
        "gate": gate,
        "selected": selected,
    }
    if args.output.exists():
        raise WTV1ComparisonError(f"refusing existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps({key: value for key, value in result.items() if key != "selected"}, sort_keys=True))
    return 0 if result["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
