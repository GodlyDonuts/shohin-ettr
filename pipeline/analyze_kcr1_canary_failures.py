#!/usr/bin/env python3
"""Read-only attribution for one completed KCR1 development canary."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


class KCR1AttributionError(RuntimeError):
    """KCR1 attribution inputs are incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _group_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(items)
    if not total:
        raise KCR1AttributionError("empty KCR1 attribution group")
    counters = {
        "action_correct": sum(row.get("action_correct") is True for row in items),
        "semantic_correct": sum(row.get("correct") is True for row in items),
        "execution_exact": sum(row.get("execution_exact") is True for row in items),
        "valid_transaction": sum(row.get("valid_transaction") is True for row in items),
        "max_token_exhausted": sum(row.get("max_token_exhausted") is True for row in items),
        "action_correct_semantic_wrong": sum(
            row.get("action_correct") is True and row.get("correct") is not True
            for row in items
        ),
        "action_wrong_semantic_correct": sum(
            row.get("action_correct") is not True and row.get("correct") is True
            for row in items
        ),
    }
    return {
        "rows": total,
        **counters,
        **{
            f"{key}_accuracy": value / total
            for key, value in counters.items()
            if key not in {"max_token_exhausted"}
        },
        "mean_generated_tokens": sum(int(row.get("generated_tokens", 0)) for row in items)
        / total,
    }


def run(data_path: Path, candidates_path: Path, report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "shohin-kcr1-transaction-evaluation-v1"
        or report.get("status") != "complete"
        or report.get("split") != "development"
        or report.get("merged_from_shards") is not True
        or report.get("full_row_count") != 1566
        or report.get("data_sha256") != sha256_file(data_path)
        or report.get("candidates_sha256") != sha256_file(candidates_path)
    ):
        raise KCR1AttributionError("KCR1 merged canary custody differs")
    data = jsonl(data_path)
    candidates = jsonl(candidates_path)
    if len(data) != 1566 or len(candidates) != len(data):
        raise KCR1AttributionError("KCR1 attribution population differs")
    identities = [row.get("identity_sha256") for row in data]
    if len(set(identities)) != len(identities) or identities != [
        row.get("identity_sha256") for row in candidates
    ]:
        raise KCR1AttributionError("KCR1 attribution identity order differs")

    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_presentation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    source_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source, candidate in zip(data, candidates, strict=True):
        action = str(source.get("expected_action"))
        presentation = str(source.get("presentation"))
        if candidate.get("expected_action") != action:
            raise KCR1AttributionError("KCR1 expected action differs")
        by_action[action].append(candidate)
        by_presentation[presentation].append(candidate)
        confusion[action][str(candidate.get("predicted_action"))] += 1
        source_rows[str(source.get("source_identity_sha256"))].append(candidate)
    expected_actions = {"<KEEP>", "<CONTINUE>", "<RESTART>"}
    if set(by_action) != expected_actions or any(len(rows) != 3 for rows in source_rows.values()):
        raise KCR1AttributionError("KCR1 attribution branch geometry differs")

    source_outcomes = Counter()
    for rows in source_rows.values():
        source_outcomes[
            "all_actions_correct" if all(row.get("action_correct") is True for row in rows)
            else "action_inconsistent"
        ] += 1
        source_outcomes[
            "all_semantics_correct" if all(row.get("correct") is True for row in rows)
            else "semantic_inconsistent"
        ] += 1
    result = {
        "schema": "shohin-kcr1-canary-failure-attribution-v1",
        "status": "complete",
        "holdout_used": False,
        "data_sha256": sha256_file(data_path),
        "candidates_sha256": sha256_file(candidates_path),
        "evaluation_report_sha256": sha256_file(report_path),
        "overall": _group_summary(candidates),
        "by_expected_action": {
            key: _group_summary(value) for key, value in sorted(by_action.items())
        },
        "by_presentation": {
            key: _group_summary(value) for key, value in sorted(by_presentation.items())
        },
        "action_confusion": {
            key: dict(sorted(value.items())) for key, value in sorted(confusion.items())
        },
        "source_outcomes": dict(sorted(source_outcomes.items())),
        "claim_boundary": (
            "Read-only localization of the frozen KCR1 development canary; it does "
            "not change its pass/fail result or open holdout."
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise KCR1AttributionError("KCR1 attribution output already exists")
    result = run(args.data, args.candidates, args.evaluation_report)
    temporary = args.output.with_name(f"{args.output.name}.partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
