#!/usr/bin/env python3
"""Apply KR2's model-owned keep action and score the selected trajectory."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from build_idr1_revision_data import _atomic_json, _atomic_lines
from build_kr2_training_data import KEEP_ACTION
from build_vcr1_revision_data import _load_jsonl, sha256_file
from hf_product_reasoning_rollouts import score_completion
from hf_vcr1_evaluate_reviser import summarize


class KR2RescoreError(RuntimeError):
    """The raw generation or keep-action lineage differs."""


def rescore(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_candidates.exists() or args.output_report.exists():
        raise KR2RescoreError("refusing existing KR2 rescore output")
    raw_report = json.loads(args.raw_report.read_text(encoding="utf-8"))
    if (
        raw_report.get("status") != "complete"
        or raw_report.get("candidates_sha256") != sha256_file(args.raw_candidates)
    ):
        raise KR2RescoreError("raw candidates are not report-bound")
    rows = _load_jsonl(args.data)
    raw = _load_jsonl(args.raw_candidates)
    if len(rows) != 1289 or len(raw) != len(rows):
        raise KR2RescoreError("KR2 development geometry differs")
    raw_by_id = {str(row["identity_sha256"]): row for row in raw}
    if len(raw_by_id) != len(raw):
        raise KR2RescoreError("raw identities are duplicated")

    outputs: list[dict[str, Any]] = []
    actions: Counter[str] = Counter()
    for row in rows:
        identity = str(row["identity_sha256"])
        candidate = raw_by_id.get(identity)
        if candidate is None or candidate.get("task") != row.get("task"):
            raise KR2RescoreError("raw identity/task binding differs")
        raw_completion = str(candidate.get("completion", ""))
        keep = args.mode == "keep_or_repair" and raw_completion.strip() == KEEP_ACTION
        predecessor = row.get("internal_draft")
        if not isinstance(predecessor, dict):
            raise KR2RescoreError("recursive predecessor is missing")
        selected = str(predecessor.get("completion", "")) if keep else raw_completion
        score = score_completion(row["assessor"], selected, code_timeout=args.code_timeout)
        predecessor_correct = bool(predecessor.get("correct"))
        actions["keep"] += int(keep)
        actions["rewrite"] += int(not keep)
        actions["keep_correct"] += int(keep and predecessor_correct)
        actions["keep_incorrect"] += int(keep and not predecessor_correct)
        outputs.append(
            {
                "schema": "shohin-kr2-selected-candidate-v1",
                "identity_sha256": identity,
                "task": row["task"],
                "action": "keep" if keep else "rewrite",
                "raw_completion": raw_completion,
                "completion": selected,
                "predecessor_correct": predecessor_correct,
                "raw_generated_tokens": candidate.get("generated_tokens"),
                "raw_max_token_exhausted": candidate.get("max_token_exhausted"),
                **score,
            }
        )
    selected_sha = _atomic_lines(args.output_candidates, outputs)
    metrics = summarize(rows, outputs)["metrics"]
    keep_precision = actions["keep_correct"] / actions["keep"] if actions["keep"] else 0.0
    report = {
        "schema": "shohin-kr2-keep-action-rescore-v1",
        "status": "complete",
        "mode": args.mode,
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "raw_candidates": str(args.raw_candidates.resolve()),
        "raw_candidates_sha256": sha256_file(args.raw_candidates),
        "raw_report": str(args.raw_report.resolve()),
        "raw_report_sha256": sha256_file(args.raw_report),
        "selected_candidates": str(args.output_candidates.resolve()),
        "selected_candidates_sha256": selected_sha,
        "actions": dict(actions),
        "keep_precision": keep_precision,
        "metrics": metrics,
    }
    _atomic_json(args.output_report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("keep_or_repair", "direct_rewrite"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--raw-candidates", type=Path, required=True)
    parser.add_argument("--raw-report", type=Path, required=True)
    parser.add_argument("--output-candidates", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--code-timeout", type=float, default=3.0)
    report = rescore(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
