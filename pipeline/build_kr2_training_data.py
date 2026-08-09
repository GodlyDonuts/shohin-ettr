#!/usr/bin/env python3
"""Build matched keep-or-repair and direct-rewrite stage-two training data."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from build_idr1_revision_data import REPORT_SCHEMA, _atomic_json, _atomic_lines
from build_vcr1_revision_data import _load_jsonl, sha256_file, source_task_prompt
from ttr1_revision import internal_revision_prompt


KEEP_ACTION = "<KEEP_PREVIOUS>"
TRAIN_SCHEMA = "shohin-kr2-stage-owner-train-v1"


class KR2DataError(RuntimeError):
    """The predecessor or stage-two training lineage differs."""


def _report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "complete":
        raise KR2DataError(f"incomplete report: {path}")
    return value


def _write_arm(
    root: Path,
    arm: str,
    train_rows: list[dict[str, Any]],
    development_rows: list[dict[str, Any]],
    provenance: dict[str, Any],
    counts: Counter[str],
) -> dict[str, Any]:
    root.mkdir(parents=True)
    train_path = root / "train.jsonl"
    development_path = root / "development.jsonl"
    train_sha = _atomic_lines(train_path, train_rows)
    development_sha = _atomic_lines(development_path, development_rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "experiment": "kr2-stage-specific-owner-v1",
        "arm": arm,
        "counts": {"train": dict(counts), "development": {"pairs": 1289}},
        "outputs": {
            "train": {"path": str(train_path.resolve()), "sha256": train_sha, "rows": 9655},
            "development": {
                "path": str(development_path.resolve()),
                "sha256": development_sha,
                "rows": 1289,
            },
        },
        "runtime_fields": ["question"],
        "internal_draft_visible": True,
        "external_candidate_text_visible": False,
        "assessor_fields_visible_to_model": False,
        **provenance,
    }
    _atomic_json(root / "report.json", report)
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise KR2DataError("refusing existing KR2 data output")
    idr_report = _report(args.idr_report)
    predecessor_report = _report(args.predecessor_report)
    recursive_report = _report(args.recursive_report)
    if (
        idr_report.get("outputs", {}).get("train", {}).get("sha256")
        != sha256_file(args.idr_train)
        or predecessor_report.get("candidates_sha256")
        != sha256_file(args.predecessor_candidates)
        or recursive_report.get("outputs", {}).get("development", {}).get("sha256")
        != sha256_file(args.recursive_development)
    ):
        raise KR2DataError("KR2 input/report hash binding differs")

    original = _load_jsonl(args.idr_train)
    predecessors = _load_jsonl(args.predecessor_candidates)
    predecessor_rows = _load_jsonl(args.predecessor_data)
    development = _load_jsonl(args.recursive_development)
    predecessor_by_id = {str(row["identity_sha256"]): row for row in predecessors}
    source_by_id = {str(row["identity_sha256"]): row for row in predecessor_rows}
    if (
        len(original) != 9655
        or len(predecessor_by_id) != 5824
        or set(predecessor_by_id) != set(source_by_id)
        or len(development) != 1289
    ):
        raise KR2DataError("KR2 data geometry differs")

    treatment: list[dict[str, Any]] = []
    direct: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in original:
        source_identity = str(row["source_identity_sha256"])
        candidate = predecessor_by_id[source_identity]
        source_row = source_by_id[source_identity]
        assessor = source_row["assessor"]
        task = str(source_row["task"])
        completion = str(candidate.get("completion", ""))
        prompt = internal_revision_prompt(source_task_prompt(assessor), completion, task)
        target = str(row["response"])
        common = {
            "schema": TRAIN_SCHEMA,
            "identity_sha256": row["identity_sha256"],
            "source_identity_sha256": source_identity,
            "outcome_class": row["outcome_class"],
            "presentation": row["presentation"],
            "question": prompt,
            "predecessor_correct": bool(candidate.get("correct")),
            "internal_draft_visible": True,
            "external_candidate_text_visible": False,
        }
        treatment.append(
            {
                **common,
                "response": KEEP_ACTION if candidate.get("correct") else target,
                "target_kind": "keep_previous" if candidate.get("correct") else "verified_repair",
            }
        )
        direct.append({**common, "response": target, "target_kind": "direct_rewrite"})
        counts["pairs"] += 1
        counts[task] += 1
        counts["keep_previous"] += int(bool(candidate.get("correct")))
        counts["verified_repair"] += int(not bool(candidate.get("correct")))

    provenance = {
        "predecessor_data": str(args.predecessor_data.resolve()),
        "predecessor_data_sha256": sha256_file(args.predecessor_data),
        "predecessor_candidates": str(args.predecessor_candidates.resolve()),
        "predecessor_candidates_sha256": sha256_file(args.predecessor_candidates),
        "predecessor_report": str(args.predecessor_report.resolve()),
        "predecessor_report_sha256": sha256_file(args.predecessor_report),
        "recursive_development": str(args.recursive_development.resolve()),
        "recursive_development_sha256": sha256_file(args.recursive_development),
        "recursive_report": str(args.recursive_report.resolve()),
        "recursive_report_sha256": sha256_file(args.recursive_report),
        "keep_action": KEEP_ACTION,
    }
    args.output.mkdir(parents=True)
    reports = {
        "keep_or_repair": _write_arm(
            args.output / "keep_or_repair", "keep_or_repair", treatment, development, provenance, counts
        ),
        "direct_rewrite": _write_arm(
            args.output / "direct_rewrite", "direct_rewrite", direct, development, provenance, counts
        ),
    }
    summary = {
        "schema": "shohin-kr2-stage-owner-data-summary-v1",
        "status": "complete",
        "counts": dict(counts),
        "reports": {
            arm: {
                "path": str((args.output / arm / "report.json").resolve()),
                "sha256": sha256_file(args.output / arm / "report.json"),
            }
            for arm in reports
        },
    }
    _atomic_json(args.output / "report.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idr-train", type=Path, required=True)
    parser.add_argument("--idr-report", type=Path, required=True)
    parser.add_argument("--predecessor-data", type=Path, required=True)
    parser.add_argument("--predecessor-candidates", type=Path, required=True)
    parser.add_argument("--predecessor-report", type=Path, required=True)
    parser.add_argument("--recursive-development", type=Path, required=True)
    parser.add_argument("--recursive-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    report = build(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
