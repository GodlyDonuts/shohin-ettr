#!/usr/bin/env python3
"""Build a hash-bound IDR1 evaluation where revision one becomes the draft."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_idr1_revision_data import EVAL_SCHEMA, REPORT_SCHEMA, _atomic_json, _atomic_lines
from build_vcr1_revision_data import _load_jsonl, sha256_file, source_task_prompt
from ttr1_revision import internal_revision_prompt


class RecursiveIDRDataError(RuntimeError):
    """The frozen source, first revision, or identity binding differs."""


def _load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("status") != "complete":
        raise RecursiveIDRDataError(f"incomplete report: {path}")
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise RecursiveIDRDataError(f"refusing existing output root: {args.output}")

    source_report = _load_report(args.source_report)
    first_report = _load_report(args.first_revision_report)
    source_expected = source_report.get("outputs", {}).get("development", {})
    if (
        source_report.get("schema") != REPORT_SCHEMA
        or source_expected.get("sha256") != sha256_file(args.source_data)
        or Path(source_expected.get("path", "")).resolve() != args.source_data.resolve()
    ):
        raise RecursiveIDRDataError("source development data is not report-bound")
    if (
        first_report.get("schema") != "shohin-idr1-revision-evaluation-v1"
        or first_report.get("split") != "development"
        or first_report.get("data_sha256") != sha256_file(args.source_data)
        or first_report.get("candidates_sha256")
        != sha256_file(args.first_revision_candidates)
    ):
        raise RecursiveIDRDataError("first revision is not source/report-bound")

    sources = _load_jsonl(args.source_data)
    first = _load_jsonl(args.first_revision_candidates)
    if len(sources) != 1289 or len(first) != len(sources):
        raise RecursiveIDRDataError("development geometry differs")
    first_by_id = {str(row.get("identity_sha256")): row for row in first}
    source_ids = [str(row.get("identity_sha256")) for row in sources]
    if len(first_by_id) != len(first) or set(first_by_id) != set(source_ids):
        raise RecursiveIDRDataError("first-revision identity coverage differs")

    output_rows: list[dict[str, Any]] = []
    task_counts: dict[str, int] = {}
    for source in sources:
        if source.get("schema") != EVAL_SCHEMA or source.get("split") != "development":
            raise RecursiveIDRDataError("source row schema/split differs")
        identity = str(source["identity_sha256"])
        candidate = first_by_id[identity]
        if candidate.get("task") != source.get("task"):
            raise RecursiveIDRDataError("first-revision task binding differs")
        completion = candidate.get("completion")
        if not isinstance(completion, str) or not completion.strip():
            raise RecursiveIDRDataError("first revision is empty")
        assessor = source.get("assessor")
        if not isinstance(assessor, dict):
            raise RecursiveIDRDataError("source assessor is invalid")
        recursive_draft = {
            **candidate,
            "identity_sha256": identity,
            "lineage": "idr1_depth1",
        }
        task = str(source["task"])
        output_rows.append(
            {
                **source,
                "question": internal_revision_prompt(
                    source_task_prompt(assessor), completion, task
                ),
                "internal_draft": recursive_draft,
                "recursion_depth": 2,
                "depth_one_candidates_sha256": sha256_file(
                    args.first_revision_candidates
                ),
            }
        )
        task_counts[task] = task_counts.get(task, 0) + 1

    args.output.mkdir(parents=True)
    data_path = args.output / "development.jsonl"
    data_sha256 = _atomic_lines(data_path, output_rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "experiment": "ridr1-recurrent-revision-depth-v1",
        "recursion_depth": 2,
        "source_data": str(args.source_data.resolve()),
        "source_data_sha256": sha256_file(args.source_data),
        "source_report": str(args.source_report.resolve()),
        "source_report_sha256": sha256_file(args.source_report),
        "first_revision_candidates": str(args.first_revision_candidates.resolve()),
        "first_revision_candidates_sha256": sha256_file(
            args.first_revision_candidates
        ),
        "first_revision_report": str(args.first_revision_report.resolve()),
        "first_revision_report_sha256": sha256_file(args.first_revision_report),
        "counts": {"development": {"pairs": len(output_rows), **task_counts}},
        "outputs": {
            "development": {
                "path": str(data_path.resolve()),
                "sha256": data_sha256,
                "rows": len(output_rows),
            }
        },
        "runtime_fields": ["question"],
        "internal_draft_visible": True,
        "external_candidate_text_visible": False,
        "assessor_fields_visible_to_model": False,
    }
    _atomic_json(args.output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--first-revision-candidates", type=Path, required=True)
    parser.add_argument("--first-revision-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    report = build(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
