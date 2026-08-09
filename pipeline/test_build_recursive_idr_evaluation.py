from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_idr1_revision_data import EVAL_SCHEMA, REPORT_SCHEMA
from build_recursive_idr_evaluation import RecursiveIDRDataError, build
from build_vcr1_revision_data import sha256_file
from compare_recursive_idr import DEPTH_TWO_FLOOR, RETENTION_FLOOR, compare


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    source = tmp_path / "source.jsonl"
    source_report = tmp_path / "source_report.json"
    first = tmp_path / "first.jsonl"
    first_report = tmp_path / "first_report.json"
    rows: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    task_geometry = (("math500", 623, 223), ("bbh_logic", 637, 349), ("mbpp", 29, 17))
    index = 0
    for task, count, correct_count in task_geometry:
        for task_index in range(count):
            identity = f"{index:064x}"
            assessor = (
                {"task": "mbpp", "text": "return one", "test_list": ["f()==1"]}
                if task == "mbpp"
                else {"task": task, "question": "What is one?", "answer": "1"}
            )
            rows.append(
                {
                    "schema": EVAL_SCHEMA,
                    "identity_sha256": identity,
                    "split": "development",
                    "task": task,
                    "question": "old prompt",
                    "internal_draft": {"identity_sha256": identity, "completion": "old"},
                    "candidates": [{"lineage": "base"}, {"lineage": "expert"}],
                    "assessor": assessor,
                    "runtime_fields": ["question"],
                    "internal_draft_visible": True,
                    "external_candidate_text_visible": False,
                }
            )
            candidates.append(
                {
                    "schema": "shohin-idr1-revision-candidate-v1",
                    "identity_sha256": identity,
                    "task": task,
                    "completion": "def f(): return 1" if task == "mbpp" else "\\boxed{1}",
                    "correct": task_index < correct_count,
                }
            )
            index += 1
    _write_jsonl(source, rows)
    _write_jsonl(first, candidates)
    source_report.write_text(
        json.dumps(
            {
                "schema": REPORT_SCHEMA,
                "status": "complete",
                "outputs": {
                    "development": {
                        "path": str(source.resolve()),
                        "sha256": sha256_file(source),
                    }
                },
            }
        )
    )
    first_report.write_text(
        json.dumps(
            {
                "schema": "shohin-idr1-revision-evaluation-v1",
                "status": "complete",
                "split": "development",
                "data_sha256": sha256_file(source),
                "candidates_sha256": sha256_file(first),
            }
        )
    )
    return source, source_report, first, first_report


def test_recursive_gate_is_frozen_materially_above_depth_one() -> None:
    assert DEPTH_TWO_FLOOR == 615
    assert DEPTH_TWO_FLOOR - 589 == 26
    assert RETENTION_FLOOR == 0.98


def test_recursive_builder_error_is_fail_closed() -> None:
    assert issubclass(RecursiveIDRDataError, RuntimeError)


def test_recursive_builder_and_comparison_pass_frozen_contract(tmp_path: Path) -> None:
    source, source_report, first, first_report = _fixture(tmp_path)
    output = tmp_path / "recursive"
    report = build(
        argparse.Namespace(
            source_data=source,
            source_report=source_report,
            first_revision_candidates=first,
            first_revision_report=first_report,
            output=output,
        )
    )
    assert report["outputs"]["development"]["rows"] == 1289
    recursive_row = json.loads((output / "development.jsonl").read_text().splitlines()[0])
    assert recursive_row["recursion_depth"] == 2
    assert recursive_row["internal_draft"]["lineage"] == "idr1_depth1"
    assert "\\boxed{1}" in recursive_row["question"]

    second_rows = [json.loads(line) for line in first.read_text().splitlines()]
    repaired = 0
    for row in second_rows:
        if not row["correct"] and repaired < 26:
            row["correct"] = True
            repaired += 1
    second = tmp_path / "second.jsonl"
    _write_jsonl(second, second_rows)
    second_report = tmp_path / "second_report.json"
    second_report.write_text(
        json.dumps(
            {
                "status": "complete",
                "split": "development",
                "candidates_sha256": sha256_file(second),
            }
        )
    )
    comparison_path = tmp_path / "comparison.json"
    comparison = compare(
        argparse.Namespace(
            depth_one_candidates=first,
            depth_one_report=first_report,
            depth_two_candidates=second,
            depth_two_report=second_report,
            output=comparison_path,
        )
    )
    assert comparison["gate_pass"] is True
    assert comparison["metrics"]["overall"]["depth_two_correct"] == 615
    assert comparison["metrics"]["overall"]["repaired_error"] == 26
