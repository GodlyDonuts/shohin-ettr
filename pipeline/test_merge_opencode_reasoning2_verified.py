#!/usr/bin/env python3
"""Focused merge-contract tests for independently verified OCR2 data."""

import json
from pathlib import Path

import pytest

from merge_opencode_reasoning2_verified import (
    OpenCodeReasoningMergeError,
    question_key,
    read_bound_inputs,
)
from verify_opencode_reasoning2_candidates import REPORT_SCHEMA, SCHEMA, sha256_file


def verified_row(dataset: str, identity: str, question: str, response_size: int = 20):
    return {
        "schema": SCHEMA,
        "identity_sha256": identity * 64,
        "source_dataset": dataset,
        "question": question,
        "response": "x" * response_size,
        "solution": "print(1)",
        "verified_cases": 3,
        "verification": "execution_verified_source_tests",
    }


def write_pair(root: Path, dataset: str, rows: list[dict]):
    data = root / f"{dataset}.jsonl"
    report = root / f"{dataset}.report.json"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report.write_text(
        json.dumps(
            {
                "schema": REPORT_SCHEMA,
                "status": "complete",
                "dataset": dataset,
                "source_revision": dataset + "-revision",
                "output_sha256": sha256_file(data),
                "counters": {"kept": len(rows)},
            }
        )
    )
    return data, report


def test_question_key_ignores_case_and_punctuation():
    assert question_key("Sort [3, 1]!") == question_key("sort 3 1")


def test_merge_requires_all_sources_and_deduplicates_questions(tmp_path):
    pairs = [
        write_pair(tmp_path, "apps", [verified_row("apps", "a", "Same question", 30)]),
        write_pair(tmp_path, "taco", [verified_row("taco", "b", "same QUESTION!", 20)]),
        write_pair(
            tmp_path,
            "code_contests",
            [verified_row("code_contests", "c", "Different question", 25)],
        ),
    ]
    rows, evidence = read_bound_inputs(
        [pair[0] for pair in pairs], [pair[1] for pair in pairs]
    )
    assert [row["identity_sha256"][0] for row in rows] == ["b", "c"]
    assert all(row["training_group"] == "code" for row in rows)
    assert all(row["reasoning_subtype"] == "ocr2_execution_verified" for row in rows)
    assert evidence["counters"]["duplicate_questions"] == 1
    assert evidence["counters"]["verified_cases"] == 6


def test_merge_rejects_tampered_input(tmp_path):
    pairs = [
        write_pair(tmp_path, dataset, [verified_row(dataset, key, dataset)])
        for dataset, key in (("apps", "a"), ("taco", "b"), ("code_contests", "c"))
    ]
    pairs[0][0].write_text("tampered\n")
    with pytest.raises(OpenCodeReasoningMergeError, match="hash differs"):
        read_bound_inputs([pair[0] for pair in pairs], [pair[1] for pair in pairs])


def test_merge_accepts_multiple_disjoint_shards_per_dataset(tmp_path):
    pairs = [
        write_pair(tmp_path, "apps", [verified_row("apps", "a", "apps one")]),
        write_pair(tmp_path, "taco", [verified_row("taco", "b", "taco one")]),
        write_pair(
            tmp_path,
            "code_contests",
            [verified_row("code_contests", "c", "contest one")],
        ),
    ]
    extra_dir = tmp_path / "extra"
    extra_dir.mkdir()
    extra = write_pair(extra_dir, "taco", [verified_row("taco", "d", "taco two")])
    rows, evidence = read_bound_inputs(
        [pair[0] for pair in pairs] + [extra[0]],
        [pair[1] for pair in pairs] + [extra[1]],
    )
    assert len(rows) == 4
    assert evidence["counters"]["verified_cases"] == 12
