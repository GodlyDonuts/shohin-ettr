#!/usr/bin/env python3
"""AQC1 conditional product data tests."""

from __future__ import annotations

import argparse
import hashlib
import json

from build_aqc1_product import pairs, revision, source


def write_lines(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_revision_and_pairs(tmp_path) -> None:
    tasks = ("gsm8k", "math500", "gpqa", "bbh_logic", "humaneval", "mbpp", "aime")
    product_rows = []
    for index in range(568):
        task = tasks[index % len(tasks)]
        assessor = {"task": task, "question": f"question {index}", "answer": "1"}
        if task == "bbh_logic":
            assessor["target"] = "A"
        if task == "gpqa":
            assessor["choices"] = [{"label": "A", "text": "one"}]
        if task == "humaneval":
            assessor.update(
                prompt="def f():\n",
                test="def check(candidate): assert candidate() == 1",
                entry_point="f",
            )
        if task == "mbpp":
            assessor.update(text=f"code {index}", test_list=["assert True"])
        product_rows.append(
            {
                "schema": "shohin-vcr1-product-eval-v1",
                "identity_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
                "task": task,
                "assessor": assessor,
            }
        )
    product = tmp_path / "product.jsonl"
    product_hash = write_lines(product, product_rows)
    product_report = tmp_path / "product.report.json"
    product_report.write_text(
        json.dumps(
            {
                "schema": "shohin-vcr1-product-data-report-v1",
                "status": "complete",
                "output": str(product.resolve()),
                "output_sha256": product_hash,
            }
        )
    )
    source_path, source_report = (
        tmp_path / "source.jsonl",
        tmp_path / "source.report.json",
    )
    source(
        argparse.Namespace(
            input=product,
            input_report=product_report,
            output=source_path,
            report=source_report,
        )
    )
    sources = [json.loads(line) for line in source_path.read_text().splitlines()]

    draft_path, draft_report = tmp_path / "draft.jsonl", tmp_path / "draft.report.json"
    drafts = [
        {
            "schema": "shohin-aqc1-product-candidates-v1",
            "identity_sha256": row["identity_sha256"],
            "task": row["task"],
            "lineage": "draft",
            "completion": "draft",
            "correct": False,
        }
        for row in sources
    ]
    draft_hash = write_lines(draft_path, drafts)
    draft_report.write_text(
        json.dumps(
            {
                "schema": "shohin-aqc1-product-data-report-v1",
                "status": "complete",
                "stage": "merged-draft",
                "output": str(draft_path.resolve()),
                "output_sha256": draft_hash,
            }
        )
    )
    revision_path, revision_report = (
        tmp_path / "revision.jsonl",
        tmp_path / "revision.report.json",
    )
    revision(
        argparse.Namespace(
            source=source_path,
            source_report=source_report,
            drafts=draft_path,
            drafts_report=draft_report,
            output=revision_path,
            report=revision_report,
        )
    )
    revision_rows = [
        json.loads(line) for line in revision_path.read_text().splitlines()
    ]
    assert len(revision_rows) == 568
    humaneval = next(row for row in revision_rows if row["task"] == "humaneval")
    assert "only executable Python code" in humaneval["question"]

    lineage_paths = []
    for lineage in ("idr1", "control"):
        path, report = (
            tmp_path / f"{lineage}.jsonl",
            tmp_path / f"{lineage}.report.json",
        )
        rows = [
            {
                "schema": "shohin-aqc1-product-candidates-v1",
                "identity_sha256": row["identity_sha256"],
                "task": row["task"],
                "lineage": lineage,
                "completion": lineage,
                "correct": lineage == "idr1",
            }
            for row in sources
        ]
        digest = write_lines(path, rows)
        report.write_text(
            json.dumps(
                {
                    "schema": "shohin-aqc1-product-data-report-v1",
                    "status": "complete",
                    "stage": f"merged-{lineage}",
                    "output": str(path.resolve()),
                    "output_sha256": digest,
                }
            )
        )
        lineage_paths.append((path, report))
    pair_path, pair_report = tmp_path / "pairs.jsonl", tmp_path / "pairs.report.json"
    pairs(
        argparse.Namespace(
            source=source_path,
            source_report=source_report,
            idr=lineage_paths[0][0],
            idr_report=lineage_paths[0][1],
            control=lineage_paths[1][0],
            control_report=lineage_paths[1][1],
            output=pair_path,
            report=pair_report,
        )
    )
    pair_rows = [json.loads(line) for line in pair_path.read_text().splitlines()]
    assert len(pair_rows) == 568
    assert all(row["outcome_class"] == "idr1_only" for row in pair_rows)
