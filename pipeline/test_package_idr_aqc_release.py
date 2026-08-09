#!/usr/bin/env python3
"""Tests for the complete draft/revision/commit release packager."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from package_idr_aqc_release import IDRAQCReleaseError, package, sha256_file


REVISION = "model-revision"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def fixture_args(tmp_path: Path) -> argparse.Namespace:
    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    config.write_text('{"model_type":"test"}\n', encoding="utf-8")
    draft = tmp_path / "draft.pt"
    draft.write_bytes(b"draft")
    revision = tmp_path / "revision.pt"
    revision.write_bytes(b"revision")
    commit = tmp_path / "commit.pt"
    commit.write_bytes(b"commit")
    draft_report = tmp_path / "draft-report.json"
    write_json(
        draft_report,
        {"schema": "train", "status": "complete", "model_revision": REVISION},
    )
    revision_report = tmp_path / "revision-report.json"
    write_json(
        revision_report,
        {
            "schema": "train",
            "status": "complete",
            "model_revision": REVISION,
            "warm_start_sha256": sha256_file(draft),
        },
    )
    commit_report = tmp_path / "commit-report.json"
    write_json(
        commit_report,
        {
            "schema": "shohin-aqc1-commit-report-v1",
            "status": "complete",
            "arm": "antisymmetric",
            "model_revision": REVISION,
            "adapter_checkpoint_sha256": sha256_file(draft),
            "checkpoint_sha256": sha256_file(commit),
            "holdout_gate_pass": True,
            "protected_adapter_unchanged": True,
        },
    )
    summary = {
        "solved": 383,
        "total": 538,
        "macro_accuracy": 0.75815,
    }
    product_report = tmp_path / "product-report.json"
    write_json(
        product_report,
        {
            "schema": "shohin-aqc1-product-application-v1",
            "status": "complete",
            "arm": "antisymmetric",
            "gate_pass": True,
            "commit_sha256": sha256_file(commit),
            "commit_report_sha256": sha256_file(commit_report),
            "gates": {"qualified": True},
            "arms": {
                "selected": summary,
                "idr1": {**summary, "solved": 374},
                "control": {**summary, "solved": 316},
            },
        },
    )
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        '{"id":"x","question":"q","response_mode":"math"}\n',
        encoding="utf-8",
    )
    return argparse.Namespace(
        model_name="test/model",
        model_revision=REVISION,
        model_root=model,
        expected_model_config_sha256=sha256_file(config),
        draft_checkpoint=draft,
        expected_draft_checkpoint_sha256=sha256_file(draft),
        draft_report=draft_report,
        expected_draft_report_sha256=sha256_file(draft_report),
        revision_checkpoint=revision,
        expected_revision_checkpoint_sha256=sha256_file(revision),
        revision_report=revision_report,
        expected_revision_report_sha256=sha256_file(revision_report),
        commit_checkpoint=commit,
        expected_commit_checkpoint_sha256=sha256_file(commit),
        commit_report=commit_report,
        expected_commit_report_sha256=sha256_file(commit_report),
        product_report=product_report,
        expected_product_report_sha256=sha256_file(product_report),
        interaction_prompts=prompts,
        expected_interaction_prompts_sha256=sha256_file(prompts),
        output=tmp_path / "release",
    )


def test_packages_complete_qualified_lineage(tmp_path: Path) -> None:
    args = fixture_args(tmp_path)
    manifest = package(args)
    assert manifest["status"] == "qualified"
    assert manifest["inference_stages"] == [
        "internal_draft",
        "trained_revision",
        "unchanged_continuation",
        "whole_trajectory_commit",
    ]
    assert manifest["product_summary"]["solved"] == 383
    assert (args.output / "commit.pt").is_file()
    assert (args.output / "SHA256SUMS").is_file()
    with pytest.raises(IDRAQCReleaseError, match="existing output"):
        package(args)


def test_refuses_unqualified_or_misbound_commit(tmp_path: Path) -> None:
    args = fixture_args(tmp_path)
    report = json.loads(args.commit_report.read_text(encoding="utf-8"))
    report["adapter_checkpoint_sha256"] = "0" * 64
    write_json(args.commit_report, report)
    args.expected_commit_report_sha256 = sha256_file(args.commit_report)
    product = json.loads(args.product_report.read_text(encoding="utf-8"))
    product["commit_report_sha256"] = args.expected_commit_report_sha256
    write_json(args.product_report, product)
    args.expected_product_report_sha256 = sha256_file(args.product_report)
    with pytest.raises(IDRAQCReleaseError, match="draft checkpoint"):
        package(args)
    assert not args.output.exists()


def test_refuses_failed_product_gate(tmp_path: Path) -> None:
    args = fixture_args(tmp_path)
    report = json.loads(args.product_report.read_text(encoding="utf-8"))
    report["gate_pass"] = False
    write_json(args.product_report, report)
    args.expected_product_report_sha256 = sha256_file(args.product_report)
    with pytest.raises(IDRAQCReleaseError, match="product gate did not pass"):
        package(args)
