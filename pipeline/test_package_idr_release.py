#!/usr/bin/env python3
"""Tests for qualified internal-draft/revision release packaging."""

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from package_idr_release import IDRReleaseError, package


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def product_report(gate_pass: bool = True) -> dict:
    summary = {
        "solved": 300,
        "total": 538,
        "macro_accuracy": 0.6,
        "aime": {"correct": 3, "total": 30, "accuracy": 0.1},
        "domains": {
            name: {"correct": 60, "total": 100, "accuracy": 0.6}
            for name in (
                "grade_school_math",
                "competition_math",
                "science",
                "logic",
                "code",
            )
        },
    }
    control = json.loads(json.dumps(summary))
    control["solved"] = 270
    control["macro_accuracy"] = 0.54
    gates = {
        "at_least_27_additional_main_answers": gate_pass,
        "at_least_0_05_macro_gain": gate_pass,
        "all_five_domain_deltas_nonnegative": gate_pass,
    }
    return {
        "schema": "shohin-idr-product-comparison-v1",
        "status": "complete",
        "gate_pass": gate_pass,
        "gates": gates,
        "treatment_summary": summary,
        "control_summary": control,
        "deltas": {
            "solved": 30,
            "macro_accuracy": 0.06,
            "domains": {},
        },
    }


def fixture_args(tmp_path: Path, gate_pass: bool = True) -> argparse.Namespace:
    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    config.write_text('{"model_type":"test"}\n')
    draft = tmp_path / "draft.pt"
    draft.write_bytes(b"draft")
    revision = tmp_path / "revision.pt"
    revision.write_bytes(b"revision")
    report = tmp_path / "product.json"
    report.write_text(json.dumps(product_report(gate_pass)) + "\n")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"id":"x","question":"q","response_mode":"general"}\n')
    return argparse.Namespace(
        model_name="test/model",
        model_revision="abc123",
        model_root=model,
        expected_model_config_sha256=sha(config),
        draft_checkpoint=draft,
        expected_draft_checkpoint_sha256=sha(draft),
        revision_checkpoint=revision,
        expected_revision_checkpoint_sha256=sha(revision),
        product_report=report,
        expected_product_report_sha256=sha(report),
        interaction_prompts=prompts,
        expected_interaction_prompts_sha256=sha(prompts),
        output=tmp_path / "release",
    )


def test_packages_complete_qualified_delta(tmp_path):
    args = fixture_args(tmp_path)
    manifest = package(args)
    assert manifest["status"] == "qualified"
    assert manifest["base_model_included"] is False
    assert manifest["inference_path"] == [
        "draft_adapter.pt",
        "revision_adapter.pt",
    ]
    assert (args.output / "SHA256SUMS").is_file()
    assert (args.output / "MODEL_CARD.md").read_text().startswith(
        "# Shohin Internal Draft/Revision Delta"
    )
    with pytest.raises(IDRReleaseError, match="existing output"):
        package(args)


def test_refuses_failed_product_gate(tmp_path):
    args = fixture_args(tmp_path, gate_pass=False)
    with pytest.raises(IDRReleaseError, match="did not pass"):
        package(args)
    assert not args.output.exists()


def test_refuses_mismatched_hash(tmp_path):
    args = fixture_args(tmp_path)
    args.expected_revision_checkpoint_sha256 = "0" * 64
    with pytest.raises(IDRReleaseError, match="revision checkpoint SHA-256 differs"):
        package(args)
    assert not args.output.exists()
