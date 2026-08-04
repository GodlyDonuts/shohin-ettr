#!/usr/bin/env python3
"""Focused tests for matched token-balanced reasoning mixes."""

import hashlib
import json
from pathlib import Path

import pytest

from audit_token_balanced_mix_pair import MixPairAuditError, audit_pair


def _write_mix(path: Path, rows: list[dict], *, seed: int = 7) -> Path:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    groups = {}
    for row in rows:
        group = row["training_group"]
        groups.setdefault(group, {"rows": 0})["rows"] += 1
    report = {
        "schema": "shohin-token-balanced-reasoning-mix-v1",
        "status": "complete",
        "model_revision": "revision",
        "tokenizer_name_or_path": "/model",
        "max_sequence_length": 4096,
        "workspace_slots": 0,
        "weights": {"code": 0.5, "math": 0.5},
        "requested_total_target_tokens": 100,
        "seed": seed,
        "selected_rows": len(rows),
        "selected_groups": groups,
        "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    report_path = path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report))
    return report_path


def _row(question: str, group: str, *, treatment: bool = False) -> dict:
    row = {"question": question, "response": "answer", "training_group": group}
    if treatment:
        row.update(
            reasoning_subtype="ocr2_execution_verified",
            verification="execution_verified_source_tests",
        )
    return row


def test_pair_requires_identical_non_code_rows_and_verified_treatment(tmp_path):
    shared = _row("shared math", "math")
    control = tmp_path / "control.jsonl"
    treatment = tmp_path / "treatment.jsonl"
    control_report = _write_mix(control, [shared, _row("old code", "code")])
    treatment_report = _write_mix(
        treatment, [shared, _row("new code", "code", treatment=True)]
    )
    report = audit_pair(
        control,
        control_report,
        treatment,
        treatment_report,
        treatment_group="code",
        treatment_subtype="ocr2_execution_verified",
        minimum_subtype_fraction=1.0,
    )
    assert report["shared_non_treatment_rows"] == 1
    assert report["treatment_subtype_fraction"] == 1.0
    assert report["shared_treatment_group_questions"] == 0


def test_pair_rejects_changed_non_code_row(tmp_path):
    control = tmp_path / "control.jsonl"
    treatment = tmp_path / "treatment.jsonl"
    control_report = _write_mix(
        control, [_row("math one", "math"), _row("old code", "code")]
    )
    treatment_report = _write_mix(
        treatment,
        [_row("math two", "math"), _row("new code", "code", treatment=True)],
    )
    with pytest.raises(MixPairAuditError, match="non-treatment selections differ"):
        audit_pair(
            control,
            control_report,
            treatment,
            treatment_report,
            treatment_group="code",
            treatment_subtype="ocr2_execution_verified",
            minimum_subtype_fraction=1.0,
        )


def test_pair_rejects_unverified_treatment_fraction(tmp_path):
    shared = _row("shared math", "math")
    control = tmp_path / "control.jsonl"
    treatment = tmp_path / "treatment.jsonl"
    control_report = _write_mix(control, [shared, _row("old code", "code")])
    treatment_report = _write_mix(
        treatment,
        [
            shared,
            _row("new verified", "code", treatment=True),
            _row("new weak", "code"),
        ],
    )
    with pytest.raises(MixPairAuditError, match="subtype fraction"):
        audit_pair(
            control,
            control_report,
            treatment,
            treatment_report,
            treatment_group="code",
            treatment_subtype="ocr2_execution_verified",
            minimum_subtype_fraction=1.0,
        )
