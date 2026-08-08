#!/usr/bin/env python3
"""AQC1 frozen arm aggregation tests."""

from __future__ import annotations

import argparse
import json

from aggregate_aqc1 import aggregate


def _report(arm: str, score: int, gate: bool) -> dict:
    return {
        "schema": "shohin-aqc1-commit-report-v1",
        "status": "complete",
        "arm": arm,
        "protected_adapter_unchanged": True,
        "adapter_checkpoint_sha256": "a" * 64,
        "pairs_sha256": "b" * 64,
        "model_revision": "revision",
        "updates": 128,
        "max_sequence_length": 3072,
        "holdout_gate_pass": gate,
        "holdout": {"overall": {"selected_correct": score}},
    }


def test_aggregate_requires_capability_and_relational_delta(tmp_path) -> None:
    treatment = tmp_path / "treatment.json"
    control = tmp_path / "control.json"
    output = tmp_path / "aggregate.json"
    treatment.write_text(json.dumps(_report("antisymmetric", 651, True)))
    control.write_text(json.dumps(_report("independent", 646, True)))
    report = aggregate(
        argparse.Namespace(treatment=treatment, control=control, output=output)
    )
    assert report["mechanism_gate_pass"] is True
    assert report["practical_winner"] == "antisymmetric"
