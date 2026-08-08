from __future__ import annotations

import json
from pathlib import Path

import pytest

from hf_cvg1_completion_verifier import sha256_file
from hf_pcj1_apply_pairwise_judge import (
    PCJ1ApplicationError,
    _load_qualified_report,
)
from hf_pcj1_pairwise_judge import REPORT_SCHEMA


def test_application_requires_a_qualified_hash_bound_judge(tmp_path: Path) -> None:
    judge = tmp_path / "judge.pt"
    adapter = tmp_path / "adapter.pt"
    judge.write_bytes(b"judge")
    adapter.write_bytes(b"adapter")
    report_path = tmp_path / "report.json"
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "holdout": {"gate_pass": True},
        "judge_sha256": sha256_file(judge),
        "adapter_checkpoint_sha256": sha256_file(adapter),
        "inference_fields": ["question", "candidate_a", "candidate_b"],
        "task_or_benchmark_label_at_inference": False,
    }
    report_path.write_text(json.dumps(report))
    assert _load_qualified_report(report_path, judge, adapter) == report

    report["holdout"]["gate_pass"] = False
    report_path.write_text(json.dumps(report))
    with pytest.raises(PCJ1ApplicationError, match="did not pass"):
        _load_qualified_report(report_path, judge, adapter)
