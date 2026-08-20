from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import score_gpt_oss_120b_commit_confirmation as module


def _write_source(
    path: Path, *, rows_count: int = 256, include_label: bool = False
) -> None:
    rows = []
    for index in range(rows_count):
        row = {
            "schema": module.SOURCE_SCHEMA,
            "split": "external_validation",
            "identity_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
            "task": "mmlu_pro",
            "source_prompt": f"question {index}",
        }
        if include_label and index == 0:
            row["answer"] = "leak"
        rows.append(row)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_confirmation_source_projection_binds_hash_and_rejects_labels(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    _write_source(source)
    digest = module.sha256_file(source)
    assert len(module._sources(source, digest)) == 256
    with pytest.raises(module.ConfirmationScoreError, match="bytes differ"):
        module._sources(source, "0" * 64)
    _write_source(source, include_label=True)
    with pytest.raises(module.ConfirmationScoreError, match="projection differs"):
        module._sources(source, module.sha256_file(source))


def test_confirmation_source_accepts_frozen_1023_geometry(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_source(source, rows_count=1_023)
    assert len(module._sources(source, module.sha256_file(source), 1_023)) == 1_023
    assert (1_023, 16) in module.ALLOWED_GEOMETRIES


def test_score_job_is_cpu_only_and_post_generation() -> None:
    root = Path(__file__).resolve().parents[1]
    wrapper = (
        root / "pipeline/jobs/gpt_oss_120b_commit_confirmation_score.sbatch"
    ).read_text()
    assert "--gres" not in wrapper
    assert "EXPECTED_ASSESSORS_SHA256" in wrapper
    assert "score_gpt_oss_120b_commit_confirmation.py" in wrapper
    assert "for ((index=0; index<EXPECTED_SHARDS; index++))" in wrapper
    assert '--expected-rows "$EXPECTED_ROWS"' in wrapper
    assert "SANDBOX_RECEIPT" not in wrapper
