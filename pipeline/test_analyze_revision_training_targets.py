import hashlib
import json
from pathlib import Path

import pytest

from analyze_revision_training_targets import (
    REPORT_SCHEMA,
    RevisionTrainingTargetError,
    analyze,
)

TRAIN_SCHEMA = "test-revision-train-v1"
ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "train.jsonl"
    rows = [
        {
            "schema": TRAIN_SCHEMA,
            "identity_sha256": "1" * 64,
            "source_identity_sha256": "a" * 64,
            "target_kind": "source_verified_repair",
            "outcome_class": "both_wrong",
            "question": "source and draft",
            "response": "\\boxed{7}",
            "presentation": 0,
            "internal_draft_visible": True,
            "external_candidate_text_visible": False,
        },
        {
            "schema": TRAIN_SCHEMA,
            "identity_sha256": "2" * 64,
            "source_identity_sha256": "b" * 64,
            "target_kind": "verified_candidate",
            "outcome_class": "expert_only",
            "question": "another source and draft",
            "response": "<think>reasoning</think>\\n\\boxed{8}",
            "presentation": 1,
            "internal_draft_visible": True,
            "external_candidate_text_visible": False,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_analyze_binds_target_horizon_and_cross_tab(tmp_path: Path) -> None:
    path, digest = _fixture(tmp_path)
    report = analyze(path, digest, TRAIN_SCHEMA)
    assert report["schema"] == REPORT_SCHEMA
    assert report["input"]["rows"] == 2
    assert report["input"]["unique_source_identity_sha256"] == 2
    assert report["overall"]["exact_boxed_response"] == 1
    assert report["overall"]["contains_think_open_tag"] == 1
    assert report["by_target_kind"]["source_verified_repair"]["rows"] == 1
    assert report["target_kind_by_outcome_class"] == {
        "source_verified_repair": {"both_wrong": 1},
        "verified_candidate": {"expert_only": 1},
    }


def test_analyze_rejects_hash_drift(tmp_path: Path) -> None:
    path, _ = _fixture(tmp_path)
    with pytest.raises(RevisionTrainingTargetError, match="SHA-256 differs"):
        analyze(path, "0" * 64, TRAIN_SCHEMA)


def test_analyze_rejects_duplicate_identity(tmp_path: Path) -> None:
    path, _ = _fixture(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["identity_sha256"] = rows[0]["identity_sha256"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RevisionTrainingTargetError, match="identity is duplicated"):
        analyze(path, digest, TRAIN_SCHEMA)


def test_analyze_rejects_model_visibility_drift(tmp_path: Path) -> None:
    path, _ = _fixture(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["external_candidate_text_visible"] = True
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(RevisionTrainingTargetError, match="revision schema"):
        analyze(path, digest, TRAIN_SCHEMA)


def test_report_schema_is_frozen() -> None:
    assert REPORT_SCHEMA == "shohin-revision-training-target-horizon-analysis-v1"


def test_frozen_qwen9_report_matches_newton_execution() -> None:
    path = (
        ROOT / "docs/research/SHOHIN_QWEN9_IDR1_TRAINING_TARGET_HORIZON_20260820.json"
    )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "7a868f2f83dc2486294adcbffbeba6ac89c2fe421ee71af6409cfabcf654f4a3"
    )
    report = json.loads(path.read_text())
    assert report["input"]["rows"] == 9655
    assert report["input"]["unique_source_identity_sha256"] == 5824
    short = report["by_target_kind"]["source_verified_repair"]
    full = report["by_target_kind"]["verified_candidate"]
    assert short["rows"] == 3294
    assert short["response_characters"]["median"] == 11.0
    assert short["response_characters"]["below_threshold"]["20"] == 2969
    assert short["exact_boxed_response"] == 3245
    assert full["rows"] == 5108
    assert full["response_characters"]["median"] == 706.0
