from __future__ import annotations

from pathlib import Path

import pytest

from freeze_ettr_il_v2_phase1 import (
    COMPLETE_PARAMETERS,
    Phase1FreezeError,
    build_phase1_freeze,
    canonical_json_bytes,
    publish_no_replace,
)


ROOT = Path(__file__).resolve().parents[1]


def test_repository_phase1_freeze_is_complete_and_no_fit() -> None:
    report = build_phase1_freeze(ROOT)
    assert report["status"] == "pass"
    assert report["architecture"]["complete_parameters"] == COMPLETE_PARAMETERS
    assert report["authorization"] == {
        "fitting_authorized": False,
        "phase1_complete": True,
        "phase2_authorized": False,
        "pretraining_authorized": False,
        "production_population_materialized": False,
        "reasoning_capability_claimed": False,
        "weight_updates_performed": 0,
    }
    assert len(report["source_inventory"]) >= 30
    assert len(report["source_inventory_sha256"]) == 64
    assert len(report["test_inventory"]) >= 20
    assert len(report["test_inventory_sha256"]) == 64
    assert report["tokenizer"]["sha256"] == (
        "87532df5c121753de3b29194e1f9e3de47986d3f5359548fdf93606773a233d4"
    )
    assert canonical_json_bytes(report).endswith(b"\n")


def test_phase1_freeze_publication_is_no_replace(tmp_path: Path) -> None:
    report = build_phase1_freeze(ROOT)
    destination = tmp_path / "phase1.json"
    digest = publish_no_replace(report, destination)
    assert len(digest) == 64
    assert destination.read_bytes() == canonical_json_bytes(report)
    assert destination.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        publish_no_replace(report, destination)


def test_phase1_freeze_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(Phase1FreezeError, match="unavailable"):
        build_phase1_freeze(tmp_path)
