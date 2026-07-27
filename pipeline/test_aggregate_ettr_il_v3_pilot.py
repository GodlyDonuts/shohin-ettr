"""Tests for the independent ETTR-IL-v3 pilot aggregator."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from aggregate_ettr_il_v3_pilot import (
    PilotAggregateError,
    aggregate_reports,
)
from ettr_il_v3_pilot import SCHEMA, pilot_cells
from ettr_il_v3_protocol import PROTOCOL, canonical_json_bytes


COMMIT = "a" * 40
FREEZE = "b" * 64


def _write_reports(root: Path) -> None:
    root.mkdir()
    for index, cell in enumerate(pilot_cells()):
        report: dict[str, object] = {
            "beam_width": 8,
            "bucket_count": 1,
            "bucket_index": 0,
            "cell": cell.to_value(),
            "compressed_bytes": 1000,
            "compressed_bytes_per_core": 100.0,
            "cpu_seconds": 2.0,
            "cores": 10,
            "cores_per_cpu_second": 5.0,
            "episode_population_sha256": hashlib.sha256(
                str(index).encode("ascii")
            ).hexdigest(),
            "outcome_counts": {"answer": 10},
            "peak_rss_kib": 1024,
            "primary_replay_mismatches": 0,
            "protocol": PROTOCOL,
            "protocol_freeze_sha256": FREEZE,
            "query_answer_counts": {"false": 10, "true": 10},
            "schema": SCHEMA,
            "source_commit": COMMIT,
            "status": "pass",
            "uncompressed_bytes": 2000,
            "wall_seconds": 2.0,
        }
        report["report_sha256"] = hashlib.sha256(
            canonical_json_bytes(report)
        ).hexdigest()
        (root / f"cell-{index}.json").write_bytes(
            canonical_json_bytes(report)
        )


def test_complete_matrix_aggregates_and_projects_production(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    _write_reports(reports)
    aggregate = aggregate_reports(reports)
    assert aggregate["status"] == "pass"
    assert aggregate["cell_count"] == 54
    assert aggregate["candidate_population_floor"] > 62_500
    assert aggregate["conservative_projected_cpu_hours"] > 0
    assert aggregate["protocol_freeze_sha256"] == FREEZE
    assert len(aggregate["aggregate_sha256"]) == 64


def test_missing_report_fails_closed(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _write_reports(reports)
    (reports / "cell-0.json").unlink()
    with pytest.raises(PilotAggregateError, match="inventory"):
        aggregate_reports(reports)


def test_self_hash_mutation_fails_closed(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _write_reports(reports)
    path = reports / "cell-0.json"
    report = __import__("json").loads(path.read_text("ascii"))
    report["cores"] = 11
    path.write_bytes(canonical_json_bytes(report))
    with pytest.raises(PilotAggregateError, match="self-hash"):
        aggregate_reports(reports)
