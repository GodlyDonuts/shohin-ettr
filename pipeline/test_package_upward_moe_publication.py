from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from package_upward_moe_publication import UpwardMoEPublicationError, package


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    analysis = _write(
        tmp_path / "analysis.json",
        {
            "schema": "shohin-upward-moe-scaling-analysis-v1",
            "status": "complete_curve",
            "point_count": 3,
            "points": [
                {
                    "host": host,
                    "source_sha256": str(index) * 64,
                }
                for index, host in enumerate(
                    ("Qwen3.6-35B-A3B", "Mixtral-8x22B", "Nemotron-Super")
                )
            ],
            "claim": "supported",
            "capability_curve_claim": "supported",
            "conservative_retention_curve_claim": "supported",
        },
    )
    figure_root = tmp_path / "figure"
    figure_root.mkdir()
    records = []
    for name, value in (
        ("shohin-upward-moe-scaling.svg", b"<svg/>\n"),
        ("shohin-upward-moe-scaling-points.csv", b"host,gain\n"),
    ):
        path = figure_root / name
        path.write_bytes(value)
        records.append(
            {
                "name": name,
                "sha256": hashlib.sha256(value).hexdigest(),
                "bytes": len(value),
            }
        )
    _write(
        figure_root / "manifest.json",
        {
            "schema": "shohin-upward-moe-scaling-figure-manifest-v1",
            "status": "complete",
            "analysis_sha256": hashlib.sha256(analysis.read_bytes()).hexdigest(),
            "point_source_sha256s": [str(index) * 64 for index in range(3)],
            "records": records,
            "scientific_scores_changed": False,
            "automatic_successor_authorized": False,
        },
    )
    accounting = []
    for index, host in enumerate(("Mixtral-8x22B", "Nemotron-Super"), start=1):
        accounting.append(
            _write(
                tmp_path / f"accounting-{index}.json",
                {
                    "schema": "shohin-upward-moe-slurm-accounting-v1",
                    "status": "complete",
                    "host": host,
                    "source_commit": str(index) * 40,
                    "allocation_count": 13 + index,
                    "charged_h100_hours": float(index),
                    "allocation_identity_sha256": str(index) * 64,
                    "all_required_complete": True,
                    "retry_count": 0,
                },
            )
        )
    return analysis, figure_root, accounting


def test_packages_complete_scores_figure_and_accounting(tmp_path: Path) -> None:
    analysis, figure, accounting = _fixture(tmp_path)
    output = tmp_path / "publication.json"
    result = package(
        analysis_path=analysis.resolve(),
        figure_root=figure.resolve(),
        accounting_paths=[path.resolve() for path in accounting],
        output=output.resolve(),
    )
    assert result["status"] == "complete"
    assert result["total_charged_h100_hours"] == 3.0
    assert len(result["accounting_records"]) == 2
    assert {row["name"] for row in result["figure_records"]} == {
        "shohin-upward-moe-scaling.svg",
        "shohin-upward-moe-scaling-points.csv",
    }
    assert not output.stat().st_mode & 0o222


def test_rejects_figure_or_accounting_tamper(tmp_path: Path) -> None:
    analysis, figure, accounting = _fixture(tmp_path)
    (figure / "shohin-upward-moe-scaling.svg").write_text("forged")
    with pytest.raises(UpwardMoEPublicationError, match="figure asset"):
        package(
            analysis_path=analysis.resolve(),
            figure_root=figure.resolve(),
            accounting_paths=[path.resolve() for path in accounting],
            output=(tmp_path / "bad-figure.json").resolve(),
        )
    analysis, figure, accounting = _fixture(tmp_path / "second")
    value = json.loads(accounting[0].read_text())
    value["retry_count"] = 1
    _write(accounting[0], value)
    with pytest.raises(UpwardMoEPublicationError, match="accounting"):
        package(
            analysis_path=analysis.resolve(),
            figure_root=figure.resolve(),
            accounting_paths=[path.resolve() for path in accounting],
            output=(tmp_path / "bad-accounting.json").resolve(),
        )


def test_job_is_cpu_only_runtime_bound_and_nonrequeueing() -> None:
    source = (
        Path(__file__).with_name("jobs") / "package_upward_moe_publication.sbatch"
    ).read_text()
    assert "#SBATCH --gres" not in source
    assert "#SBATCH --no-requeue" in source
    assert "q36_verify_runtime" in source
    assert "RUNTIME_MANIFEST_SHA256" in source
    assert "[[ ${#accounting_paths[@]} -eq 2 ]]" in source
