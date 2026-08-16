"""Tests for deterministic upward MoE publication assets."""

from __future__ import annotations

import copy
import csv
import io
import json
from pathlib import Path

import pytest

from render_upward_moe_scaling import UpwardMoEFigureError, render

JOB = Path(__file__).with_name("jobs") / "render_upward_moe_scaling.sbatch"


def _analysis() -> dict:
    points = []
    for host, total, active, treatment, unchanged, wins, losses, retained in (
        ("Qwen3.6-35B-A3B", 35_000_000_000, 3_000_000_000, 141, 111, 36, 6, 105),
        (
            "NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
            120_000_000_000,
            12_000_000_000,
            146,
            112,
            38,
            4,
            108,
        ),
        (
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
            141_000_000_000,
            39_000_000_000,
            150,
            112,
            41,
            3,
            109,
        ),
    ):
        points.append(
            {
                "host": host,
                "architecture_series": "trained_revision",
                "total_parameters": total,
                "active_parameters": active,
                "rows": 256,
                "treatment_arm": "revision",
                "treatment_correct": treatment,
                "unchanged_correct": unchanged,
                "self_refinement_correct": 120,
                "gain_over_unchanged_count": treatment - unchanged,
                "gain_over_unchanged_percentage_points": 100
                * (treatment - unchanged)
                / 256,
                "gain_over_self_refinement_count": treatment - 120,
                "paired_wins": wins,
                "paired_losses": losses,
                "mcnemar_exact_two_sided_p": 0.001,
                "unchanged_correct_retained": retained,
                "unchanged_correct_retention": retained / unchanged,
                "source_sha256": "a" * 64,
            }
        )
    return {
        "schema": "shohin-upward-moe-scaling-analysis-v1",
        "status": "complete_curve",
        "point_count": 3,
        "minimum_points_for_curve": 3,
        "architecture_series": "trained_revision",
        "points": points,
        "curve": {
            "total_parameter_fit": {"slope_percentage_points_per_log10_parameter": 5.0},
            "active_parameter_fit": {
                "slope_percentage_points_per_log10_parameter": 4.0
            },
        },
        "claim": "positive_upward_cross_family_moe_scaling_supported",
        "capability_curve_claim": (
            "positive_upward_cross_family_moe_capability_scaling_supported"
        ),
        "conservative_retention_curve_claim": (
            "positive_upward_cross_family_moe_scaling_with_retention_supported"
        ),
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def test_render_writes_read_only_svg_csv_and_manifest(tmp_path: Path) -> None:
    analysis = _write(tmp_path / "analysis.json", _analysis())
    output = tmp_path / "figure"
    manifest = render(analysis.resolve(), output.resolve())
    assert manifest["status"] == "complete"
    assert manifest["scientific_scores_changed"] is False
    assert {row["name"] for row in manifest["records"]} == {
        "shohin-upward-moe-scaling.svg",
        "shohin-upward-moe-scaling-points.csv",
    }
    svg = (output / "shohin-upward-moe-scaling.svg").read_text()
    assert "Shohin upward MoE transfer" in svg
    assert "Gain vs total parameters" in svg
    assert "Gain vs active parameters" in svg
    assert "retention" in svg
    assert "95% CI" in svg
    rows = list(
        csv.DictReader(
            io.StringIO((output / "shohin-upward-moe-scaling-points.csv").read_text())
        )
    )
    assert len(rows) == 3
    assert rows[0]["gain_ci95_lower_percentage_points"]
    assert rows[2]["retention_ci95_upper_percent"]
    assert not any(path.stat().st_mode & 0o222 for path in output.iterdir())


def test_renderer_escapes_host_and_refuses_score_tamper(tmp_path: Path) -> None:
    payload = _analysis()
    payload["points"][1]["host"] = "Nemotron <script>"
    output = tmp_path / "figure"
    render(_write(tmp_path / "analysis.json", payload).resolve(), output.resolve())
    svg = (output / "shohin-upward-moe-scaling.svg").read_text()
    assert "&lt;script&gt;" in svg
    forged = copy.deepcopy(payload)
    forged["points"][1]["paired_wins"] += 1
    with pytest.raises(UpwardMoEFigureError, match="paired point accounting"):
        render(
            _write(tmp_path / "forged.json", forged).resolve(),
            (tmp_path / "forged").resolve(),
        )


def test_renderer_extends_to_fourth_ultra_point(tmp_path: Path) -> None:
    payload = _analysis()
    payload["points"].append(
        {
            "host": "NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
            "architecture_series": "trained_revision",
            "total_parameters": 550_000_000_000,
            "active_parameters": 55_000_000_000,
            "rows": 256,
            "treatment_arm": "revision",
            "treatment_correct": 154,
            "unchanged_correct": 112,
            "self_refinement_correct": 122,
            "gain_over_unchanged_count": 42,
            "gain_over_unchanged_percentage_points": 100 * 42 / 256,
            "gain_over_self_refinement_count": 32,
            "paired_wins": 45,
            "paired_losses": 3,
            "mcnemar_exact_two_sided_p": 0.0001,
            "unchanged_correct_retained": 109,
            "unchanged_correct_retention": 109 / 112,
            "source_sha256": "b" * 64,
        }
    )
    payload["point_count"] = 4
    output = tmp_path / "figure"
    manifest = render(
        _write(tmp_path / "analysis.json", payload).resolve(), output.resolve()
    )
    assert len(manifest["point_source_sha256s"]) == 4
    svg = (output / "shohin-upward-moe-scaling.svg").read_text()
    assert "Nemotron Ultra-550B-A55B-NVFP4" in svg
    assert "550B" in svg
    rows = list(
        csv.DictReader(
            io.StringIO((output / "shohin-upward-moe-scaling-points.csv").read_text())
        )
    )
    assert len(rows) == 4


@pytest.mark.parametrize("status", ["complete_insufficient_points", "pending", None])
def test_renderer_refuses_incomplete_curve(tmp_path: Path, status: str | None) -> None:
    payload = _analysis()
    payload["status"] = status
    with pytest.raises(UpwardMoEFigureError, match="contract"):
        render(
            _write(tmp_path / "analysis.json", payload).resolve(),
            (tmp_path / "output").resolve(),
        )


def test_figure_job_is_cpu_only_dependency_safe_and_runtime_bound() -> None:
    source = JOB.read_text(encoding="utf-8")
    assert "#SBATCH --no-requeue" in source
    assert "#SBATCH --gres" not in source
    assert "RUNTIME_MANIFEST_SHA256" in source
    assert "q36_verify_runtime" in source
    assert "render_upward_moe_scaling.py" in source
    assert '[[ ! -e "$OUTPUT_ROOT" && ! -L "$OUTPUT_ROOT" ]]' in source
