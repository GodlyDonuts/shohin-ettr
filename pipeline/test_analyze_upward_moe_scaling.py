"""Tests for the upward MoE scaling analysis boundary."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from analyze_upward_moe_scaling import UpwardMoEScalingError, analyze

DOMAINS = {"bbh_logic": 128, "math500": 117, "mbpp": 11}


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _qwen() -> dict:
    return {
        "schema": "shohin-q36-mtr-multi-trajectory-screen-result-v1",
        "status": "complete_promoted",
        "host": {
            "model": "Qwen3.6-35B-A3B",
            "total_parameters": "35B",
            "active_parameters": "3B",
        },
        "source_disjoint_screen": {"rows": 256, "domain_rows": DOMAINS},
        "result": {
            "multi_trajectory": {
                "correct": 141,
                "domains": {"bbh_logic": 87, "math500": 44, "mbpp": 10},
            },
            "unchanged": {
                "correct": 111,
                "domains": {"bbh_logic": 71, "math500": 31, "mbpp": 9},
            },
            "absolute_gain_correct": 30,
            "paired_multi_only_correct": 35,
            "paired_unchanged_only_correct": 5,
            "mcnemar_exact_two_sided_p": 1.382612026645802e-06,
            "unchanged_correct_retained": 106,
            "unchanged_correct_retention": 106 / 111,
        },
        "matched_controls": {"self_refinement": {"correct": 121}},
    }


def _qwen_revision() -> dict:
    return {
        "schema": "shohin-q36-mtr-external-screen-result-summary-v1",
        "status": "complete",
        "host_model": "Qwen3.6-35B-A3B",
        "rows": 256,
        "arms": {
            "unchanged": {
                "correct": 111,
                "domains": {
                    "bbh_logic": [71, 128],
                    "math500": [31, 117],
                    "mbpp": [9, 11],
                },
            },
            "self_refinement": {
                "correct": 121,
                "domains": {
                    "bbh_logic": [76, 128],
                    "math500": [39, 117],
                    "mbpp": [6, 11],
                },
            },
            "revision": {
                "correct": 141,
                "gain_over_unchanged_count": 30,
                "arm_only_correct": 36,
                "unchanged_only_correct": 6,
                "mcnemar_exact_two_sided_p": 2.8288777684792876e-06,
                "unchanged_correct_retained": 105,
                "unchanged_correct_retention": 105 / 111,
                "domains": {
                    "bbh_logic": [85, 128],
                    "math500": [45, 117],
                    "mbpp": [11, 11],
                },
            },
        },
    }


def _matched(schema: str, host: str, total: int, active: int, gain: int) -> dict:
    unchanged = 112
    revision = unchanged + gain
    revision_domains = {
        "bbh_logic": {"correct": 72 + gain, "total": 128},
        "math500": {"correct": 31, "total": 117},
        "mbpp": {"correct": 9, "total": 11},
    }
    unchanged_domains = {
        "bbh_logic": {"correct": 72, "total": 128},
        "math500": {"correct": 31, "total": 117},
        "mbpp": {"correct": 9, "total": 11},
    }
    self_domains = {
        "bbh_logic": {"correct": 80, "total": 128},
        "math500": {"correct": 31, "total": 117},
        "mbpp": {"correct": 9, "total": 11},
    }
    return {
        "schema": schema,
        "status": "complete",
        "host": host,
        "total_parameters": total,
        "active_parameters": active,
        "rows": 256,
        "arms": {
            "unchanged": {
                "correct": unchanged,
                "domains": unchanged_domains,
            },
            "self_refinement": {"correct": 120, "domains": self_domains},
            "revision": {
                "correct": revision,
                "domains": revision_domains,
                "unchanged_correct_retained": 107,
                "unchanged_correct_retention": 107 / unchanged,
            },
        },
        "revision_vs_unchanged": {
            "left_only_correct": gain + 2,
            "right_only_correct": 2,
            "net_correct": gain,
            "mcnemar_exact_two_sided_p": 0.01,
        },
    }


def test_three_points_fit_positive_active_curve(tmp_path: Path) -> None:
    points = [
        _write(tmp_path / "qwen.json", _qwen_revision()),
        _write(
            tmp_path / "super.json",
            _matched(
                "shohin-nemotron-super-fixed-draft-screen-score-v1",
                "Nemotron-Super",
                120_000_000_000,
                12_000_000_000,
                32,
            ),
        ),
        _write(
            tmp_path / "mixtral.json",
            _matched(
                "shohin-mixtral-8x22b-fixed-draft-screen-score-v1",
                "Mixtral-8x22B",
                141_000_000_000,
                39_000_000_000,
                36,
            ),
        ),
    ]
    result = analyze(points)
    assert result["status"] == "complete_curve"
    assert result["capability_curve_claim"] == (
        "positive_upward_cross_family_moe_capability_scaling_supported"
    )
    assert result["claim"] == (
        "positive_moe_capability_scaling_with_conservative_retention_not_supported"
    )
    assert result["all_points_retention_at_least_95_percent"] is False
    assert (
        result["curve"]["active_parameter_fit"][
            "slope_percentage_points_per_log10_parameter"
        ]
        > 0
    )
    assert [point["active_parameters"] for point in result["points"]] == [
        3_000_000_000,
        12_000_000_000,
        39_000_000_000,
    ]


def test_two_points_refuse_scaling_claim(tmp_path: Path) -> None:
    points = [
        _write(tmp_path / "qwen.json", _qwen_revision()),
        _write(
            tmp_path / "super.json",
            _matched(
                "shohin-nemotron-super-fixed-draft-screen-score-v1",
                "Nemotron-Super",
                120_000_000_000,
                12_000_000_000,
                32,
            ),
        ),
    ]
    result = analyze(points)
    assert result["status"] == "complete_insufficient_points"
    assert result["curve"] is None
    assert result["claim"] == "insufficient_completed_moe_points_for_scaling_curve"


def test_domain_geometry_mismatch_fails(tmp_path: Path) -> None:
    qwen = _qwen_revision()
    for arm in ("unchanged", "self_refinement", "revision"):
        qwen["arms"][arm]["domains"]["math500"][1] = 116
        qwen["arms"][arm]["domains"]["mbpp"][1] = 12
    matched = _matched(
        "shohin-nemotron-super-fixed-draft-screen-score-v1",
        "Nemotron-Super",
        120_000_000_000,
        12_000_000_000,
        32,
    )
    with pytest.raises(UpwardMoEScalingError, match="geometry"):
        analyze(
            [
                _write(tmp_path / "qwen.json", qwen),
                _write(tmp_path / "super.json", matched),
            ]
        )


def test_paired_delta_tamper_fails(tmp_path: Path) -> None:
    matched = _matched(
        "shohin-nemotron-super-fixed-draft-screen-score-v1",
        "Nemotron-Super",
        120_000_000_000,
        12_000_000_000,
        32,
    )
    forged = copy.deepcopy(matched)
    forged["revision_vs_unchanged"]["net_correct"] = 31
    with pytest.raises(UpwardMoEScalingError, match="paired delta"):
        analyze([_write(tmp_path / "forged.json", forged)])


def test_different_architecture_series_cannot_be_combined(tmp_path: Path) -> None:
    matched = _matched(
        "shohin-nemotron-super-fixed-draft-screen-score-v1",
        "Nemotron-Super",
        120_000_000_000,
        12_000_000_000,
        32,
    )
    with pytest.raises(UpwardMoEScalingError, match="architecture series"):
        analyze(
            [
                _write(tmp_path / "multi.json", _qwen()),
                _write(tmp_path / "super.json", matched),
            ]
        )
