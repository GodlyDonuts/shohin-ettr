"""Tests for the upward MoE scaling analysis boundary."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from analyze_upward_moe_scaling import UpwardMoEScalingError, analyze, normalize_point

DOMAINS = {"bbh_logic": 128, "math500": 117, "mbpp": 11}
JOB = Path(__file__).with_name("jobs") / "analyze_upward_moe_scaling.sbatch"


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


def _qwen_external_score() -> dict:
    summary = _qwen_revision()
    tasks = ["bbh_logic"] * 128 + ["math500"] * 117 + ["mbpp"] * 11
    desired = {
        "unchanged": {"bbh_logic": 71, "math500": 31, "mbpp": 9},
        "self_refinement": {"bbh_logic": 76, "math500": 39, "mbpp": 6},
        "revision": {"bbh_logic": 85, "math500": 45, "mbpp": 11},
    }
    outcomes = []
    domain_indices = {domain: 0 for domain in DOMAINS}
    for index, task in enumerate(tasks):
        offset = domain_indices[task]
        domain_indices[task] += 1
        outcomes.append(
            {
                "identity_sha256": f"{index:064x}",
                "task": task,
                "correct": {arm: offset < desired[arm][task] for arm in desired},
            }
        )
    # Rewire six baseline-correct revision rows to produce 36 wins / 6 losses
    # while preserving the exact marginal and domain totals.
    bbh = [row for row in outcomes if row["task"] == "bbh_logic"]
    for row in bbh[65:71]:
        row["correct"]["revision"] = False
    for row in bbh[85:91]:
        row["correct"]["revision"] = True
    arms = {}
    for arm in ("unchanged", "self_refinement", "revision"):
        domains = {}
        for domain in DOMAINS:
            selected = [row for row in outcomes if row["task"] == domain]
            domains[domain] = {
                "correct": sum(row["correct"][arm] for row in selected),
                "total": len(selected),
            }
        arms[arm] = {
            "correct": sum(row["correct"][arm] for row in outcomes),
            "domains": domains,
        }
    arms["revision"].update(
        {
            "gain_over_unchanged_count": 30,
            "paired_vs_unchanged": {
                "arm_only_correct": 36,
                "unchanged_only_correct": 6,
                "mcnemar_exact_two_sided_p": 2.8288777684792876e-06,
            },
        }
    )
    assert arms["revision"]["correct"] == summary["arms"]["revision"]["correct"]
    return {
        "schema": "shohin-q36-mtr-external-score-v1",
        "status": "complete",
        "rows": 256,
        "arms": arms,
        "outcomes": outcomes,
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
    weighted = result["curve"]["paired_sampling_weighted_active_parameter_fit"]
    assert weighted["sampling_model"] == (
        "independent_marginal_paired_outcome_normal_approximation"
    )
    assert weighted["slope_percentage_points_per_log10_parameter"] > 0
    assert weighted["slope_standard_error_percentage_points"] > 0
    assert len(weighted["slope_ci95_percentage_points"]) == 2
    assert weighted["cross_host_identity_covariance_modeled"] is False
    for point in result["points"]:
        sampling = point["paired_gain_sampling"]
        assert sampling["model"] == "paired_outcome_normal_approximation"
        assert sampling["gain_percentage_points"] == (
            point["gain_over_unchanged_percentage_points"]
        )
        assert sampling["standard_error_percentage_points"] > 0
        assert (
            sampling["ci95_percentage_points"][0]
            < sampling["gain_percentage_points"]
            < sampling["ci95_percentage_points"][1]
        )
    assert [point["active_parameters"] for point in result["points"]] == [
        3_000_000_000,
        12_000_000_000,
        39_000_000_000,
    ]


def test_gpt_oss_point_completes_cross_family_curve(tmp_path: Path) -> None:
    points = [
        _write(tmp_path / "qwen.json", _qwen_revision()),
        _write(
            tmp_path / "gpt_oss.json",
            _matched(
                "shohin-gpt-oss-120b-fixed-draft-screen-score-v1",
                "openai/gpt-oss-120b",
                117_000_000_000,
                5_100_000_000,
                31,
            ),
        ),
        _write(
            tmp_path / "mixtral.json",
            _matched(
                "shohin-mixtral-8x22b-fixed-draft-screen-score-v1",
                "mistralai/Mixtral-8x22B-Instruct-v0.1",
                141_000_000_000,
                39_000_000_000,
                36,
            ),
        ),
    ]
    result = analyze(points)
    assert result["status"] == "complete_curve"
    assert result["point_count"] == 3
    assert [point["host"] for point in result["points"]] == [
        "Qwen3.6-35B-A3B",
        "openai/gpt-oss-120b",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
    ]
    assert result["capability_curve_claim"] == (
        "positive_upward_cross_family_moe_capability_scaling_supported"
    )


def test_raw_qwen_external_score_normalizes_without_summary_copy(
    tmp_path: Path,
) -> None:
    point = normalize_point(_write(tmp_path / "qwen_raw.json", _qwen_external_score()))
    assert point["host"] == "Qwen3.6-35B-A3B"
    assert point["treatment_correct"] == 141
    assert point["unchanged_correct"] == 111
    assert point["paired_wins"] == 36
    assert point["paired_losses"] == 6
    assert point["unchanged_correct_retained"] == 105


def test_raw_qwen_external_outcome_tamper_fails(tmp_path: Path) -> None:
    payload = _qwen_external_score()
    payload["outcomes"][0]["correct"]["revision"] = False
    with pytest.raises(UpwardMoEScalingError, match="outcome totals"):
        normalize_point(_write(tmp_path / "tampered.json", payload))


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


def test_exact_agreement_receives_finite_variance_floor(tmp_path: Path) -> None:
    matched = _matched(
        "shohin-nemotron-super-fixed-draft-screen-score-v1",
        "Nemotron-Super",
        120_000_000_000,
        12_000_000_000,
        0,
    )
    matched["revision_vs_unchanged"] = {
        "left_only_correct": 0,
        "right_only_correct": 0,
        "net_correct": 0,
        "mcnemar_exact_two_sided_p": 1.0,
    }
    result = analyze([_write(tmp_path / "agreement.json", matched)])
    sampling = result["points"][0]["paired_gain_sampling"]
    assert sampling["observed_discordant_rows"] == 0
    assert sampling["variance_floor_discordant_rows"] == 1
    assert sampling["standard_error_percentage_points"] > 0


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


def test_curve_job_is_dependency_safe_cpu_only_and_runtime_bound() -> None:
    source = JOB.read_text(encoding="utf-8")
    assert "#SBATCH --no-requeue" in source
    assert "#SBATCH --gres" not in source
    assert "RUNTIME_MANIFEST_SHA256" in source
    assert "q36_verify_runtime" in source
    assert source.count("--point") == 3
    assert "analyze_upward_moe_scaling.py" in source
    assert '[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]' in source
