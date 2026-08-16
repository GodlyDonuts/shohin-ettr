"""Tests for evidence-driven larger-MoE temporal promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from select_upward_moe_temporal_promotion import (
    UpwardMoETemporalPromotionError,
    atomic_json,
    select,
)

DOMAINS = {"bbh_logic": 128, "math500": 117, "mbpp": 11}


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _matched(
    *, schema: str, host: str, total: int, active: int, gain: int, retention: int = 108
) -> dict:
    unchanged = 112
    revision = unchanged + gain
    wins = max(gain, 0) + 3
    losses = wins - gain
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
    return {
        "schema": schema,
        "status": "complete",
        "host": host,
        "total_parameters": total,
        "active_parameters": active,
        "rows": 256,
        "arms": {
            "unchanged": {"correct": unchanged, "domains": unchanged_domains},
            "self_refinement": {
                "correct": 120,
                "domains": {
                    "bbh_logic": {"correct": 80, "total": 128},
                    "math500": {"correct": 31, "total": 117},
                    "mbpp": {"correct": 9, "total": 11},
                },
            },
            "revision": {
                "correct": revision,
                "domains": revision_domains,
                "unchanged_correct_retained": retention,
                "unchanged_correct_retention": retention / unchanged,
            },
        },
        "revision_vs_unchanged": {
            "left_only_correct": wins,
            "right_only_correct": losses,
            "net_correct": gain,
            "mcnemar_exact_two_sided_p": 0.01,
        },
    }


def _points(tmp_path: Path, super_gain: int = 24, mixtral_gain: int = 28) -> list[Path]:
    return [
        _write(
            tmp_path / "super.json",
            _matched(
                schema="shohin-nemotron-super-fixed-draft-screen-score-v1",
                host="NVIDIA-Nemotron-3-Super-120B-A12B-FP8",
                total=120_000_000_000,
                active=12_000_000_000,
                gain=super_gain,
            ),
        ),
        _write(
            tmp_path / "mixtral.json",
            _matched(
                schema="shohin-mixtral-8x22b-fixed-draft-screen-score-v1",
                host="mistralai/Mixtral-8x22B-Instruct-v0.1",
                total=141_000_000_000,
                active=39_000_000_000,
                gain=mixtral_gain,
            ),
        ),
    ]


def test_selects_largest_capability_gain(tmp_path: Path) -> None:
    result = select(_points(tmp_path))
    assert result["status"] == "promote"
    assert result["selected_dispatcher_host"] == "mixtral-8x22b"
    assert all(candidate["qualifies"] for candidate in result["candidates"])
    assert result["automatic_launch"] is False


def test_equal_capability_prefers_larger_active_host(tmp_path: Path) -> None:
    result = select(_points(tmp_path, super_gain=28, mixtral_gain=28))
    assert result["selected_dispatcher_host"] == "mixtral-8x22b"


def test_retention_failure_excludes_otherwise_stronger_host(tmp_path: Path) -> None:
    paths = _points(tmp_path, super_gain=24, mixtral_gain=40)
    mixtral = json.loads(paths[1].read_text(encoding="utf-8"))
    mixtral["arms"]["revision"]["unchanged_correct_retained"] = 106
    mixtral["arms"]["revision"]["unchanged_correct_retention"] = 106 / 112
    paths[1].write_text(json.dumps(mixtral) + "\n", encoding="utf-8")
    result = select(paths)
    assert result["selected_dispatcher_host"] == "nemotron-super"
    assert result["candidates"][1]["qualifies"] is False


def test_domain_regression_can_close_without_promotion(tmp_path: Path) -> None:
    paths = _points(tmp_path, super_gain=1, mixtral_gain=1)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        revision = payload["arms"]["revision"]
        revision["domains"]["bbh_logic"]["correct"] -= 2
        revision["domains"]["math500"]["correct"] += 2
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    result = select(paths)
    assert result["status"] == "no_qualifying_larger_host"
    assert result["selected_host"] is None
    assert result["next_action"].startswith("preserve_both")


def test_paired_delta_tamper_fails_closed(tmp_path: Path) -> None:
    paths = _points(tmp_path)
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    payload["revision_vs_unchanged"]["net_correct"] += 1
    paths[0].write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(UpwardMoETemporalPromotionError, match="validation"):
        select(paths)


def test_host_substitution_and_duplicate_fail_closed(tmp_path: Path) -> None:
    paths = _points(tmp_path)
    forged = json.loads(paths[1].read_text(encoding="utf-8"))
    forged["host"] = "unmeasured-host"
    paths[1].write_text(json.dumps(forged) + "\n", encoding="utf-8")
    with pytest.raises(UpwardMoETemporalPromotionError, match="identity"):
        select(paths)
    with pytest.raises(UpwardMoETemporalPromotionError, match="exactly"):
        select(paths[:1])


def test_atomic_output_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "decision.json"
    atomic_json(output, {"schema": "test"})
    with pytest.raises(UpwardMoETemporalPromotionError, match="exists"):
        atomic_json(output, {"schema": "test"})


def test_source_hashes_are_bound(tmp_path: Path) -> None:
    paths = _points(tmp_path)
    result = select(paths)
    assert {candidate["source_path"] for candidate in result["candidates"]} == {
        str(path.resolve()) for path in paths
    }
    assert all(
        len(candidate["source_sha256"]) == 64 for candidate in result["candidates"]
    )
