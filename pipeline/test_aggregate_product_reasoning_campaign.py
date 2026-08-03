from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.aggregate_product_reasoning_campaign import (
    CampaignAggregationError,
    TASKS,
    aggregate_campaign,
)


def _write_arm(
    root: Path,
    arm: str,
    scores: dict[str, tuple[int, int]],
    *,
    changed_seed_task: str | None = None,
) -> Path:
    prefix = root / arm
    for task in TASKS:
        correct, total = scores[task]
        report = {
            "correct": correct,
            "data_sha256": f"data-{task}",
            "effective_enable_thinking": False,
            "generation_mode": "greedy",
            "generation_seed": 31 if task != changed_seed_task else 99,
            "max_new_tokens": 1024 if task in {"humaneval", "mbpp"} else 768,
            "selection_sha256": f"selection-{task}",
            "status": "complete",
            "subset_seed": 20260802,
            "task": task,
            "total": total,
        }
        path = Path(f"{prefix}_{task}_dev_v2.json")
        path.write_text(json.dumps(report), encoding="utf-8")
    return prefix


def _scores(value: int) -> dict[str, tuple[int, int]]:
    return {
        "gsm8k": (value, 100),
        "math500": (value, 100),
        "humaneval": (value // 5, 20),
        "mbpp": (value // 5, 20),
        "gpqa": (value, 198),
        "bbh_logic": (value, 100),
    }


def test_campaign_passes_only_when_treatment_beats_both_arms(tmp_path: Path) -> None:
    baseline = _write_arm(tmp_path, "b1", _scores(10))
    treatment = _write_arm(tmp_path, "t2", _scores(15))
    control = _write_arm(tmp_path, "c2", _scores(12))

    report = aggregate_campaign(
        baseline_name="B1",
        baseline_prefix=baseline,
        treatment_name="T2",
        treatment_prefix=treatment,
        control_name="C2",
        control_prefix=control,
    )

    assert report["comparison"]["numeric_gate_pass"] is True
    assert report["comparison"]["improved_domain_count"] == 5
    assert report["comparison"]["transcript_coherence_gate"] == "manual_review_required"


def test_dense_control_win_rejects_recurrence(tmp_path: Path) -> None:
    baseline = _write_arm(tmp_path, "b1", _scores(10))
    treatment = _write_arm(tmp_path, "t2", _scores(15))
    control = _write_arm(tmp_path, "c2", _scores(16))

    report = aggregate_campaign(
        baseline_name="B1",
        baseline_prefix=baseline,
        treatment_name="T2",
        treatment_prefix=treatment,
        control_name="C2",
        control_prefix=control,
    )

    gates = report["comparison"]["numeric_gates"]
    assert gates["treatment_beats_dense_control"] is False
    assert report["comparison"]["numeric_gate_pass"] is False


def test_unmatched_decode_configuration_fails_closed(tmp_path: Path) -> None:
    baseline = _write_arm(tmp_path, "b1", _scores(10))
    treatment = _write_arm(
        tmp_path, "t2", _scores(15), changed_seed_task="gsm8k"
    )
    control = _write_arm(tmp_path, "c2", _scores(12))

    with pytest.raises(CampaignAggregationError, match="unmatched gsm8k"):
        aggregate_campaign(
            baseline_name="B1",
            baseline_prefix=baseline,
            treatment_name="T2",
            treatment_prefix=treatment,
            control_name="C2",
            control_prefix=control,
        )
