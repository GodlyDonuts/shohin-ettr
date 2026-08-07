from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.aggregate_qpt1_product_gate import (
    QPT1AggregationError,
    TASKS,
    aggregate_qpt1,
)


TOTALS = {
    "gsm8k": 100,
    "math500": 100,
    "humaneval": 20,
    "mbpp": 20,
    "gpqa": 198,
    "bbh_logic": 100,
    "aime": 30,
}


def _write_evals(
    root: Path,
    arm: str,
    scores: dict[str, int],
    *,
    changed_seed_task: str | None = None,
) -> Path:
    prefix = root / arm
    for task in TASKS:
        payload = {
            "correct": scores[task],
            "data_sha256": f"data-{task}",
            "effective_enable_thinking": False,
            "generation_mode": "greedy",
            "generation_seed": 99 if task == changed_seed_task else 31,
            "generation_stop_token_ids": [1, 2],
            "max_new_tokens": 1024 if task in {"humaneval", "mbpp", "aime"} else 768,
            "selection_sha256": f"selection-{task}",
            "status": "complete",
            "subset_seed": 20260802,
            "task": task,
            "total": TOTALS[task],
        }
        Path(f"{prefix}_{task}.json").write_text(json.dumps(payload), encoding="utf-8")
    return prefix


def _write_training(root: Path, arm: str) -> Path:
    path = root / f"{arm}_training.json"
    payload = {
        "arm": arm,
        "batch_size": 1,
        "data_seed": 20260802,
        "data_sha256": "v10",
        "frozen_parameters_unchanged": arm == "diverge_qpt1" or None,
        "gradient_accumulation": 16,
        "learning_rate": 2e-4,
        "lora_alpha": 16.0,
        "lora_layers": 4,
        "lora_rank": 8,
        "max_sequence_length": 1024,
        "model_revision": "qwen4b",
        "seed": 2026080702,
        "selected_rows": 26387,
        "status": "complete",
        "trace": [{"gradient_norm": 1.0, "language_loss": 0.5, "loss": 0.6}],
        "updates": 256,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run(root: Path, baseline: dict[str, int], treatment: dict[str, int]):
    return aggregate_qpt1(
        baseline_prefix=_write_evals(root, "b1", baseline),
        treatment_prefix=_write_evals(root, "qpt1", treatment),
        baseline_training=_write_training(root, "baseline"),
        treatment_training=_write_training(root, "diverge_qpt1"),
    )


def test_frozen_gate_passes_material_broad_gain(tmp_path: Path) -> None:
    baseline = {
        "gsm8k": 50,
        "math500": 20,
        "humaneval": 2,
        "mbpp": 2,
        "gpqa": 10,
        "bbh_logic": 20,
        "aime": 0,
    }
    treatment = {
        "gsm8k": 56,
        "math500": 26,
        "humaneval": 3,
        "mbpp": 2,
        "gpqa": 20,
        "bbh_logic": 20,
        "aime": 2,
    }
    report = _run(tmp_path, baseline, treatment)
    comparison = report["comparison"]
    assert comparison["score_and_training_gate_pass"] is True
    assert comparison["solved_delta_qpt1_vs_b1"] == 23
    assert report["promotion_authorized"] is False
    assert report["required_next_step"].endswith("controls")


def test_large_domain_regression_closes_qpt1(tmp_path: Path) -> None:
    baseline = {task: min(10, TOTALS[task]) for task in TASKS}
    treatment = {task: min(20, TOTALS[task]) for task in TASKS}
    treatment["bbh_logic"] = 0
    report = _run(tmp_path, baseline, treatment)
    gates = report["comparison"]["numeric_gates"]
    assert gates["no_domain_regression_over_two_points"] is False
    assert report["comparison"]["score_and_training_gate_pass"] is False
    assert report["required_next_step"] == "close_exact_qpt1"


def test_unmatched_evaluator_configuration_fails_closed(tmp_path: Path) -> None:
    scores = {task: min(10, TOTALS[task]) for task in TASKS}
    baseline_prefix = _write_evals(tmp_path, "b1", scores)
    treatment_prefix = _write_evals(tmp_path, "qpt1", scores, changed_seed_task="gsm8k")
    with pytest.raises(QPT1AggregationError, match="generation_seed"):
        aggregate_qpt1(
            baseline_prefix=baseline_prefix,
            treatment_prefix=treatment_prefix,
            baseline_training=_write_training(tmp_path, "baseline"),
            treatment_training=_write_training(tmp_path, "diverge_qpt1"),
        )
