from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.aggregate_sag1_product_gate import (
    BASE_CHECKPOINT_SHA256,
    SAG1AggregationError,
    TASKS,
    aggregate_sag1,
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


def _write_training(
    root: Path, arm: str, *, router_rates: tuple[float, ...] = (0.0, 1.0, 0.0)
) -> Path:
    path = root / f"{arm}_training.json"
    payload = {
        "arm": arm,
        "batch_size": 1,
        "charged_tokens": 500000 if arm == "baseline" else None,
        "data_seed": 20260802,
        "data_sha256": "2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549",
        "frozen_parameters_unchanged": arm == "diverge_sag1" or None,
        "gradient_accumulation": 16,
        "learning_rate": 2e-4,
        "logical_charged_tokens": 500000 if arm == "diverge_sag1" else None,
        "lora_alpha": 16.0,
        "lora_layers": 4,
        "lora_rank": 8,
        "max_sequence_length": 1024,
        "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "seed": 2026080711,
        "selected_rows": 26387,
        "status": "complete",
        "trace": [
            {
                "gradient_norm": 1.0,
                "language_loss": 0.5,
                "loss": 0.6,
                "router_commit_rate": router_rate,
            }
            for router_rate in router_rates
        ],
        "updates": 256,
        "warm_start_sha256": BASE_CHECKPOINT_SHA256 if arm == "baseline" else None,
        "warm_start_update": 256 if arm == "baseline" else None,
        "base_checkpoint_sha256": (
            BASE_CHECKPOINT_SHA256 if arm == "diverge_sag1" else None
        ),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run(
    root: Path,
    original: dict[str, int],
    continuation: dict[str, int],
    treatment: dict[str, int],
    *,
    router_rates: tuple[float, ...] = (0.0, 1.0, 0.0),
):
    return aggregate_sag1(
        original_prefix=_write_evals(root, "original", original),
        continuation_prefix=_write_evals(root, "continuation", continuation),
        treatment_prefix=_write_evals(root, "sag1", treatment),
        continuation_training=_write_training(root, "baseline"),
        treatment_training=_write_training(
            root, "diverge_sag1", router_rates=router_rates
        ),
    )


def test_gate_passes_broad_gain_over_both_references(tmp_path: Path) -> None:
    original = {
        "gsm8k": 80,
        "math500": 40,
        "humaneval": 15,
        "mbpp": 15,
        "gpqa": 30,
        "bbh_logic": 50,
        "aime": 2,
    }
    continuation = {**original, "gsm8k": 81, "math500": 41}
    treatment = {
        "gsm8k": 88,
        "math500": 49,
        "humaneval": 15,
        "mbpp": 15,
        "gpqa": 50,
        "bbh_logic": 56,
        "aime": 4,
    }
    report = _run(tmp_path, original, continuation, treatment)
    assert report["comparison"]["development_gate_pass"] is True
    assert report["required_next_step"].startswith("source_disjoint")
    assert report["promotion_authorized"] is False


def test_code_loss_or_collapsed_router_closes_sag1(tmp_path: Path) -> None:
    original = {task: min(30, TOTALS[task]) for task in TASKS}
    continuation = original.copy()
    treatment = {task: min(60, TOTALS[task]) for task in TASKS}
    treatment["humaneval"] = 14
    treatment["mbpp"] = 15
    report = _run(
        tmp_path, original, continuation, treatment, router_rates=(1.0, 1.0)
    )
    assert report["comparison"]["numeric_gates"][
        "retains_original_b1_code_30_of_40"
    ] is False
    assert report["comparison"]["training_gates"][
        "nontrivial_nonuniversal_router"
    ] is False
    assert report["required_next_step"] == "close_exact_sag1"


def test_unmatched_evaluator_configuration_fails_closed(tmp_path: Path) -> None:
    scores = {task: min(10, TOTALS[task]) for task in TASKS}
    original = _write_evals(tmp_path, "original", scores)
    continuation = _write_evals(tmp_path, "continuation", scores)
    treatment = _write_evals(
        tmp_path, "sag1", scores, changed_seed_task="gsm8k"
    )
    with pytest.raises(SAG1AggregationError, match="generation_seed"):
        aggregate_sag1(
            original_prefix=original,
            continuation_prefix=continuation,
            treatment_prefix=treatment,
            continuation_training=_write_training(tmp_path, "baseline"),
            treatment_training=_write_training(tmp_path, "diverge_sag1"),
        )
