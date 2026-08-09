"""CPU tests for the frozen ESR1 comparison."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import tempfile

from compare_esr1_development import compare


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _evaluation(correct: int, domains: tuple[int, int, int]) -> dict:
    return {
        "schema": "shohin-idr1-revision-evaluation-v1",
        "status": "complete",
        "split": "development",
        "model_revision": "revision",
        "data_sha256": "data",
        "data_report_sha256": "report",
        "full_row_count": 1289,
        "merged_from_shards": True,
        "shard_count": 8,
        "metrics": {
            "overall": {"generated_correct": correct, "total": 1289},
            "math500": {"generated_correct": domains[0], "total": 500},
            "bbh_logic": {"generated_correct": domains[1], "total": 533},
            "mbpp": {"generated_correct": domains[2], "total": 256},
        },
    }


def _fit(arm: str) -> dict:
    return {
        "schema": "shohin-hf-product-reasoning-training-v1",
        "status": "complete",
        "arm": arm,
        "architecture": (
            "shohin-error-syndrome-revision-v1"
            if arm == "syndrome"
            else "shohin-product-reasoning-v1"
        ),
        "model_root": "/model",
        "model_revision": "revision",
        "data_sha256": "train",
        "selected_rows": 9655,
        "seed": 1,
        "data_seed": 2,
        "lora_layers": 4,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "unfreeze_layers": 0,
        "workspace_config": {"workspace_width": 512},
        "updates": 256,
        "gradient_accumulation": 8,
        "batch_size": 1,
        "max_sequence_length": 4096,
        "learning_rate": 2e-5,
    }


def test_esr1_gate_requires_both_margins_and_domain_nonregression() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        paths = {name: root / f"{name}.json" for name in ("s", "e", "a", "sf", "ef")}
        _write(paths["s"], _evaluation(340, (100, 220, 20)))
        _write(paths["e"], _evaluation(290, (85, 190, 15)))
        _write(paths["a"], _evaluation(259, (80, 170, 9)))
        _write(paths["sf"], _fit("syndrome"))
        _write(paths["ef"], _fit("ettr"))
        output = root / "comparison.json"
        result = compare(
            Namespace(
                syndrome_report=paths["s"],
                ettr_report=paths["e"],
                always_report=paths["a"],
                syndrome_fit_report=paths["sf"],
                ettr_fit_report=paths["ef"],
                output=output,
            )
        )
        assert result["gate_pass"] is True
        assert result["holdout_authorized"] is True
