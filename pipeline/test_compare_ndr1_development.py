import argparse
import hashlib
import json
from pathlib import Path

from compare_ndr1_development import compare


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value))


def evaluation(correct: int, math: int, logic: int, code: int, exhausted: int) -> dict:
    return {
        "schema": "shohin-idr1-revision-evaluation-v1",
        "status": "complete",
        "split": "development",
        "full_row_count": 1289,
        "merged_from_shards": True,
        "shard_count": 8,
        "model_root": "/model",
        "model_revision": "revision",
        "data_sha256": "eval",
        "data_report_sha256": "eval-report",
        "max_token_exhausted": exhausted,
        "metrics": {
            "overall": {"generated_correct": correct, "total": 1289},
            "math500": {"generated_correct": math, "total": 623},
            "bbh_logic": {"generated_correct": logic, "total": 637},
            "mbpp": {"generated_correct": code, "total": 29},
        },
    }


def fit(data: Path) -> dict:
    return {
        "schema": "shohin-hf-product-reasoning-training-v1",
        "status": "complete",
        "updates": 512,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "max_sequence_length": 4096,
        "learning_rate": 2e-5,
        "lora_layers": 4,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "warm_start_update": 256,
        "warm_start_sha256": "warm",
        "model_root": "/model",
        "model_revision": "revision",
        "model_loader": "multimodal",
        "trainable_parameters": 123,
        "charged_tokens": 456,
        "selected_rows": 10000,
        "seed": 2026080921,
        "data_seed": 2026080920,
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
    }


def test_ndr1_conjunctive_gate(tmp_path: Path) -> None:
    aligned_data = tmp_path / "aligned.jsonl"
    shuffled_data = tmp_path / "shuffled.jsonl"
    aligned_data.write_text("a\n")
    shuffled_data.write_text("s\n")
    data_report = tmp_path / "data.json"
    write(
        data_report,
        {
            "schema": "shohin-ndr1-natural-revision-data-report-v1",
            "status": "complete",
            "outputs": {
                "aligned": {"path": str(aligned_data)},
                "shuffled": {"path": str(shuffled_data)},
            },
        },
    )
    aligned_report = tmp_path / "aligned-report.json"
    shuffled_report = tmp_path / "shuffled-report.json"
    aligned_fit = tmp_path / "aligned-fit.json"
    shuffled_fit = tmp_path / "shuffled-fit.json"
    write(aligned_report, evaluation(603, 223, 349, 17, 325))
    write(shuffled_report, evaluation(593, 220, 355, 18, 300))
    write(aligned_fit, fit(aligned_data))
    write(shuffled_fit, fit(shuffled_data))
    result = compare(
        argparse.Namespace(
            aligned_report=aligned_report,
            shuffled_report=shuffled_report,
            aligned_fit=aligned_fit,
            shuffled_fit=shuffled_fit,
            data_report=data_report,
            output=tmp_path / "comparison.json",
        )
    )
    assert result["gate_pass"] is True
    assert result["aligned_minus_shuffled_answers"] == 10


def test_ndr1_verbosity_gate_is_conjunctive(tmp_path: Path) -> None:
    aligned_data = tmp_path / "aligned.jsonl"
    shuffled_data = tmp_path / "shuffled.jsonl"
    aligned_data.write_text("a\n")
    shuffled_data.write_text("s\n")
    data_report = tmp_path / "data.json"
    write(
        data_report,
        {
            "schema": "shohin-ndr1-natural-revision-data-report-v1",
            "status": "complete",
            "outputs": {
                "aligned": {"path": str(aligned_data)},
                "shuffled": {"path": str(shuffled_data)},
            },
        },
    )
    aligned_report = tmp_path / "aligned-report.json"
    shuffled_report = tmp_path / "shuffled-report.json"
    aligned_fit = tmp_path / "aligned-fit.json"
    shuffled_fit = tmp_path / "shuffled-fit.json"
    write(aligned_report, evaluation(620, 230, 360, 18, 401))
    write(shuffled_report, evaluation(590, 220, 350, 18, 300))
    write(aligned_fit, fit(aligned_data))
    write(shuffled_fit, fit(shuffled_data))
    result = compare(
        argparse.Namespace(
            aligned_report=aligned_report,
            shuffled_report=shuffled_report,
            aligned_fit=aligned_fit,
            shuffled_fit=shuffled_fit,
            data_report=data_report,
            output=tmp_path / "comparison.json",
        )
    )
    assert result["gate_pass"] is False
    assert result["gates"]["aligned_exhaustion_at_most_400"] is False
