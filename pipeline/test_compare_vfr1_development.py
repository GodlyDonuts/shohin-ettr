import argparse
import json
from pathlib import Path

from compare_vfr1_development import compare


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _evaluation(correct: int, *, math: int, logic: int, code: int) -> dict:
    return {
        "schema": "shohin-vfr1-capability-evaluation-v1",
        "status": "complete",
        "merged_from_shards": True,
        "full_row_count": 1289,
        "shard_count": 8,
        "model_root": "/model",
        "model_revision": "revision",
        "data_sha256": "development",
        "data_report_sha256": "report",
        "max_new_tokens": 1536,
        "batch_size": 2,
        "seed": 7,
        "parse_fraction": 0.98,
        "metrics": {
            "overall": {"generated_correct": correct, "total": 1289},
            "math500": {"generated_correct": math, "total": 500},
            "bbh_logic": {"generated_correct": logic, "total": 701},
            "mbpp": {"generated_correct": code, "total": 88},
        },
    }


def _fit(path: Path, warm: str) -> dict:
    return {
        "schema": "shohin-hf-product-reasoning-training-v1",
        "status": "complete",
        "updates": 256,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "max_sequence_length": 4096,
        "learning_rate": 2e-5,
        "lora_layers": 4,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "warm_start_update": 256,
        "warm_start_sha256": warm,
        "model_root": "/model",
        "model_revision": "revision",
        "model_loader": "multimodal",
        "trainable_parameters": 123,
        "charged_tokens": 456,
        "selected_rows": 9655,
        "seed": 11,
        "data_seed": 12,
        "data_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
    }


def test_gate_requires_absolute_and_matched_margin(tmp_path: Path) -> None:
    treatment_data = tmp_path / "treatment.jsonl"
    shuffled_data = tmp_path / "shuffled.jsonl"
    treatment_data.write_text("treatment\n", encoding="utf-8")
    shuffled_data.write_text("shuffled\n", encoding="utf-8")
    data_report = tmp_path / "data.json"
    _write(
        data_report,
        {
            "status": "complete",
            "outputs": {
                "train_treatment": {"path": str(treatment_data)},
                "train_shuffled": {"path": str(shuffled_data)},
            },
        },
    )
    paths = {name: tmp_path / f"{name}.json" for name in ("tr", "sh", "tr_fit", "sh_fit")}
    _write(paths["tr"], _evaluation(603, math=223, logic=349, code=17))
    _write(paths["sh"], _evaluation(593, math=220, logic=355, code=18))
    _write(paths["tr_fit"], _fit(treatment_data, "warm"))
    _write(paths["sh_fit"], _fit(shuffled_data, "warm"))
    output = tmp_path / "comparison.json"
    result = compare(
        argparse.Namespace(
            treatment_report=paths["tr"],
            shuffled_report=paths["sh"],
            treatment_fit=paths["tr_fit"],
            shuffled_fit=paths["sh_fit"],
            data_report=data_report,
            output=output,
        )
    )
    assert result["gate_pass"] is True
    assert result["treatment_minus_shuffled_answers"] == 10
    assert output.is_file()


def test_margin_failure_closes_gate(tmp_path: Path) -> None:
    treatment_data = tmp_path / "treatment.jsonl"
    shuffled_data = tmp_path / "shuffled.jsonl"
    treatment_data.write_text("treatment\n", encoding="utf-8")
    shuffled_data.write_text("shuffled\n", encoding="utf-8")
    data_report = tmp_path / "data.json"
    _write(
        data_report,
        {
            "status": "complete",
            "outputs": {
                "train_treatment": {"path": str(treatment_data)},
                "train_shuffled": {"path": str(shuffled_data)},
            },
        },
    )
    treatment_report = tmp_path / "treatment_report.json"
    shuffled_report = tmp_path / "shuffled_report.json"
    treatment_fit = tmp_path / "treatment_fit.json"
    shuffled_fit = tmp_path / "shuffled_fit.json"
    _write(treatment_report, _evaluation(603, math=223, logic=349, code=17))
    _write(shuffled_report, _evaluation(600, math=220, logic=360, code=20))
    _write(treatment_fit, _fit(treatment_data, "warm"))
    _write(shuffled_fit, _fit(shuffled_data, "warm"))
    result = compare(
        argparse.Namespace(
            treatment_report=treatment_report,
            shuffled_report=shuffled_report,
            treatment_fit=treatment_fit,
            shuffled_fit=shuffled_fit,
            data_report=data_report,
            output=tmp_path / "comparison.json",
        )
    )
    assert result["gate_pass"] is False
    assert result["decision"] == "close_exact_vfr1_without_rescue"
