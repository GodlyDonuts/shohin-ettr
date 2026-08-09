import json
from pathlib import Path

from compare_mpr1_development import compare


def write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value))
    return path


def evaluation(correct: int, math: int, logic: int, code: int) -> dict:
    return {
        "schema": "shohin-idr1-revision-evaluation-v1",
        "status": "complete",
        "split": "development",
        "full_row_count": 1289,
        "merged_from_shards": True,
        "shard_count": 8,
        "model_root": "model",
        "model_revision": "revision",
        "data_sha256": "development",
        "data_report_sha256": "development-report",
        "metrics": {
            "overall": {"generated_correct": correct, "total": 1289},
            "math500": {"generated_correct": math, "total": 623},
            "bbh_logic": {"generated_correct": logic, "total": 637},
            "mbpp": {"generated_correct": code, "total": 29},
        },
    }


def fit(data_sha: str, control: str) -> dict:
    return {
        "schema": "shohin-rme1-product-training-v1",
        "status": "complete",
        "updates": 256,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "max_sequence_length": 4096,
        "learning_rate": 2e-5,
        "seed": 2026080901,
        "data_seed": 2026080814,
        "data_sha256": data_sha,
        "trainable_parameters": 1_179_648,
        "protected_router_expert_trainables": 0,
        "rme1_draft_control": control,
        "rme1_config": {
            "mode": "shared",
            "controlled_layers": 16,
            "rank": 18,
            "alpha": 18.0,
        },
        "model_root": "model",
        "model_revision": "revision",
        "charged_tokens": 1234,
        "selected_rows": 100,
        "adapter_macs_per_token_per_layer": 73728,
    }


def args(tmp_path: Path, aligned_correct: int = 240):
    aligned_data = write(tmp_path / "aligned.jsonl", {"row": "aligned"})
    shuffled_data = write(tmp_path / "shuffled.jsonl", {"row": "shuffled"})
    import hashlib

    aligned_sha = hashlib.sha256(aligned_data.read_bytes()).hexdigest()
    shuffled_sha = hashlib.sha256(shuffled_data.read_bytes()).hexdigest()
    reports = {
        "aligned_report": write(tmp_path / "aligned_report.json", evaluation(aligned_correct, 50, 180, 10)),
        "shuffled_report": write(tmp_path / "shuffled_report.json", evaluation(220, 45, 168, 7)),
        "hidden_report": write(tmp_path / "hidden_report.json", evaluation(221, 45, 169, 7)),
        "unchanged_report": write(tmp_path / "unchanged_report.json", evaluation(191, 40, 145, 5)),
        "aligned_fit": write(tmp_path / "aligned_fit.json", fit(aligned_sha, "normal")),
        "shuffled_fit": write(tmp_path / "shuffled_fit.json", fit(shuffled_sha, "normal")),
        "hidden_fit": write(tmp_path / "hidden_fit.json", fit(aligned_sha, "draft_unavailable")),
        "data_report": write(
            tmp_path / "data_report.json",
            {
                "schema": "shohin-mpr1-revision-data-report-v1",
                "status": "complete",
                "complete_retention": True,
                "holdout_used": False,
                "outputs": {
                    "aligned": {"sha256": aligned_sha},
                    "shuffled": {"sha256": shuffled_sha},
                },
            },
        ),
        "semantic_attribution": write(
            tmp_path / "semantic.json",
            {
                "schema": "shohin-moe-semantic-repair-attribution-v1",
                "status": "complete",
                "counts": {
                    "remaining_possible_semantic_repairs": 30,
                    "strict_breaks": 10,
                },
            },
        ),
        "output": tmp_path / "comparison.json",
    }
    from argparse import Namespace

    return Namespace(**reports)


def test_mpr1_gate_passes_only_full_conjunction(tmp_path):
    result = compare(args(tmp_path))
    assert result["gate_pass"] is True
    assert result["holdout_authorized"] is True
    assert result["margins"] == {
        "aligned_minus_unchanged": 49,
        "aligned_minus_shuffled": 20,
        "aligned_minus_hidden": 19,
    }


def test_mpr1_gate_closes_when_absolute_margin_fails(tmp_path):
    result = compare(args(tmp_path, aligned_correct=229))
    assert result["gate_pass"] is False
    assert result["holdout_authorized"] is False
