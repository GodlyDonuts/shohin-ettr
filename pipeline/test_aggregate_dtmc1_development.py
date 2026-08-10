import argparse
import json

from aggregate_dtmc1_development import run


def _report(control: str, correct: int) -> dict:
    return {
        "schema": "shohin-dtmc1-development-evaluation-v1",
        "status": "complete",
        "holdout_used": False,
        "public_test_opened": False,
        "control": control,
        "model_revision": "revision",
        "owner_checkpoint_sha256": "a" * 64,
        "compiler_checkpoint_sha256": "b" * 64,
        "compiler_updates": 4096,
        "compiler_training_data_sha256": "c" * 64,
        "development_data_sha256": "d" * 64,
        "draft_report_sha256": "e" * 64,
        "lam_checkpoint_sha256": "f" * 64,
        "counts": {
            "rows": 666,
            "normal:correct": correct,
            "normal:invalid": 0,
            "operation_fields": 1000,
            "operation_correct": 900,
            "operand_fields": 2000,
            "operand_correct": 1800,
            "normal_correct_multi_digit_rows": 300,
            "carry_reset:normal_correct_multi_digit_correct": 30,
            "opcode_permuted:correct": 30,
        },
    }


def test_aggregate_passes_only_conjunctive_gate(tmp_path) -> None:
    paths = {}
    for control, correct in (
        ("normal", 320),
        ("draft_shuffled", 220),
        ("source_draft_shuffled", 30),
    ):
        path = tmp_path / f"{control}.json"
        path.write_text(json.dumps(_report(control, correct)))
        paths[control] = path
    output = tmp_path / "aggregate.json"
    result = run(
        argparse.Namespace(
            normal=paths["normal"],
            draft_shuffled=paths["draft_shuffled"],
            source_draft_shuffled=paths["source_draft_shuffled"],
            output=output,
        )
    )
    assert result["status"] == "pass"
    assert all(result["gates"].values())


def test_aggregate_rejects_decorative_draft(tmp_path) -> None:
    paths = {}
    for control, correct in (
        ("normal", 320),
        ("draft_shuffled", 300),
        ("source_draft_shuffled", 30),
    ):
        path = tmp_path / f"{control}.json"
        path.write_text(json.dumps(_report(control, correct)))
        paths[control] = path
    result = run(
        argparse.Namespace(
            normal=paths["normal"],
            draft_shuffled=paths["draft_shuffled"],
            source_draft_shuffled=paths["source_draft_shuffled"],
            output=tmp_path / "aggregate.json",
        )
    )
    assert result["status"] == "fail"
    assert not result["gates"]["draft_margin_at_least_10_points"]
