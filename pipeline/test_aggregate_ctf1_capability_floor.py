import argparse
import json
from pathlib import Path

from aggregate_ctf1_capability_floor import run


def _report(control: str, correct: int) -> dict:
    return {
        "schema": "shohin-ctf1-capability-floor-evaluation-v1",
        "status": "complete",
        "holdout_used": False,
        "public_test_opened": False,
        "control": control,
        "model_revision": "revision",
        "development_data_sha256": "data",
        "lam_checkpoint_sha256": "lam",
        "seed": 2026081053,
        "max_new_tokens": 512,
        "exhausted": 0,
        "counts": {
            "rows": 666,
            "correct": correct,
            "compiled_rows": 620,
            "executable_rows": 620,
            "linked_rows": 500,
            "linked_correct": 300,
            "state_reset_linked_correct": 0,
            "opcode_permuted_correct": 0,
            "execution_invalid": 0,
        },
    }


def test_full_pass(tmp_path: Path) -> None:
    normal = tmp_path / "normal.json"
    shuffled = tmp_path / "shuffled.json"
    output = tmp_path / "aggregate.json"
    normal.write_text(json.dumps(_report("normal", 320)))
    shuffled.write_text(json.dumps(_report("source_shuffled", 4)))
    result = run(argparse.Namespace(normal=normal, source_shuffled=shuffled, output=output))
    assert result["status"] == "pass"


def test_capability_miss_fails(tmp_path: Path) -> None:
    normal = tmp_path / "normal.json"
    shuffled = tmp_path / "shuffled.json"
    output = tmp_path / "aggregate.json"
    normal.write_text(json.dumps(_report("normal", 299)))
    shuffled.write_text(json.dumps(_report("source_shuffled", 4)))
    result = run(argparse.Namespace(normal=normal, source_shuffled=shuffled, output=output))
    assert result["status"] == "fail"
