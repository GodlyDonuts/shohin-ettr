import argparse
import json

from aggregate_cte1_development import run


def report(control: str, correct: int, **counts):
    return {
        "schema": "shohin-cte1-development-evaluation-v1",
        "status": "complete",
        "control": control,
        "holdout_used": False,
        "public_test_opened": False,
        "model_revision": "r",
        "checkpoint_sha256": "c",
        "checkpoint_update": 1024,
        "training_data_sha256": "t",
        "development_data_sha256": "d",
        "lam_checkpoint_sha256": "l",
        "seed": 2026081053,
        "max_new_tokens": 512,
        "exhausted": 0,
        "counts": {"rows": 666, "correct": correct, **counts},
    }


def test_reducer_requires_every_gate(tmp_path) -> None:
    normal = tmp_path / "normal.json"
    shuffled = tmp_path / "shuffled.json"
    output = tmp_path / "aggregate.json"
    normal.write_text(
        json.dumps(
            report(
                "normal",
                300,
                compiled_rows=650,
                executable_rows=650,
                linked_rows=500,
                linked_correct=250,
                state_reset_linked_correct=100,
                opcode_permuted_correct=50,
                execution_invalid=0,
            )
        )
    )
    shuffled.write_text(json.dumps(report("source_shuffled", 10)))
    result = run(
        argparse.Namespace(normal=normal, source_shuffled=shuffled, output=output)
    )
    assert result["status"] == "pass"
    result["gates"]["aligned_at_least_300"] = False
    assert not all(result["gates"].values())
