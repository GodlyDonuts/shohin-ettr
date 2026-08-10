import argparse
import json
from pathlib import Path

from aggregate_ectr0_executor_revision import run


def _write(path: Path, control: str, start: int, end: int, correct: int) -> None:
    details = []
    for index in range(start, end):
        details.append({"identity_sha256": f"{index:064x}"})
    counts = {
        "rows": end - start,
        "correct": correct,
        "direct_correct": 244 if start == 0 else 243,
        "explicit_final": end - start,
        "repairs": 20 if start == 0 else 20,
        "breaks": 10 if start == 0 else 10,
    }
    path.write_text(
        json.dumps(
            {
                "schema": "shohin-ectr0-executor-conditioned-revision-v1",
                "status": "complete",
                "holdout_used": False,
                "public_test_opened": False,
                "control": control,
                "model_revision": "revision",
                "adapter_checkpoint_sha256": "a" * 64,
                "data_sha256": "b" * 64,
                "ctf_report_sha256": "c" * 64,
                "seed": 2026081061,
                "max_new_tokens": 512,
                "max_sequence_length": 4096,
                "batch_size": 4,
                "shard_count": 2,
                "shard_index": 0 if start == 0 else 1,
                "generated_tokens": 10,
                "elapsed_seconds": 1.0,
                "peak_gpu_memory_bytes": 1,
                "max_input_tokens": 100,
                "counts": counts,
                "details": details,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_aggregate_pass(tmp_path: Path) -> None:
    paths = {}
    for control, scores in {
        "aligned": (260, 250),
        "receipt_absent": (250, 245),
        "receipt_shuffled": (249, 245),
    }.items():
        arm = []
        for shard, (start, end, score) in enumerate(((0, 336, scores[0]), (336, 666, scores[1]))):
            path = tmp_path / f"{control}_{shard}.json"
            _write(path, control, start, end, score)
            arm.append(path)
        paths[control] = arm
    output = tmp_path / "aggregate.json"
    result = run(
        argparse.Namespace(
            aligned_report=paths["aligned"],
            absent_report=paths["receipt_absent"],
            shuffled_report=paths["receipt_shuffled"],
            output=output,
        )
    )
    assert result["gate_pass"] is True
    assert result["deltas"]["aligned_minus_direct"] == 23
