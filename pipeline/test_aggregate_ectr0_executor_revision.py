import argparse
import hashlib
import json
from pathlib import Path

from aggregate_ectr0_executor_revision import run


def _write(path: Path, control: str, start: int, end: int, correct: int) -> None:
    details = []
    for offset, index in enumerate(range(start, end)):
        details.append({"identity_sha256": f"{index:064x}", "correct": offset < correct})
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
    data = tmp_path / "data.jsonl"
    ctf = tmp_path / "ctf.json"
    data.write_text(
        "".join(
            json.dumps({"identity_sha256": f"{index:064x}", "gold_answer": "1"}) + "\n"
            for index in range(666)
        ),
        encoding="utf-8",
    )
    ctf.write_text(
        json.dumps(
            {
                "details": [
                    {
                        "identity_sha256": f"{index:064x}",
                        "completion": "#### 1" if index < 487 else "#### 0",
                    }
                    for index in range(666)
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = run(
        argparse.Namespace(
            aligned_report=paths["aligned"],
            absent_report=paths["receipt_absent"],
            shuffled_report=paths["receipt_shuffled"],
            data=data,
            expected_data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
            ctf_report=ctf,
            expected_ctf_sha256=hashlib.sha256(ctf.read_bytes()).hexdigest(),
            output=output,
        )
    )
    assert result["gate_pass"] is True
    assert result["deltas"]["aligned_minus_direct"] == 23
