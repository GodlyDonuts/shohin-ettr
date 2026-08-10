import argparse
import hashlib
import json
from pathlib import Path

from analyze_ectr0_outcomes import canonical_prediction, run


def _write_arm(
    path: Path,
    control: str,
    shard_index: int,
    start: int,
    end: int,
    correct_indices: set[int],
) -> None:
    details = []
    for index in range(start, end):
        direct_prediction = "1" if index < 487 else "0"
        executor_prediction = "1" if index % 2 == 0 else "2"
        prediction = direct_prediction
        if control == "aligned" and index % 3 == 0:
            prediction = executor_prediction
        elif control == "receipt_shuffled" and index % 5 == 0:
            prediction = "3"
        details.append(
            {
                "identity_sha256": f"{index:064x}",
                "correct": index in correct_indices,
                "direct_prediction": direct_prediction,
                "executor_prediction": executor_prediction,
                "executor_correct": index % 2 == 0,
                "prediction": prediction,
            }
        )
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
                "seed": 1,
                "max_new_tokens": 512,
                "max_sequence_length": 4096,
                "batch_size": 4,
                "shard_count": 2,
                "shard_index": shard_index,
                "generated_tokens": end - start,
                "elapsed_seconds": 1.0,
                "peak_gpu_memory_bytes": 1,
                "max_input_tokens": 100,
                "counts": {
                    "rows": end - start,
                    "correct": sum(index in correct_indices for index in range(start, end)),
                },
                "details": details,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_canonical_prediction_equates_numeric_surfaces() -> None:
    assert canonical_prediction("1,200") == canonical_prediction("1200.0")
    assert canonical_prediction("0.5") == canonical_prediction("1/2")


def test_attribution_covers_all_paired_rows(tmp_path: Path) -> None:
    arm_paths = {}
    correct = {
        "aligned": set(range(476)),
        "receipt_absent": set(range(479)),
        "receipt_shuffled": set(range(468)),
    }
    for control in correct:
        paths = []
        for shard_index, (start, end) in enumerate(((0, 333), (333, 666))):
            path = tmp_path / f"{control}_{shard_index}.json"
            _write_arm(path, control, shard_index, start, end, correct[control])
            paths.append(path)
        arm_paths[control] = paths

    data = tmp_path / "data.jsonl"
    data.write_text(
        "".join(
            json.dumps({"identity_sha256": f"{index:064x}", "gold_answer": "1"}) + "\n"
            for index in range(666)
        ),
        encoding="utf-8",
    )
    ctf = tmp_path / "ctf.json"
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
    output = tmp_path / "attribution.json"
    report = run(
        argparse.Namespace(
            aligned_report=arm_paths["aligned"],
            absent_report=arm_paths["receipt_absent"],
            shuffled_report=arm_paths["receipt_shuffled"],
            data=data,
            expected_data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
            ctf_report=ctf,
            expected_ctf_sha256=hashlib.sha256(ctf.read_bytes()).hexdigest(),
            output=output,
        )
    )
    assert sum(row["rows"] for row in report["outcome_matrix"]) == 666
    assert report["correct"]["direct"] == 487
    assert report["correct"]["aligned"] == 476
    assert report["paired"]["aligned_vs_absent"]["second_only_correct"] == 3
    assert report["prediction_behavior"]["direct_executor_disagree"] == 422
