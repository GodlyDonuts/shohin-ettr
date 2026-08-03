import hashlib
import json
from pathlib import Path

import pytest

from aggregate_product_rollouts import ProductRolloutAggregateError, aggregate


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload)
    return hashlib.sha256(payload.encode()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, list[Path]]:
    bank = tmp_path / "bank.jsonl"
    bank_rows = [
        {"identity_sha256": f"id-{index}", "training_group": group}
        for index, group in enumerate(("math", "science", "math", "science"))
    ]
    bank_sha = _write_jsonl(bank, bank_rows)
    reports: list[Path] = []
    for shard in range(2):
        candidates: list[dict] = []
        positives: list[dict] = []
        for index in range(shard * 2, shard * 2 + 2):
            identity = f"id-{index}"
            for sample in range(2):
                candidates.append(
                    {
                        "identity_sha256": identity,
                        "sample_index": sample,
                        "completion": f"response-{index}-{sample}",
                        "correct": sample == 0,
                    }
                )
            positives.append(
                {
                    "source_identity_sha256": identity,
                    "chosen_sample_index": 0,
                    "response": f"response-{index}-0",
                    "training_group": bank_rows[index]["training_group"],
                }
            )
        candidate_path = tmp_path / f"candidates-{shard}.jsonl"
        positive_path = tmp_path / f"positives-{shard}.jsonl"
        candidate_sha = _write_jsonl(candidate_path, candidates)
        positive_sha = _write_jsonl(positive_path, positives)
        report = {
            "schema": "shohin-hf-product-reasoning-rollouts-v1",
            "status": "complete",
            "model_root": "/model",
            "model_revision": "revision",
            "adapter_checkpoint": "/adapter",
            "data_sha256": bank_sha,
            "samples": 2,
            "max_new_tokens": 32,
            "skip": shard * 2,
            "count": 2,
            "prompt_batch_size": 2,
            "candidates_output": str(candidate_path),
            "candidates_sha256": candidate_sha,
            "positives_output": str(positive_path),
            "positives_sha256": positive_sha,
            "counters": {"positive_prompts": 2, "correct_candidates": 2},
        }
        report_path = tmp_path / f"report-{shard}.json"
        report_path.write_text(json.dumps(report))
        reports.append(report_path)
    return bank, reports


def test_aggregate_admits_complete_verified_fan(tmp_path: Path) -> None:
    bank, reports = _fixture(tmp_path)
    result = aggregate(
        bank,
        reports,
        tmp_path / "all-candidates.jsonl",
        tmp_path / "all-positives.jsonl",
        tmp_path / "aggregate.json",
        min_positive_total=4,
        min_positive_per_group=2,
    )
    assert result["admitted"]
    assert result["positive_prompts"] == 4
    assert result["positive_group_counts"] == {"math": 2, "science": 2}


def test_aggregate_rejects_slice_gap(tmp_path: Path) -> None:
    bank, reports = _fixture(tmp_path)
    payload = json.loads(reports[1].read_text())
    payload["skip"] = 3
    reports[1].write_text(json.dumps(payload))
    with pytest.raises(ProductRolloutAggregateError, match="gap or overlap"):
        aggregate(
            bank,
            reports,
            tmp_path / "all-candidates.jsonl",
            tmp_path / "all-positives.jsonl",
            tmp_path / "aggregate.json",
            min_positive_total=4,
            min_positive_per_group=2,
        )
