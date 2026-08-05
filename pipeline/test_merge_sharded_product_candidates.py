from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from merge_sharded_product_candidates import (
    ShardedCandidateMergeError,
    _identity,
    merge_shard_reports,
)


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _shard(tmp_path: Path, index: int, bank_rows: list[dict]) -> Path:
    bank = tmp_path / f"bank{index}.jsonl"
    candidates = tmp_path / f"candidates{index}.jsonl"
    report = tmp_path / f"report{index}.json"
    bank_sha = _write_jsonl(bank, bank_rows)
    candidate_rows = []
    for row in bank_rows:
        identity = _identity("mbpp", row)
        for sample in range(2):
            candidate_rows.append(
                {
                    "identity_sha256": identity,
                    "sample_index": sample,
                    "task": "mbpp",
                    "correct": sample == 0,
                }
            )
    candidate_sha = _write_jsonl(candidates, candidate_rows)
    report.write_text(
        json.dumps(
            {
                "schema": "shohin-hf-product-reasoning-rollouts-v1",
                "status": "complete",
                "model_root": "model",
                "model_revision": "revision",
                "adapter_checkpoint": "checkpoint",
                "samples": 2,
                "max_new_tokens": 16,
                "seed": 31,
                "count": len(bank_rows),
                "data": str(bank),
                "data_sha256": bank_sha,
                "candidates_output": str(candidates),
                "candidates_sha256": candidate_sha,
                "counters": {"correct_candidates": len(bank_rows)},
            }
        )
    )
    return report


def test_merge_covers_canonical_bank_in_canonical_order(tmp_path: Path) -> None:
    rows = [{"text": f"problem {index}"} for index in range(4)]
    bank = tmp_path / "bank.jsonl"
    _write_jsonl(bank, rows)
    reports = [_shard(tmp_path, 0, rows[::2]), _shard(tmp_path, 1, rows[1::2])]
    merged, report = merge_shard_reports(bank, reports, task="mbpp")
    assert report["identities"] == 4
    assert report["samples_per_identity"] == 2
    assert [row["identity_sha256"] for row in merged[::2]] == [
        _identity("mbpp", row) for row in rows
    ]


def test_merge_rejects_repeated_shard_identity(tmp_path: Path) -> None:
    rows = [{"text": "problem"}]
    bank = tmp_path / "bank.jsonl"
    _write_jsonl(bank, rows)
    first = _shard(tmp_path, 0, rows)
    second = _shard(tmp_path, 1, rows)
    with pytest.raises(ShardedCandidateMergeError, match="repeats across shards"):
        merge_shard_reports(bank, [first, second], task="mbpp")
