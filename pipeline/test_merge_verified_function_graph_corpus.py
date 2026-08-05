from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from build_verified_function_graph_corpus import SCHEMA, generate_rows
from merge_verified_function_graph_corpus import FunctionGraphMergeError, merge_shards


def _write_shard(tmp_path: Path, index: int, rows: list[dict]) -> tuple[Path, Path]:
    shard = tmp_path / f"shard{index}.jsonl"
    report = tmp_path / f"shard{index}.report.json"
    payload = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode() for row in rows
    )
    shard.write_bytes(payload)
    report.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "complete",
                "rows": len(rows),
                "seed": 31,
                "shard_index": index,
                "shard_count": 2,
                "ngram_width": 13,
                "output": str(shard.resolve()),
                "output_sha256": hashlib.sha256(payload).hexdigest(),
                "evaluation_sources": [],
                "counters": {"generated": len(rows), "kept": len(rows)},
            }
        )
    )
    return shard, report


def test_merge_requires_complete_unique_shards(tmp_path: Path) -> None:
    first, _ = generate_rows(
        count=4,
        seed=31,
        shard_index=0,
        shard_count=2,
        blocked_grams=set(),
        ngram_width=13,
        timeout_seconds=2,
    )
    second, _ = generate_rows(
        count=4,
        seed=31,
        shard_index=1,
        shard_count=2,
        blocked_grams=set(),
        ngram_width=13,
        timeout_seconds=2,
    )
    paths = [_write_shard(tmp_path, 0, first), _write_shard(tmp_path, 1, second)]
    rows, report = merge_shards(
        [item[0] for item in paths],
        [item[1] for item in paths],
        expected_shards=2,
        expected_rows=8,
    )
    assert len(rows) == 8
    assert report["rows"] == 8
    assert sum(report["family_counts"].values()) == 8


def test_merge_rejects_hash_mismatch(tmp_path: Path) -> None:
    rows, _ = generate_rows(
        count=2,
        seed=31,
        shard_index=0,
        shard_count=1,
        blocked_grams=set(),
        ngram_width=13,
        timeout_seconds=2,
    )
    shard, report = _write_shard(tmp_path, 0, rows)
    metadata = json.loads(report.read_text())
    metadata["shard_count"] = 1
    metadata["output_sha256"] = "0" * 64
    report.write_text(json.dumps(metadata))
    with pytest.raises(FunctionGraphMergeError, match="hash differs"):
        merge_shards([shard], [report], expected_shards=1, expected_rows=2)


def test_merge_rejects_verification_hash_mismatch(tmp_path: Path) -> None:
    rows, _ = generate_rows(
        count=2,
        seed=31,
        shard_index=0,
        shard_count=1,
        blocked_grams=set(),
        ngram_width=13,
        timeout_seconds=2,
    )
    rows[0]["verification_sha256"] = "0" * 64
    shard, report = _write_shard(tmp_path, 0, rows)
    metadata = json.loads(report.read_text())
    metadata["shard_count"] = 1
    report.write_text(json.dumps(metadata))
    with pytest.raises(FunctionGraphMergeError, match="verification hash differs"):
        merge_shards([shard], [report], expected_shards=1, expected_rows=2)
