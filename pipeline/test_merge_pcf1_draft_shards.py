"""Focused custody tests for source-only PCF1 draft merging."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from merge_pcf1_draft_shards import PCF1DraftMergeError, merge
from hf_pcf1_generate_drafts import (
    PCF1DraftError,
    reject_protected_path,
    shard_bounds,
)

MODEL_REVISION = "81eaece1948f3875421d9a45bc55487d10e2d894"
ADAPTER_SHA256 = "a" * 64
ENVIRONMENT_SHA256 = "e" * 64
TRAIN_ROWS = 5824
DEVELOPMENT_ROWS = 1289
TOTAL_ROWS = TRAIN_ROWS + DEVELOPMENT_ROWS
SHARDS = 16


def _write_lines(path: Path, rows: list[dict]) -> str:
    encoded = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode() for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _fixture(root: Path) -> tuple[argparse.Namespace, list[Path], list[dict]]:
    source_root = root / "safe" / "sources"
    source_rows: list[dict] = []
    for index in range(TOTAL_ROWS):
        split = "train" if index < TRAIN_ROWS else "development"
        identity = hashlib.sha256(f"source-{index}".encode()).hexdigest()
        source_rows.append(
            {
                "schema": f"shohin-pcf1-{split}-source-v1",
                "identity_sha256": identity,
                "split": split,
                "task": ("math500", "bbh_logic", "mbpp")[index % 3],
                "source_prompt": f"prompt {index}",
                "runtime_fields": ["source_prompt"],
            }
        )
    train = source_rows[:TRAIN_ROWS]
    development = source_rows[TRAIN_ROWS:]
    train_sha = _write_lines(source_root / "train_sources.jsonl", train)
    development_sha = _write_lines(
        source_root / "development_sources.jsonl", development
    )
    source_report = {
        "schema": "shohin-pcf1-data-freeze-report-v1",
        "status": "complete",
        "source_disjoint": True,
        "sealed_content_materialized": False,
        "counts": {"train": TRAIN_ROWS, "development": DEVELOPMENT_ROWS},
        "outputs": {
            "train_sources.jsonl": {"sha256": train_sha, "rows": TRAIN_ROWS},
            "development_sources.jsonl": {
                "sha256": development_sha,
                "rows": DEVELOPMENT_ROWS,
            },
        },
    }
    source_report_sha = _write_json(source_root / "report.json", source_report)
    report_paths: list[Path] = []
    draft_rows: list[dict] = []
    for source in source_rows:
        exhausted = False
        draft_rows.append(
            {
                "schema": "shohin-pcf1-model-draft-v1",
                "identity_sha256": source["identity_sha256"],
                "split": source["split"],
                "task": source["task"],
                "completion": f"completion {source['identity_sha256'][:4]}",
                "generated_tokens": 8,
                "max_token_exhausted": exhausted,
                "prompt_sha256": hashlib.sha256(
                    source["source_prompt"].encode()
                ).hexdigest(),
                "adapter_checkpoint_sha256": ADAPTER_SHA256,
                "model_revision": MODEL_REVISION,
                "finish_reason": "stop",
                "wall_seconds": 1.0,
            }
        )
    for shard in range(SHARDS):
        start, end = shard_bounds(TOTAL_ROWS, shard, SHARDS, 2)
        candidates = root / "safe" / "shards" / str(shard) / "candidates.jsonl"
        candidates_sha = _write_lines(candidates, draft_rows[start:end])
        report = {
            "schema": "shohin-pcf1-draft-shard-v1",
            "status": "complete",
            "model_root": "/safe/model",
            "model_revision": MODEL_REVISION,
            "model_loader": "multimodal",
            "adapter_checkpoint_sha256": ADAPTER_SHA256,
            "environment_receipt_sha256": ENVIRONMENT_SHA256,
            "environment_tree_sha256": "f" * 64,
            "source_root": str(source_root.resolve()),
            "source_report_sha256": source_report_sha,
            "source_counts": {
                "train": TRAIN_ROWS,
                "development": DEVELOPMENT_ROWS,
            },
            "runtime_fields": ["source_prompt"],
            "supervisor_fields_visible_to_model": False,
            "generation_mode": "greedy",
            "thinking_enabled": False,
            "max_new_tokens": 768,
            "seed": 2026080818,
            "batch_size": 2,
            "shard_index": shard,
            "shard_count": SHARDS,
            "row_start": start,
            "row_end": end,
            "full_row_count": TOTAL_ROWS,
            "candidates_output": str(candidates.resolve()),
            "candidates_sha256": candidates_sha,
            "rows": end - start,
            "prompt_tokens": 10,
            "generated_tokens": 8 * (end - start),
            "max_token_exhausted": 0,
            "elapsed_seconds": 2.0,
            "peak_gpu_memory_bytes": 100,
            "sealed_access": {"holdout": 0, "product": 0, "public": 0},
        }
        report_path = candidates.parent / "report.json"
        _write_json(report_path, report)
        report_paths.append(report_path)
    args = argparse.Namespace(
        source_root=source_root,
        shard_reports=report_paths,
        shard_candidates=[
            Path(json.loads(path.read_text())["candidates_output"])
            for path in report_paths
        ],
        output=root / "safe" / "merged" / "drafts.jsonl",
        report=root / "safe" / "merged" / "report.json",
    )
    return args, report_paths, draft_rows


def _refresh_report(report_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidates = Path(report["candidates_output"])
    report["candidates_sha256"] = hashlib.sha256(candidates.read_bytes()).hexdigest()
    _write_json(report_path, report)


def test_merge_binds_exact_source_order_and_fields(tmp_path: Path) -> None:
    args, _, rows = _fixture(tmp_path)
    report = merge(args)
    assert report["rows"] == TOTAL_ROWS
    assert [
        json.loads(line)["identity_sha256"]
        for line in args.output.read_text(encoding="utf-8").splitlines()
    ] == [row["identity_sha256"] for row in rows]


def test_rejects_identity_substitution_even_with_rehashed_shard(tmp_path: Path) -> None:
    args, reports, _ = _fixture(tmp_path)
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    candidates = Path(report["candidates_output"])
    rows = [json.loads(line) for line in candidates.read_text().splitlines()]
    rows[0]["identity_sha256"] = "f" * 64
    _write_lines(candidates, rows)
    _refresh_report(reports[0])
    with pytest.raises(PCF1DraftMergeError, match="row binding"):
        merge(args)


def test_rejects_reordered_rows_even_with_rehashed_shard(tmp_path: Path) -> None:
    args, reports, _ = _fixture(tmp_path)
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    candidates = Path(report["candidates_output"])
    rows = [json.loads(line) for line in candidates.read_text().splitlines()]
    _write_lines(candidates, list(reversed(rows)))
    _refresh_report(reports[0])
    with pytest.raises(PCF1DraftMergeError, match="row binding"):
        merge(args)


def test_rejects_missing_shard(tmp_path: Path) -> None:
    args, reports, _ = _fixture(tmp_path)
    args.shard_reports = reports[:1]
    with pytest.raises(PCF1DraftMergeError, match="shard count"):
        merge(args)


def test_rejects_explicit_candidate_path_substitution(tmp_path: Path) -> None:
    args, _, _ = _fixture(tmp_path)
    substitute = tmp_path / "safe" / "substitute.jsonl"
    substitute.write_bytes(args.shard_candidates[0].read_bytes())
    args.shard_candidates[0] = substitute
    with pytest.raises(PCF1DraftMergeError, match="explicit shard candidate path"):
        merge(args)


@pytest.mark.parametrize("term", ("holdout", "product", "public"))
def test_firewall_rejects_protected_paths(tmp_path: Path, term: str) -> None:
    with pytest.raises(PCF1DraftError, match="protected path"):
        reject_protected_path(tmp_path / term / "drafts.jsonl")
