from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import build_q36_mtr_data as data_module
import merge_upward_moe_drafts as merge_module
from hf_product_reasoning_eval import GENERATED_ONLY_SEQUENCE_CONTRACT
from hf_upward_moe_generate_drafts import (
    REPORT_SCHEMA as SHARD_REPORT_SCHEMA,
    SCHEMA as DRAFT_SCHEMA,
    UpwardMoEDraftError,
    host_spec,
)


def _json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(identity: str, split: str) -> dict:
    return {
        "identity_sha256": identity,
        "split": split,
        "task": "math500",
        "source_prompt": f"problem-{identity[:4]}",
        "runtime_fields": ["source_prompt"],
    }


def _candidate(source: dict, host: str) -> dict:
    spec = host_spec(host)
    return {
        "schema": DRAFT_SCHEMA,
        "host": spec.host,
        "identity_sha256": source["identity_sha256"],
        "split": source["split"],
        "task": source["task"],
        "prompt_sha256": hashlib.sha256(source["source_prompt"].encode()).hexdigest(),
        "owner_checkpoint_sha256": "a" * 64,
        "owner_state_sha256": "b" * 64,
        "model_revision": spec.model_revision,
        "completion": "host-owned reasoning",
        "generated_tokens": 3,
        "max_token_exhausted": False,
        "finish_reason": "stop",
        "wall_seconds": 0.1,
    }


def _merge_fixture(tmp_path: Path, monkeypatch, host: str = "nemotron-super"):
    spec = host_spec(host)
    sources = [_source("1" * 64, "train"), _source("2" * 64, "development")]
    monkeypatch.setattr(merge_module, "DRAFT_IDENTITIES", 2)
    monkeypatch.setattr(merge_module, "DRAFT_SHARDS", 2)
    monkeypatch.setattr(
        merge_module,
        "load_sources",
        lambda *_args: (
            sources,
            {"identity_receipts": {"train": {}, "development": {}}},
        ),
    )
    train = _jsonl(tmp_path / "train.jsonl", [sources[0]])
    development = _jsonl(tmp_path / "development.jsonl", [sources[1]])
    freeze = _json(tmp_path / "freeze.json", {})
    reports = []
    candidates = []
    for index, source in enumerate(sources):
        candidate = _jsonl(
            tmp_path / f"candidate-{index}.jsonl", [_candidate(source, host)]
        )
        report = {
            "schema": SHARD_REPORT_SCHEMA,
            "status": "complete",
            "host": spec.host,
            "model_revision": spec.model_revision,
            "host_contract": spec.receipt(),
            "model_receipt": {"tree": "c" * 64},
            "owner_checkpoint_sha256": "a" * 64,
            "owner_state_sha256": "b" * 64,
            "owner_role": "owner",
            "owner_update": 256,
            "owner_restore_exact": True,
            "mechanics_report_sha256": "d" * 64,
            "freeze_report_sha256": "e" * 64,
            "freeze_identity_receipts": {},
            "train_source_sha256": "f" * 64,
            "development_source_sha256": "0" * 64,
            "generation_mode": "greedy",
            "generation_sequence_contract": GENERATED_ONLY_SEQUENCE_CONTRACT,
            "rendered_chat_tokenization": "add_special_tokens_false",
            "max_new_tokens": 768,
            "seed": 2026080818,
            "shard_index": index,
            "shard_count": 2,
            "full_rows": 2,
            "row_start": index,
            "row_end": index + 1,
            "rows": 1,
            "prompt_tokens": 4,
            "generated_tokens": 3,
            "max_token_exhausted": 0,
            "elapsed_seconds": 0.1,
            "peak_gpu_memory_bytes": {"0": 1, "1": 2},
            "capability_scored": False,
            "sealed_access": {"holdout": 0, "product": 0, "public": 0},
            "output": str(candidate.resolve()),
            "output_sha256": _sha(candidate),
        }
        reports.append(_json(tmp_path / f"report-{index}.json", report))
        candidates.append(candidate)
    return SimpleNamespace(
        host=host,
        train_source=train,
        development_source=development,
        freeze_report=freeze,
        shard_reports=reports,
        shard_candidates=candidates,
        output=tmp_path / "merged.jsonl",
        report=tmp_path / "merged.json",
    )


def test_upward_host_specs_are_large_moe_only() -> None:
    assert host_spec("nemotron-super").host == "Nemotron-Super-120B-A12B"
    assert host_spec("mixtral-8x22b").host == "Mixtral-8x22B-141B-A39B"
    assert host_spec("nemotron-ultra").host == "Nemotron-Ultra-550B-A55B"
    with pytest.raises(UpwardMoEDraftError):
        host_spec("small-moe")


def test_upward_merge_binds_exact_host_owned_draft_lineage(
    tmp_path: Path, monkeypatch
) -> None:
    args = _merge_fixture(tmp_path, monkeypatch)
    result = merge_module.merge(args)
    assert result["rows"] == 2
    assert result["host"] == "Nemotron-Super-120B-A12B"
    assert result["owner_checkpoint_sha256"] == "a" * 64
    assert result["exact_identity_coverage"] is True
    assert result["sealed_access"] == {"holdout": 0, "product": 0, "public": 0}


def test_upward_merge_rejects_cross_host_draft(tmp_path: Path, monkeypatch) -> None:
    args = _merge_fixture(tmp_path, monkeypatch)
    row = json.loads(args.shard_candidates[0].read_text())
    row["host"] = "mixtral-8x22b"
    _jsonl(args.shard_candidates[0], [row])
    report = json.loads(args.shard_reports[0].read_text())
    report["output_sha256"] = _sha(args.shard_candidates[0])
    _json(args.shard_reports[0], report)
    with pytest.raises(merge_module.UpwardMoEDraftMergeError):
        merge_module.merge(args)


def test_materializer_selects_upward_draft_contract() -> None:
    contract = data_module._draft_contract("mixtral-8x22b")
    assert contract["host"] == "Mixtral-8x22B-141B-A39B"
    assert contract["draft_schema"] == DRAFT_SCHEMA
    assert contract["report_schema"] == merge_module.SCHEMA
    assert contract["model_revision"] == host_spec("mixtral-8x22b").model_revision
