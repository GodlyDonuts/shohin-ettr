from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import build_q36_mtr_data as data_module
import merge_q36_mtr_drafts as draft_merge_module
import merge_q36_mtr_evaluations as eval_merge_module
from hf_q36_mtr_evaluate import CANDIDATE_SCHEMA, REPORT_SCHEMA as EVAL_REPORT_SCHEMA
from hf_q36_mtr_generate_drafts import (
    REPORT_SCHEMA as DRAFT_SHARD_REPORT_SCHEMA,
    exact_model_owned_completion,
)
from q36_mtr_roles import MODEL_REVISION, TRAINABLE_PARAMETERS


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, payload: dict) -> Path:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _source(identity: str, split: str, task: str = "math500") -> dict:
    row = {
        "identity_sha256": identity,
        "split": split,
        "task": task,
        "source_prompt": f"problem-{identity[:4]}",
        "runtime_fields": ["source_prompt"],
    }
    if split == "train":
        row.update(
            {
                "assessor": {
                    "schema": data_module.ASSESSOR_SCHEMA,
                    "identity_sha256": identity,
                    "task": task,
                },
                "response": "solution",
                "target_kind": "boxed",
                "outcome_class": "base_only",
            }
        )
    return row


def _draft(source: dict, checkpoint: str = "a" * 64) -> dict:
    return {
        "schema": data_module.DRAFT_SCHEMA,
        "identity_sha256": source["identity_sha256"],
        "split": source["split"],
        "task": source["task"],
        "prompt_sha256": hashlib.sha256(source["source_prompt"].encode()).hexdigest(),
        "owner_checkpoint_sha256": checkpoint,
        "model_revision": MODEL_REVISION,
        "completion": "draft reasoning",
        "generated_tokens": 2,
        "max_token_exhausted": False,
        "finish_reason": "stop",
        "wall_seconds": 0.1,
    }


def test_materializer_emits_matched_revision_and_label_free_development(
    tmp_path: Path, monkeypatch
) -> None:
    train = {"1" * 64: _source("1" * 64, "train")}
    development = {"2" * 64: _source("2" * 64, "development")}
    source_root = tmp_path / "sources"
    source_root.mkdir()
    _json(
        source_root / "report.json",
        {
            "schema": data_module.FREEZE_REPORT_SCHEMA,
            "status": "complete",
            "counts": {"train": 1, "development": 1, "holdout": 1_279},
            "source_disjoint": True,
            "sealed_content_materialized": False,
        },
    )
    monkeypatch.setattr(data_module, "TRAIN_IDENTITIES", 1)
    monkeypatch.setattr(data_module, "DEVELOPMENT_IDENTITIES", 1)
    monkeypatch.setattr(data_module, "DRAFT_IDENTITIES", 2)
    monkeypatch.setattr(data_module, "REVISION_PRESENTATIONS", 4)
    monkeypatch.setattr(
        data_module,
        "_load_source_view",
        lambda _path, _schema, split: train if split == "train" else development,
    )
    train_draft = _draft(train["1" * 64])
    train_draft["completion"] = " \n draft reasoning\t"
    development_draft = _draft(development["2" * 64])
    development_draft["completion"] = "\n draft reasoning \n"
    drafts = _jsonl(tmp_path / "drafts.jsonl", [train_draft, development_draft])
    draft_report = _json(
        tmp_path / "draft_report.json",
        {
            "schema": data_module.DRAFT_REPORT_SCHEMA,
            "status": "complete",
            "output": str(drafts.resolve()),
            "output_sha256": _sha(drafts),
            "rows": 2,
            "model_revision": MODEL_REVISION,
            "owner_checkpoint_sha256": "a" * 64,
            "exact_identity_coverage": True,
            "duplicate_identities": 0,
            "sealed_access": {"holdout": 0, "product": 0, "public": 0},
        },
    )
    assessor = _json(
        tmp_path / "assessor_receipt.json",
        {
            "schema": data_module.CONFIRMATION_ASSESSOR_RECEIPT_SCHEMA,
            "status": "complete",
            "rows": 1,
            "semantic_access": "final_score_only",
            "board_sha256": "b" * 64,
        },
    )
    output = tmp_path / "materialized"
    report = data_module.materialize(
        SimpleNamespace(
            source_root=source_root,
            drafts=drafts,
            draft_report=draft_report,
            assessor_receipt=assessor,
            output=output,
        )
    )
    assert report["counts"] == {
        "train_unique_identities": 1,
        "revision_train_presentations": 4,
        "calibration_rows": 1,
        "development_rows": 1,
    }
    assert report["draft_byte_custody"]["canonicalized_drafts"] == 2
    assert report["draft_byte_custody"]["raw_decode_preserved_in_merged_drafts"] is True
    development_row = json.loads((output / "development_eval.jsonl").read_text())
    assert development_row["split"] == "development"
    assert development_row["runtime_fields"] == ["question", "source_prompt"]
    assert not ({"assessor", "answer", "response", "gold"} & set(development_row))
    assert development_row["internal_draft"]["completion"] == "draft reasoning"
    assert (
        development_row["model_owned_draft_sha256"]
        == hashlib.sha256(b"draft reasoning").hexdigest()
    )
    assert (
        development_row["raw_model_owned_draft_sha256"]
        == hashlib.sha256(b"\n draft reasoning \n").hexdigest()
    )
    assert (
        development_row["draft_canonicalization"] == "unicode_outer_whitespace_strip_v1"
    )
    assert "Internal draft:\ndraft reasoning\n\n" in development_row["question"]
    revision_rows = (output / "revision_train.jsonl").read_text().splitlines()
    assert len(revision_rows) == 4


def test_owner_decode_bytes_are_preserved_until_materialization() -> None:
    decoded = " \nmodel-owned draft\t\n"
    assert exact_model_owned_completion(decoded) == decoded


def test_draft_merge_rejects_duplicate_or_missing_shard_ranges(
    tmp_path: Path, monkeypatch
) -> None:
    sources = [_source("1" * 64, "train"), _source("2" * 64, "development")]
    monkeypatch.setattr(draft_merge_module, "DRAFT_IDENTITIES", 2)
    monkeypatch.setattr(draft_merge_module, "DRAFT_SHARDS", 2)
    monkeypatch.setattr(
        draft_merge_module,
        "load_sources",
        lambda *_args: (
            sources,
            {"identity_receipts": {"train": {}, "development": {}}},
        ),
    )
    train_source = _jsonl(tmp_path / "train.jsonl", [sources[0]])
    development_source = _jsonl(tmp_path / "development.jsonl", [sources[1]])
    freeze = _json(tmp_path / "freeze.json", {})
    reports = []
    candidates = []
    for index, source in enumerate(sources):
        candidate = _jsonl(tmp_path / f"candidate-{index}.jsonl", [_draft(source)])
        report = _json(
            tmp_path / f"report-{index}.json",
            {
                "schema": DRAFT_SHARD_REPORT_SCHEMA,
                "status": "complete",
                "capability_scored": False,
                "sealed_access": {"holdout": 0, "product": 0, "public": 0},
                "model_revision": MODEL_REVISION,
                "model_loader": "causal",
                "owner_checkpoint_sha256": "a" * 64,
                "owner_update": 256,
                "owner_role": "owner",
                "freeze_report_sha256": "b" * 64,
                "freeze_identity_receipts": {},
                "train_source_sha256": "c" * 64,
                "development_source_sha256": "d" * 64,
                "generation_mode": "greedy",
                "generation_sequence_contract": (
                    "inputs_embeds_generated_tokens_only_v1"
                ),
                "rendered_chat_tokenization": "add_special_tokens_false",
                "max_new_tokens": 768,
                "seed": 2026080818,
                "shard_index": index,
                "shard_count": 2,
                "full_rows": 2,
                "row_start": index,
                "row_end": index + 1,
                "rows": 1,
                "generated_tokens": 2,
                "prompt_tokens": 3,
                "max_token_exhausted": 0,
                "elapsed_seconds": 0.1,
                "peak_gpu_memory_bytes": 1,
                "output": str(candidate.resolve()),
                "output_sha256": _sha(candidate),
            },
        )
        reports.append(report)
        candidates.append(candidate)
    args = SimpleNamespace(
        train_source=train_source,
        development_source=development_source,
        freeze_report=freeze,
        shard_reports=reports,
        shard_candidates=candidates,
        output=tmp_path / "merged.jsonl",
        report=tmp_path / "merged.json",
    )
    result = draft_merge_module.merge(args)
    assert result["rows"] == 2
    args.output = tmp_path / "bad.jsonl"
    args.report = tmp_path / "bad.json"
    args.shard_reports = [reports[0], reports[0]]
    args.shard_candidates = [candidates[0], candidates[0]]
    with pytest.raises(draft_merge_module.Q36MTRDraftMergeError):
        draft_merge_module.merge(args)


def test_development_merge_remains_label_free(tmp_path: Path, monkeypatch) -> None:
    sources = [
        {
            "identity_sha256": "1" * 64,
            "task": "math500",
        },
        {
            "identity_sha256": "2" * 64,
            "task": "bbh_logic",
        },
    ]
    monkeypatch.setitem(eval_merge_module.EXPECTED_SHARDS, "development", 2)
    monkeypatch.setitem(eval_merge_module.EXPECTED_FULL_ROWS, "development", 2)
    monkeypatch.setattr(eval_merge_module, "load_rows", lambda *_args: sources)
    data = _jsonl(tmp_path / "data.jsonl", sources)
    data_report = _json(
        tmp_path / "data_report.json",
        {
            "schema": eval_merge_module.DATA_REPORT_SCHEMA,
            "status": "complete",
            "outputs": {
                "development": {
                    "path": str(data.resolve()),
                    "sha256": _sha(data),
                }
            },
            "sealed_access": {"holdout": 0, "product": 0, "public": 0},
        },
    )
    reports = []
    candidate_paths = []
    for index, source in enumerate(sources):
        candidate = {
            "schema": CANDIDATE_SCHEMA,
            "arm": "revision",
            "identity_sha256": source["identity_sha256"],
            "task": source["task"],
            "completion": "answer",
            "generated_tokens": 1,
            "max_token_exhausted": False,
        }
        candidate_path = _jsonl(tmp_path / f"eval-candidate-{index}.jsonl", [candidate])
        report = _json(
            tmp_path / f"eval-report-{index}.json",
            {
                "schema": EVAL_REPORT_SCHEMA,
                "status": "complete",
                "arm": "revision",
                "split": "development",
                "model_revision": MODEL_REVISION,
                "model_loader": "causal",
                "adapter_checkpoint_sha256": "a" * 64,
                "adapter_metadata_sha256": "b" * 64,
                "trainable_parameters": TRAINABLE_PARAMETERS,
                "trainable_parameter_name_sha256": "c" * 64,
                "controlled_layer_indices": list(range(24, 40)),
                "role": "aligned",
                "data_sha256": _sha(data),
                "data_report_sha256": _sha(data_report),
                "runtime_fields": ["question"],
                "assessor_fields_visible_to_model": False,
                "assessment_mode": "development_deferred",
                "assessor_board_access_count": 0,
                "generation_mode": "greedy",
                "generation_sequence_contract": (
                    "inputs_embeds_generated_tokens_only_v1"
                ),
                "rendered_chat_tokenization": "add_special_tokens_false",
                "max_new_tokens": 768,
                "seed": eval_merge_module.EVALUATION_SEED,
                "batch_size": 1,
                "shard_index": index,
                "shard_count": 2,
                "row_start": index,
                "row_end": index + 1,
                "full_row_count": 2,
                "candidates_output": str(candidate_path.resolve()),
                "candidates_sha256": _sha(candidate_path),
                "counters": {
                    "rows": 1,
                    "prompt_tokens": 2,
                    "generated_tokens": 1,
                    "max_token_exhausted": 0,
                    "empty_completions": 0,
                },
                "elapsed_seconds": 0.1,
                "peak_gpu_memory_bytes": 1,
                "code_sandbox_config_sha256": "d" * 64,
                "code_sandbox_binary_sha256": "e" * 64,
                "sandbox_status": "not_applicable_no_scoring",
                "sandbox_receipt_sha256": None,
                "sandbox_probe_sha256": None,
                "mbpp_setup_receipts": [],
                "mbpp_setup_receipts_sha256": None,
                "environment_receipt_sha256": "f" * 64,
                "environment_tree_sha256": "0" * 64,
                "sealed_access": {"holdout": 0, "product": 0, "public": 0},
            },
        )
        reports.append(report)
        candidate_paths.append(candidate_path)
    result = eval_merge_module.merge(
        SimpleNamespace(
            arm="revision",
            split="development",
            data=data,
            data_report=data_report,
            shard_reports=reports,
            shard_candidates=candidate_paths,
            shard_sandbox_probes=[],
            output=tmp_path / "merged-eval.jsonl",
            report=tmp_path / "merged-eval.json",
        )
    )
    assert result["rows"] == 2
    assert result["metrics"] is None
    assert result["assessor_board_access_count"] == 0
    assert result["exact_identity_coverage"] is True
