"""Adversarial integration tests for label-free PCF1 confirmation merging."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from hf_pcf1_evaluate import (
    PCF1EvaluationError,
    load_rows,
    model_visible_runtime_fields,
    nonpadding_prompt_tokens,
    self_refinement_prompt,
)
from hf_pcf1_apply_commit import PCF1ApplyError, load_pairs as load_application_pairs
from merge_pcf1_evaluation_shards import (
    PCF1MergeError,
    merge,
    validate_shard_setup_receipts,
)
from pcf1_code_sandbox import (
    BWRAP_SHA256,
    CANDIDATE_POLICY_SHA256,
    SANDBOX_CONFIG_SHA256,
    mbpp_allocation_setup_receipts_sha256,
)

TOTAL = 1289
SEALED_ZERO = {"holdout": 0, "product": 0, "public": 0}


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _setup_receipt(source: str, probe_sha256: str = "9" * 64) -> dict[str, Any]:
    receipt = {
        "schema": "shohin-pcf1-mbpp-setup-qualification-v1",
        "status": "pass",
        "setup_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "candidate_policy_sha256": CANDIDATE_POLICY_SHA256,
        "sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
        "allocation_probe_sha256": probe_sha256,
        "termination_classification": "trusted_tests_completed",
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return receipt


def _source_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tasks = ("math500", "bbh_logic", "mbpp")
    for index in range(TOTAL):
        identity = hashlib.sha256(f"pcf1-merge-{index}".encode()).hexdigest()
        draft = f"draft {index}"
        rows.append(
            {
                "schema": "shohin-pcf1-eval-v1",
                "identity_sha256": identity,
                "split": "confirmation",
                "task": tasks[index % 3],
                "question": f"question {index}\n{draft}",
                "source_prompt": f"question {index}",
                "runtime_fields": ["question", "source_prompt"],
                "internal_draft_visible": True,
                "external_candidate_text_visible": False,
                "internal_draft": {
                    "identity_sha256": identity,
                    "completion": draft,
                },
                "candidates": [],
            }
        )
    return rows


def _fixture(root: Path) -> tuple[argparse.Namespace, dict[str, Any]]:
    shard_root = root / "shards"
    shard_root.mkdir(parents=True)
    sources = _source_rows()
    data = _write_jsonl(root / "data.jsonl", sources)
    data_report = _write_json(
        root / "data_report.json",
        {
            "schema": "shohin-pcf1-data-report-v1",
            "status": "complete",
            "outputs": {
                "confirmation": {
                    "path": str(data),
                    "sha256": _sha(data),
                    "rows": TOTAL,
                }
            },
            "sealed_access": SEALED_ZERO,
        },
    )
    candidates = [
        {
            "schema": "shohin-pcf1-candidate-v1",
            "arm": "revision",
            "identity_sha256": row["identity_sha256"],
            "task": row["task"],
            "completion": "" if index == 0 else f"completion {index}",
            "generated_tokens": 0 if index == 0 else 4,
            "max_token_exhausted": False,
        }
        for index, row in enumerate(sources)
    ]
    candidate_path = _write_jsonl(shard_root / "shard_candidates.jsonl", candidates)
    adapter_metadata = {
        "arm": "baseline",
        "model_revision": "81eaece1948f3875421d9a45bc55487d10e2d894",
        "model_loader": "multimodal",
        "lora_layers": 4,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_scope": "token_mixer",
        "trainable_parameters": 1234,
        "trainable_parameter_name_sha256": "b" * 64,
    }
    adapter_metadata_sha256 = hashlib.sha256(
        json.dumps(adapter_metadata, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    shard_report = {
        "schema": "shohin-pcf1-evaluation-v1",
        "status": "complete",
        "arm": "revision",
        "split": "confirmation",
        "metrics": None,
        "assessment_mode": "confirmation_deferred",
        "assessor_board_access_count": 0,
        "runtime_fields": ["question"],
        "model_root": "/models/ministral",
        "model_revision": "81eaece1948f3875421d9a45bc55487d10e2d894",
        "model_loader": "multimodal",
        "adapter_checkpoint_sha256": "a" * 64,
        "adapter_metadata": adapter_metadata,
        "adapter_metadata_sha256": adapter_metadata_sha256,
        "trainable_parameters": 1234,
        "trainable_parameter_name_sha256": "b" * 64,
        "lora_layer_indices": [30, 31, 32, 33],
        "code_sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
        "code_sandbox_binary_sha256": BWRAP_SHA256,
        "code_sandbox_probe_sha256": None,
        "code_sandbox_probe_result_sha256": None,
        "sandbox_receipt_sha256": None,
        "code_sandbox_status": "not_applicable_no_code_scoring",
        "code_sandbox_probe_passed": None,
        "mbpp_allocation_setup_status": "not_applicable_no_code_scoring",
        "mbpp_allocation_setup_receipts": [],
        "mbpp_allocation_setup_receipt_count": 0,
        "mbpp_allocation_setup_receipts_sha256": None,
        "environment_verified": True,
        "environment_receipt_sha256": "c" * 64,
        "environment_tree_sha256": "d" * 64,
        "data_sha256": _sha(data),
        "data_report_sha256": _sha(data_report),
        "generation_mode": "greedy",
        "max_new_tokens": 768,
        "seed": 2026080816,
        "batch_size": 2,
        "shard_index": 0,
        "shard_count": 1,
        "row_start": 0,
        "row_end": TOTAL,
        "full_row_count": TOTAL,
        "candidates_output": str(candidate_path),
        "candidates_sha256": _sha(candidate_path),
        "counters": {
            "rows": TOTAL,
            "prompt_tokens": 10_000,
            "generated_tokens": 4 * (TOTAL - 1),
            "max_token_exhausted": 0,
            "empty_completions": 1,
            "sandbox_executions": 0,
            "capability_policy_rejections": 0,
        },
        "elapsed_seconds": 1.0,
        "peak_gpu_memory_bytes": 1,
        "sealed_access": SEALED_ZERO,
    }
    shard_report_path = _write_json(shard_root / "shard_report.json", shard_report)
    args = argparse.Namespace(
        arm="revision",
        split="confirmation",
        data=data,
        data_report=data_report,
        shard_reports=[shard_report_path],
        shard_candidates=[candidate_path],
        shard_root=shard_root,
        shard_sandbox_probes=[],
        candidates_output=root / "merged_candidates.jsonl",
        report=root / "merged_report.json",
    )
    return args, shard_report


def test_confirmation_merge_remains_label_free_and_carries_malformed_counter(
    tmp_path: Path,
) -> None:
    args, _ = _fixture(tmp_path)
    report = merge(args)
    assert report["metrics"] is None
    assert report["assessment_mode"] == "confirmation_deferred"
    assert report["assessor_board_access_count"] == 0
    assert report["counters"]["empty_completions"] == 1
    assert report["counters"]["prompt_tokens"] == 10_000
    assert report["aggregate_prompt_tokens"] == 10_000
    assert report["trainable_parameters"] == 1234
    assert report["lora_layer_indices"] == [30, 31, 32, 33]
    first = json.loads(args.candidates_output.read_text().splitlines()[0])
    assert set(first) == {
        "schema",
        "arm",
        "identity_sha256",
        "task",
        "completion",
        "generated_tokens",
        "max_token_exhausted",
    }


def test_confirmation_runtime_rejects_embedded_assessor(tmp_path: Path) -> None:
    rows = _source_rows()
    rows[0]["assessor"] = {"task": rows[0]["task"]}
    path = _write_jsonl(tmp_path / "data.jsonl", rows)
    with pytest.raises(PCF1EvaluationError, match="exposes supervision"):
        load_rows(path, "confirmation")


def test_self_refinement_prompt_and_nonpadding_accounting_are_frozen() -> None:
    row = {
        "task": "math500",
        "source_prompt": "What is 2 + 2?",
        "internal_draft": {"completion": "It might be 5."},
    }
    assert self_refinement_prompt(row) == (
        "Review the attempted solution below for mistakes, then solve the original "
        "problem correctly. Do not only critique the attempt.\n\n"
        "Original problem:\nWhat is 2 + 2?\n\nAttempt:\nIt might be 5.\n\n"
        "Follow the original problem's requested output format."
    )
    mbpp = {**row, "task": "mbpp"}
    assert self_refinement_prompt(mbpp) == self_refinement_prompt(row)

    class NoTaskAccess(dict[str, Any]):
        def __getitem__(self, key: str) -> Any:
            if key == "task":
                raise AssertionError("task router access is forbidden")
            return super().__getitem__(key)

    assert self_refinement_prompt(NoTaskAccess(row)) == self_refinement_prompt(row)

    class Tokenizer:
        def __call__(self, rendered: list[str], **_: Any) -> dict[str, Any]:
            assert rendered == ["left", "right"]
            return {"attention_mask": [[0, 1, 1], [1, 1, 1]]}

    assert nonpadding_prompt_tokens(Tokenizer(), ["left", "right"]) == 5
    assert model_visible_runtime_fields("revision") == ["question"]
    assert model_visible_runtime_fields("unchanged") == ["question"]
    assert model_visible_runtime_fields("self_refinement") == [
        "source_prompt",
        "internal_draft.completion",
    ]


def test_merge_rejects_scored_confirmation_candidate(tmp_path: Path) -> None:
    args, report = _fixture(tmp_path)
    rows = [
        json.loads(line) for line in args.shard_candidates[0].read_text().splitlines()
    ]
    rows[0]["correct"] = False
    _write_jsonl(args.shard_candidates[0], rows)
    report["candidates_sha256"] = _sha(args.shard_candidates[0])
    _write_json(args.shard_reports[0], report)
    with pytest.raises(PCF1MergeError, match="exposes assessment"):
        merge(args)


def test_merge_rejects_missing_or_report_only_candidate_shard(tmp_path: Path) -> None:
    args, report = _fixture(tmp_path)
    report["shard_count"] = 2
    _write_json(args.shard_reports[0], report)
    with pytest.raises(PCF1MergeError, match="custody binding"):
        merge(args)


def test_merge_rejects_shard_inputs_outside_explicit_root(tmp_path: Path) -> None:
    args, _ = _fixture(tmp_path)
    narrow_root = tmp_path / "narrow_shards"
    narrow_root.mkdir()
    args.shard_root = narrow_root

    with pytest.raises(PCF1MergeError, match="escapes the shard root"):
        merge(args)

    args, report = _fixture(tmp_path / "second")
    report["candidates_output"] = str(tmp_path / "holdout" / "hidden.jsonl")
    _write_json(args.shard_reports[0], report)
    with pytest.raises(PCF1MergeError, match="hash differs"):
        merge(args)


def test_merge_rejects_trainable_or_prompt_accounting_tamper(tmp_path: Path) -> None:
    args, report = _fixture(tmp_path)
    report["trainable_parameters"] += 1
    _write_json(args.shard_reports[0], report)
    with pytest.raises(PCF1MergeError, match="adapter trainables"):
        merge(args)

    args, report = _fixture(tmp_path / "prompt")
    report["counters"]["prompt_tokens"] = 0
    _write_json(args.shard_reports[0], report)
    with pytest.raises(PCF1MergeError, match="counters"):
        merge(args)


def test_merge_enforces_split_specific_explicit_sandbox_probe_boundary(
    tmp_path: Path,
) -> None:
    args, _ = _fixture(tmp_path)
    args.shard_sandbox_probes = [
        _write_json(tmp_path / "forbidden_confirmation_probe.json", {})
    ]
    with pytest.raises(PCF1MergeError, match="sandbox-probe arity"):
        merge(args)

    args, _ = _fixture(tmp_path / "calibration")
    args.split = "calibration"
    with pytest.raises(PCF1MergeError, match="sandbox-probe arity"):
        merge(args)


@pytest.mark.parametrize("mutation", ("tamper", "drop", "reorder"))
def test_calibration_setup_receipts_reject_tamper_drop_or_reorder(
    mutation: str,
) -> None:
    rows = [
        {"task": "mbpp", "assessor": {"test_setup_code": "alpha = 1\n"}},
        {"task": "math500", "assessor": {}},
        {"task": "mbpp", "assessor": {"test_setup_code": "beta = 2\n"}},
    ]
    receipts = [_setup_receipt("alpha = 1\n"), _setup_receipt("beta = 2\n")]
    report = {
        "mbpp_allocation_setup_status": "passed",
        "mbpp_allocation_setup_receipts": receipts,
        "mbpp_allocation_setup_receipt_count": len(receipts),
        "mbpp_allocation_setup_receipts_sha256": (
            mbpp_allocation_setup_receipts_sha256(receipts)
        ),
    }
    if mutation == "tamper":
        report["mbpp_allocation_setup_receipts"][0]["setup_source_sha256"] = "0" * 64
    elif mutation == "drop":
        report["mbpp_allocation_setup_receipts"].pop()
        report["mbpp_allocation_setup_receipt_count"] -= 1
        report["mbpp_allocation_setup_receipts_sha256"] = (
            mbpp_allocation_setup_receipts_sha256(
                report["mbpp_allocation_setup_receipts"]
            )
        )
    else:
        report["mbpp_allocation_setup_receipts"].reverse()
        report["mbpp_allocation_setup_receipts_sha256"] = (
            mbpp_allocation_setup_receipts_sha256(
                report["mbpp_allocation_setup_receipts"]
            )
        )

    with pytest.raises(PCF1MergeError, match="setup"):
        validate_shard_setup_receipts(
            report, rows, "calibration", {"probe_sha256": "9" * 64}
        )


def test_commit_application_rejects_hidden_top_level_label(tmp_path: Path) -> None:
    pair = {
        "schema": "shohin-pcf1-confirmation-pair-v1",
        "identity_sha256": "a" * 64,
        "split": "confirmation",
        "task": "math500",
        "question": "question",
        "candidates": [
            {"lineage": "revision", "completion": "revision"},
            {"lineage": "unchanged", "completion": "unchanged"},
        ],
        "assessor": {"answer": "secret"},
    }
    path = _write_jsonl(tmp_path / "pairs.jsonl", [pair])
    with pytest.raises(PCF1ApplyError, match="label-free pair"):
        load_application_pairs(path)


@pytest.mark.parametrize("payload", ['{"schema":', "[]\n"])
def test_commit_application_reports_malformed_pair_evidence(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(PCF1ApplyError, match="malformed|not an object"):
        load_application_pairs(path)
