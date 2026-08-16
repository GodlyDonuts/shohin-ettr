"""Tests for the pure native-to-final PCF1 report boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import pytest

from pipeline.compare_pcf1 import compare
from pipeline.normalize_pcf1_reports import PCF1NormalizationError, normalize
from pcf1_code_sandbox import (
    CANDIDATE_POLICY_SHA256,
    SANDBOX_CONFIG_SHA256 as QUALIFIED_SANDBOX_CONFIG_SHA256,
    mbpp_allocation_setup_receipts_sha256,
)

RUN_ID = "pcf1-normalization-test"
IDENTITY_SHA256 = "1" * 64
DATA_SHA256 = "2" * 64
DATA_REPORT_SHA256 = "3" * 64
MODEL_REVISION = "81eaece1948f3875421d9a45bc55487d10e2d894"
RUNTIME_SHA256 = "5" * 64
ENVIRONMENT_RECEIPT_SHA256 = "a" * 64
ENVIRONMENT_TREE_SHA256 = "b" * 64
SANDBOX_CONFIG_SHA256 = QUALIFIED_SANDBOX_CONFIG_SHA256
SANDBOX_BINARY_SHA256 = "d" * 64
SANDBOX_RECEIPT_SHA256 = "e" * 64
SANDBOX_RESULT_SHA256 = "f" * 64
COMMIT_PAIRS_SHA256 = "6" * 64
COMMIT_TRAINING_REPORT_SHA256 = "8" * 64
MODEL_ROOT = "/immutable/models/ministral"
DOMAIN_TOTALS = {"math500": 623, "bbh_logic": 637, "mbpp": 29}
DOMAIN_CORRECT = {
    "revision": {"math500": 130, "bbh_logic": 312, "mbpp": 10},
    "unchanged": {"math500": 100, "bbh_logic": 280, "mbpp": 7},
    "self_refinement": {"math500": 120, "bbh_logic": 284, "mbpp": 9},
    "learned_commit": {"math500": 140, "bbh_logic": 315, "mbpp": 10},
}
ADAPTERS = {
    "revision": "a" * 64,
    "unchanged": "b" * 64,
    "self_refinement": "b" * 64,
}
COMMIT_CHECKPOINT = "c" * 64
COMMON_EVALUATION_SETTINGS = {
    "model_loader": "multimodal",
    "generation_mode": "greedy",
    "max_new_tokens": 768,
    "seed": 2026080816,
    "batch_size": 2,
    "shard_count": 4,
}


def _setup_receipt(setup_source: str = "") -> dict:
    receipt = {
        "schema": "shohin-pcf1-mbpp-setup-qualification-v1",
        "status": "pass",
        "setup_source_sha256": hashlib.sha256(setup_source.encode()).hexdigest(),
        "candidate_policy_sha256": CANDIDATE_POLICY_SHA256,
        "sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
        "allocation_probe_sha256": SANDBOX_RESULT_SHA256,
        "setup_qualification_mode": "compile_only_before_candidate",
        "termination_classification": "trusted_tests_completed",
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return receipt


CALIBRATION_SETUP_CUSTODY = {
    arm: {
        "sandbox_probe_sha256s": [SANDBOX_RECEIPT_SHA256] * 4,
        "setup_receipt_count": 4,
        "setup_receipt_shards_sha256": digest,
    }
    for arm, digest in (("revision", "1" * 64), ("unchanged", "2" * 64))
}


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluation(arm: str) -> dict:
    return {
        "schema": "shohin-pcf1-merged-evaluation-v1",
        "status": "complete",
        "arm": arm,
        "split": "confirmation",
        "model_root": MODEL_ROOT,
        "model_revision": MODEL_REVISION,
        **COMMON_EVALUATION_SETTINGS,
        "adapter_checkpoint_sha256": ADAPTERS[arm],
        "adapter_metadata_sha256": "4" * 64,
        "trainable_parameters": 1234,
        "trainable_parameter_name_sha256": "a" * 64,
        "lora_layer_indices": [30, 31, 32, 33],
        "environment_verified": True,
        "environment_receipt_sha256": ENVIRONMENT_RECEIPT_SHA256,
        "environment_tree_sha256": ENVIRONMENT_TREE_SHA256,
        "code_sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
        "code_sandbox_binary_sha256": SANDBOX_BINARY_SHA256,
        "code_sandbox_status": "not_applicable_no_code_scoring",
        "code_sandbox_probe_passed": None,
        "code_sandbox_probe_sha256": None,
        "code_sandbox_probe_result_sha256": None,
        "sandbox_receipt_sha256": None,
        "mbpp_allocation_setup_status": "not_applicable_no_code_scoring",
        "mbpp_allocation_setup_receipt_shards": [],
        "mbpp_allocation_setup_receipt_count": 0,
        "mbpp_allocation_setup_receipt_shards_sha256": None,
        "data_sha256": DATA_SHA256,
        "data_report_sha256": DATA_REPORT_SHA256,
        "full_row_count": 1289,
        "candidates_sha256": {
            "revision": "d" * 64,
            "unchanged": "e" * 64,
            "self_refinement": "f" * 64,
        }[arm],
        "metrics": None,
        "assessment_mode": "confirmation_deferred",
        "assessor_board_access_count": 0,
        "counters": {
            "rows": 1289,
            "prompt_tokens": 200_000,
            "generated_tokens": 100_000,
            "max_token_exhausted": 0,
            "empty_completions": 0,
            "capability_policy_rejections": 0,
        },
        "exact_identity_coverage": True,
        "aggregate_prompt_tokens": 200_000,
        "aggregate_wall_seconds": 10.0,
        "aggregate_gpu_seconds": 10.0,
        "maximum_peak_gpu_memory_bytes": 1_000,
        "runtime_fields": (
            ["source_prompt", "internal_draft.completion"]
            if arm == "self_refinement"
            else ["question"]
        ),
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }


def _commit_bucket(domain: str) -> dict[str, int | float]:
    total = 1289 if domain == "overall" else DOMAIN_TOTALS[domain]
    revision = (
        sum(DOMAIN_CORRECT["revision"].values())
        if domain == "overall"
        else DOMAIN_CORRECT["revision"][domain]
    )
    unchanged = (
        sum(DOMAIN_CORRECT["unchanged"].values())
        if domain == "overall"
        else DOMAIN_CORRECT["unchanged"][domain]
    )
    selected = (
        sum(DOMAIN_CORRECT["learned_commit"].values())
        if domain == "overall"
        else DOMAIN_CORRECT["learned_commit"][domain]
    )
    revision_retained = {
        "overall": 430,
        "math500": 120,
        "bbh_logic": 300,
        "mbpp": 10,
    }[domain]
    unchanged_retained = {
        "overall": 368,
        "math500": 95,
        "bbh_logic": 267,
        "mbpp": 6,
    }[domain]
    return {
        "total": total,
        "revision_correct": revision,
        "unchanged_correct": unchanged,
        "selected_correct": selected,
        "oracle_correct": selected,
        "unchanged_commits": 100,
        "order_consistent": total,
        "revision_correct_retained": revision_retained,
        "unchanged_correct_retained": unchanged_retained,
        "selected_accuracy": selected / total,
        "order_consistency": 1.0,
        "revision_correct_retention": revision_retained / revision,
        "unchanged_correct_retention": unchanged_retained / unchanged,
    }


def _commit() -> dict:
    arm_metrics = {
        arm: {
            "overall": {
                "generated_correct": sum(DOMAIN_CORRECT[arm].values()),
                "total": 1289,
            },
            **{
                domain: {
                    "generated_correct": score,
                    "total": DOMAIN_TOTALS[domain],
                }
                for domain, score in DOMAIN_CORRECT[arm].items()
            },
        }
        for arm in ("revision", "unchanged", "self_refinement")
    }
    setup_receipts = [_setup_receipt(), _setup_receipt("value = 1\n")]
    return {
        "schema": "shohin-pcf1-commit-result-v1",
        "status": "complete",
        "model_root": MODEL_ROOT,
        "model_revision": MODEL_REVISION,
        "model_loader": "multimodal",
        "adapter_checkpoint_sha256": ADAPTERS["unchanged"],
        "checkpoint_sha256": COMMIT_CHECKPOINT,
        "training_report_sha256": COMMIT_TRAINING_REPORT_SHA256,
        "selections_sha256": "7" * 64,
        "pairs_sha256": COMMIT_PAIRS_SHA256,
        "run_id": RUN_ID,
        "max_sequence_length": 3072,
        "protected_adapter_unchanged": True,
        "arm_metrics": arm_metrics,
        "confirmation": {
            domain: _commit_bucket(domain) for domain in ("overall", *DOMAIN_TOTALS)
        },
        "confirmation_prompt_truncated": 0,
        "confirmation_malformed_selections": 0,
        "confirmation_malformed_candidates": 0,
        "arm_malformed": {
            "revision": 0,
            "unchanged": 0,
            "self_refinement": 0,
        },
        "confirmation_capability_policy_rejections": 0,
        "selected_capability_policy_rejections": 0,
        "arm_capability_policy_rejected": {
            "revision": 0,
            "unchanged": 0,
            "self_refinement": 0,
        },
        "confirmation_maximum_swap_error": 0.0,
        "assessment_calls": 1289 * 3,
        "assessor_board_semantic_reads": 1,
        "confirmation_open_count": 1,
        "outcome_rows": 1289,
        "outcomes_sha256": "9" * 64,
        "score_authorization_sha256": "0" * 64,
        "assessor_board_sha256": "a" * 64,
        "score_consumption_state": "consumed",
        "authorization_consumed": True,
        "environment_verified": True,
        "environment_receipt_sha256": ENVIRONMENT_RECEIPT_SHA256,
        "environment_tree_sha256": ENVIRONMENT_TREE_SHA256,
        "code_sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
        "code_sandbox_binary_sha256": SANDBOX_BINARY_SHA256,
        "code_sandbox_probe_sha256": SANDBOX_RECEIPT_SHA256,
        "code_sandbox_probe_result_sha256": SANDBOX_RESULT_SHA256,
        "code_sandbox_probe_passed": True,
        "sandbox_receipt_sha256": SANDBOX_RECEIPT_SHA256,
        "mbpp_allocation_setup_status": "passed",
        "mbpp_allocation_setup_receipts": setup_receipts,
        "mbpp_allocation_setup_receipt_count": len(setup_receipts),
        "mbpp_allocation_setup_receipts_sha256": (
            mbpp_allocation_setup_receipts_sha256(setup_receipts)
        ),
        "inputs": {},
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }


Mutation = Callable[[dict[str, dict]], None]


def _fixture(
    root: Path, mutation: Mutation | None = None
) -> tuple[argparse.Namespace, dict[str, Path], dict[str, dict]]:
    reports = {
        "revision": _evaluation("revision"),
        "unchanged": _evaluation("unchanged"),
        "self_refinement": _evaluation("self_refinement"),
        "learned_commit": _commit(),
        "data_custody": {
            "schema": "shohin-pcf1-data-custody-v1",
            "status": "complete",
            "run_id": RUN_ID,
            "custody_verified": True,
            "source_disjoint": True,
            "confirmation_rows": 1289,
            "identity_order_sha256": IDENTITY_SHA256,
            "data_sha256": DATA_SHA256,
            "data_report_sha256": DATA_REPORT_SHA256,
            "confirmation_pairs_sha256": COMMIT_PAIRS_SHA256,
            "holdout_sealed": True,
            "product_sealed": True,
            "public_sealed": True,
            "holdout_access_count": 0,
            "product_access_count": 0,
            "public_access_count": 0,
        },
        "model_custody": {
            "schema": "shohin-pcf1-model-custody-v1",
            "status": "complete",
            "run_id": RUN_ID,
            "custody_verified": True,
            "model_root": MODEL_ROOT,
            "model_revision": MODEL_REVISION,
            "commit_training_report_sha256": COMMIT_TRAINING_REPORT_SHA256,
            "checkpoint_sha256s": {
                "trained_revision": ADAPTERS["revision"],
                "unchanged": ADAPTERS["unchanged"],
                "self_refinement": ADAPTERS["self_refinement"],
                "learned_commit_host": ADAPTERS["unchanged"],
                "learned_commit": COMMIT_CHECKPOINT,
            },
            "mbpp_calibration_setup_qualifications_verified": True,
            "mbpp_calibration_allocation_setup_receipts": CALIBRATION_SETUP_CUSTODY,
        },
        "runtime_custody": {
            "schema": "shohin-pcf1-runtime-custody-v1",
            "status": "complete",
            "run_id": RUN_ID,
            "custody_verified": True,
            "model_revision": MODEL_REVISION,
            "runtime_sha256": RUNTIME_SHA256,
            "environment_verified": True,
            "environment_receipt_sha256": ENVIRONMENT_RECEIPT_SHA256,
            "environment_tree_sha256": ENVIRONMENT_TREE_SHA256,
            "code_sandbox_verified": True,
            "code_sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
            "code_sandbox_binary_sha256": SANDBOX_BINARY_SHA256,
            "code_sandbox_probe_sha256": SANDBOX_RECEIPT_SHA256,
            "mbpp_calibration_setup_qualifications_verified": True,
            "mbpp_calibration_allocation_setup_receipts": CALIBRATION_SETUP_CUSTODY,
            "evaluation_settings": dict(COMMON_EVALUATION_SETTINGS),
            "commit_settings": {
                "model_loader": "multimodal",
                "max_sequence_length": 3072,
            },
        },
    }
    evaluation_accounting = {
        arm: {
            "prompt_tokens": reports[arm]["counters"]["prompt_tokens"],
            "generated_tokens": reports[arm]["counters"]["generated_tokens"],
            "wall_seconds": reports[arm]["aggregate_wall_seconds"],
            "peak_gpu_memory_bytes": reports[arm]["maximum_peak_gpu_memory_bytes"],
            "trainable_parameters": reports[arm]["trainable_parameters"],
            "trainable_parameter_name_sha256": reports[arm][
                "trainable_parameter_name_sha256"
            ],
            "lora_layer_indices": reports[arm]["lora_layer_indices"],
            "capability_policy_rejections": reports[arm]["counters"][
                "capability_policy_rejections"
            ],
        }
        for arm in ("revision", "unchanged", "self_refinement")
    }
    reports["model_custody"]["evaluation_accounting"] = evaluation_accounting
    reports["runtime_custody"]["evaluation_accounting"] = evaluation_accounting
    reports["runtime_custody"]["evaluation_accounting_sha256"] = hashlib.sha256(
        json.dumps(
            evaluation_accounting, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if mutation is not None:
        mutation(reports)
    consumption_path = root / "score_consumption.json"
    consumption = {
        "schema": "shohin-pcf1-score-consumption-v1",
        "status": "complete",
        "claim_state": "consumed",
        "run_id": RUN_ID,
        "score_output_root": str(root.resolve()),
        "score_authorization_sha256": reports["learned_commit"][
            "score_authorization_sha256"
        ],
        "confirmation_assessors_sha256": reports["learned_commit"][
            "assessor_board_sha256"
        ],
        "identity_order_sha256": IDENTITY_SHA256,
        "rows": 1289,
        "semantic_read_budget": 1,
        "sandbox_probe_sha256": SANDBOX_RECEIPT_SHA256,
        "sandbox_probe_result_sha256": SANDBOX_RESULT_SHA256,
        "sandbox_receipt_sha256": SANDBOX_RECEIPT_SHA256,
    }
    _write(consumption_path, consumption)
    reports["learned_commit"].update(
        {
            "score_consumption": str(consumption_path.resolve()),
            "score_consumption_sha256": _sha256(consumption_path),
        }
    )
    paths = {
        name: _write(root / f"{name}.json", report)
        for name, report in reports.items()
        if name != "learned_commit"
    }
    reports["learned_commit"]["inputs"] = {
        "arm_report_sha256s": {
            arm: _sha256(paths[arm])
            for arm in ("revision", "unchanged", "self_refinement")
        },
        "arm_candidates_sha256s": {
            arm: reports[arm]["candidates_sha256"]
            for arm in ("revision", "unchanged", "self_refinement")
        },
        "confirmation_assessors_sha256": reports["learned_commit"][
            "assessor_board_sha256"
        ],
    }
    paths["learned_commit"] = _write(
        root / "learned_commit.json", reports["learned_commit"]
    )
    paths["normalized"] = root / "normalized"
    args = argparse.Namespace(
        learned_commit_report=paths["learned_commit"],
        revision_report=paths["revision"],
        unchanged_report=paths["unchanged"],
        self_refinement_report=paths["self_refinement"],
        data_custody=paths["data_custody"],
        model_custody=paths["model_custody"],
        runtime_custody=paths["runtime_custody"],
        score_consumption=consumption_path,
        output=paths["normalized"],
    )
    return args, paths, reports


def _final_compare(
    root: Path, normalized: Path, custody_paths: dict[str, Path]
) -> dict:
    arm_paths = {
        arm: normalized / f"{arm}.json"
        for arm in (
            "learned_commit",
            "trained_revision",
            "unchanged",
            "self_refinement",
        )
    }
    compute = {
        "schema": "shohin-pcf1-compute-custody-v1",
        "status": "complete",
        "run_id": RUN_ID,
        "custody_verified": True,
        "data_custody_sha256": _sha256(custody_paths["data_custody"]),
        "model_custody_sha256": _sha256(custody_paths["model_custody"]),
        "runtime_custody_sha256": _sha256(custody_paths["runtime_custody"]),
        "arm_report_sha256s": {arm: _sha256(path) for arm, path in arm_paths.items()},
        "environment_verified": True,
        "environment_receipt_sha256": ENVIRONMENT_RECEIPT_SHA256,
        "environment_tree_sha256": ENVIRONMENT_TREE_SHA256,
        "code_sandbox_verified": True,
        "code_sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
        "code_sandbox_binary_sha256": SANDBOX_BINARY_SHA256,
        "code_sandbox_probe_sha256": SANDBOX_RECEIPT_SHA256,
        "mbpp_calibration_setup_qualifications_verified": True,
        "mbpp_calibration_allocation_setup_receipts": CALIBRATION_SETUP_CUSTODY,
        "mbpp_final_score_setup_qualifications_verified": True,
        "mbpp_final_score_allocation_setup_receipt_count": 2,
        "mbpp_final_score_allocation_setup_receipts_sha256": (
            mbpp_allocation_setup_receipts_sha256(
                [_setup_receipt(), _setup_receipt("value = 1\n")]
            )
        ),
        "one_open_verified": True,
        "score_consumption_state": "consumed",
        "score_consumption_sha256": _sha256(
            custody_paths["learned_commit"].parent / "score_consumption.json"
        ),
        "retry_count": 0,
        "successor_authorized": False,
        "successor_submitted": False,
        "accounting_verified": True,
        "charged_gpu_seconds": 1.0,
    }
    compute_path = _write(root / "compute_custody.json", compute)
    return compare(
        argparse.Namespace(
            learned_commit_report=arm_paths["learned_commit"],
            trained_revision_report=arm_paths["trained_revision"],
            unchanged_report=arm_paths["unchanged"],
            self_refinement_report=arm_paths["self_refinement"],
            data_custody=custody_paths["data_custody"],
            model_custody=custody_paths["model_custody"],
            runtime_custody=custody_paths["runtime_custody"],
            compute_custody=compute_path,
            output=root / "final.json",
        )
    )


def test_normalizes_native_reports_and_final_gate_passes(tmp_path: Path) -> None:
    args, paths, _ = _fixture(tmp_path)
    result = normalize(args)

    assert result["compute_custody_created"] is False
    assert result["next_action"] == "bind_normalized_report_hashes_in_compute_custody"
    assert sorted(path.name for path in paths["normalized"].iterdir()) == [
        "learned_commit.json",
        "self_refinement.json",
        "trained_revision.json",
        "unchanged.json",
    ]
    for arm, receipt in result["reports"].items():
        report_path = paths["normalized"] / f"{arm}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["schema"] == "shohin-pcf1-arm-report-v1"
        assert report["split"] == "development"
        assert report["identity_order_sha256"] == IDENTITY_SHA256
        assert receipt["sha256"] == _sha256(report_path)

    final = _final_compare(tmp_path, paths["normalized"], paths)
    assert final["gate_pass"] is True
    assert final["final_result"] == "PASS"
    assert final["automatic_successor_authorized"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda reports: reports["learned_commit"]["confirmation"][
                "overall"
            ].__setitem__("revision_correct", 451),
            "commit/candidate count binding",
        ),
        (
            lambda reports: reports["unchanged"].__setitem__(
                "data_report_sha256", "0" * 64
            ),
            "matched native evaluations",
        ),
        (
            lambda reports: reports["model_custody"]["checkpoint_sha256s"].__setitem__(
                "trained_revision", "0" * 64
            ),
            "model custody binding",
        ),
        (
            lambda reports: reports["runtime_custody"][
                "evaluation_settings"
            ].__setitem__("max_new_tokens", 769),
            "runtime custody binding",
        ),
        (
            lambda reports: reports["learned_commit"][
                "arm_capability_policy_rejected"
            ].__setitem__("revision", 1),
            "capability-policy counts",
        ),
    ],
)
def test_rejects_native_or_custody_hash_mismatch(
    tmp_path: Path, mutation: Mutation, message: str
) -> None:
    args, paths, _ = _fixture(tmp_path, mutation)
    with pytest.raises(PCF1NormalizationError, match=message):
        normalize(args)
    assert not paths["normalized"].exists()


def test_gate_failures_are_preserved_not_filtered_by_normalization(
    tmp_path: Path,
) -> None:
    def mutation(reports: dict[str, dict]) -> None:
        reports["revision"]["counters"]["max_token_exhausted"] = 1
        reports["revision"]["counters"]["empty_completions"] = 1
        reports["learned_commit"]["arm_malformed"]["revision"] = 1
        reports["learned_commit"]["confirmation_malformed_candidates"] = 1
        reports["learned_commit"]["confirmation_malformed_selections"] = 1
        reports["data_custody"]["holdout_sealed"] = False

    args, paths, _ = _fixture(tmp_path, mutation)
    normalize(args)
    revision = json.loads(
        (paths["normalized"] / "trained_revision.json").read_text(encoding="utf-8")
    )
    assert revision["truncation_count"] == 1

    final = _final_compare(tmp_path, paths["normalized"], paths)
    assert final["gate_pass"] is False
    assert final["checks"]["trained_revision_zero_truncation"] is False
    assert final["checks"]["trained_revision_zero_malformed"] is False
    assert final["checks"]["learned_commit_zero_malformed"] is False
    assert final["checks"]["holdout_sealed"] is False
    assert final["next_action"] == "stop_and_preserve_evidence"


def test_capability_policy_rejection_is_explicit_and_fails_gate(
    tmp_path: Path,
) -> None:
    def mutation(reports: dict[str, dict]) -> None:
        commit = reports["learned_commit"]
        commit["arm_capability_policy_rejected"]["revision"] = 1
        commit["confirmation_capability_policy_rejections"] = 1
        commit["arm_malformed"]["revision"] = 1
        commit["confirmation_malformed_candidates"] = 1

    args, paths, _ = _fixture(tmp_path, mutation)
    normalize(args)
    revision = json.loads(
        (paths["normalized"] / "trained_revision.json").read_text(encoding="utf-8")
    )
    assert revision["capability_policy_rejection_count"] == 1
    assert revision["malformed_count"] == 1

    final = _final_compare(tmp_path, paths["normalized"], paths)
    assert final["gate_pass"] is False
    assert final["checks"]["trained_revision_zero_malformed"] is False


def test_rejects_missing_or_tampered_score_consumption_marker(tmp_path: Path) -> None:
    args, paths, _ = _fixture(tmp_path)
    args.score_consumption.unlink()
    with pytest.raises(PCF1NormalizationError, match="unreadable.*score consumption"):
        normalize(args)
    assert not paths["normalized"].exists()

    args, paths, _ = _fixture(tmp_path / "tampered")
    marker = json.loads(args.score_consumption.read_text(encoding="utf-8"))
    marker["claim_state"] = "started"
    _write(args.score_consumption, marker)
    with pytest.raises(PCF1NormalizationError, match="score consumption binding"):
        normalize(args)
    assert not paths["normalized"].exists()


@pytest.mark.parametrize("mutation", ("tamper", "drop", "reorder"))
def test_rejects_tampered_dropped_or_reordered_final_setup_receipts(
    tmp_path: Path, mutation: str
) -> None:
    def mutate(reports: dict[str, dict]) -> None:
        receipts = reports["learned_commit"]["mbpp_allocation_setup_receipts"]
        if mutation == "tamper":
            receipts[0]["setup_source_sha256"] = "0" * 64
        elif mutation == "drop":
            receipts.pop()
        else:
            receipts.reverse()

    args, paths, _ = _fixture(tmp_path, mutate)
    with pytest.raises(PCF1NormalizationError, match="MBPP setup"):
        normalize(args)
    assert not paths["normalized"].exists()


def test_normalized_directory_is_write_once(tmp_path: Path) -> None:
    args, paths, _ = _fixture(tmp_path)
    first = normalize(args)
    before = {path.name: path.read_bytes() for path in paths["normalized"].iterdir()}
    with pytest.raises(PCF1NormalizationError, match="refusing existing"):
        normalize(args)
    after = {path.name: path.read_bytes() for path in paths["normalized"].iterdir()}
    assert after == before
    assert first["compute_custody_created"] is False
