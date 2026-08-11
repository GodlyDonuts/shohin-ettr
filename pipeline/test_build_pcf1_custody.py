"""Focused tests for the pure PCF1 custody and score-authorization compiler."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import pipeline.build_pcf1_custody as custody_builder
import pipeline.score_pcf1_commit as scorer
from build_pcf1_data import revision_prompt
from pipeline.build_pcf1_custody import (
    CALIBRATION_PAIR_SCHEMA,
    EXCLUDED_NODES,
    FINAL_ACCOUNTING_STAGES,
    MODEL_REVISION,
    PCF1CustodyError,
    PRESCORE_ACCOUNTING_STAGES,
    ARRAY_TASKS,
    GPU_STAGES,
    authorize_score,
    compute,
    precompute,
    sha256_file,
)
from hf_pcf1_train_commit import PAIR_SCHEMA as TRAINER_PAIR_SCHEMA
from pipeline.compare_pcf1 import compare
from pipeline.normalize_pcf1_reports import normalize
from pipeline.score_pcf1_commit import score
from pcf1_code_sandbox import (
    BWRAP_SHA256,
    CANDIDATE_FAILURE_EXIT_CODE,
    CANDIDATE_POLICY_SHA256,
    CANDIDATE_RANDOM_SEED,
    ELF_CLOSURE_AUDIT_SHA256,
    EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR,
    EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR_SHA256,
    MEMFD_ABI,
    POLICY_REJECTION_EXIT_CODE,
    RESOURCE_LIMIT_EXIT_CODE,
    PYTHON_EXECUTABLE,
    PYTHON_SHA256,
    SANDBOX_CONFIG_SHA256,
    SANDBOX_PROBES,
    SANDBOX_RUNTIME_TREE_BYTES,
    SANDBOX_RUNTIME_TREE_DIRECTORIES,
    SANDBOX_RUNTIME_TREE_ENTRIES,
    SANDBOX_RUNTIME_TREE_FILES,
    SANDBOX_RUNTIME_TREE_SHA256,
    SETUP_FAILURE_EXIT_CODE,
    INFRASTRUCTURE_FAILURE_EXIT_CODE,
    SYSTEM_LIBRARY_BINDINGS,
    TRUSTED_COMPLETION_EXIT_CODE,
    TEST_FAILURE_EXIT_CODE,
)

RUN_ID = "pcf1-custody-test"
SEALED_ACCESS = {"holdout": 0, "product": 0, "public": 0}
DOMAIN_TOTALS = {"math500": 623, "bbh_logic": 637, "mbpp": 29}
TASKS = tuple(DOMAIN_TOTALS)
ENVIRONMENT_SHA256 = "e" * 64
ENVIRONMENT_TREE_SHA256 = "d" * 64


def _sandbox_payload() -> dict[str, Any]:
    probes = {name: True for name in SANDBOX_PROBES}
    probe_sha = hashlib.sha256(
        json.dumps(probes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": "shohin-pcf1-code-sandbox-receipt-v1",
        "status": "pass",
        "bwrap_path": "/usr/bin/bwrap",
        "bwrap_sha256": BWRAP_SHA256,
        "bwrap_version": "bubblewrap 0.4.0",
        "python_executable": str(PYTHON_EXECUTABLE),
        "python_sha256": PYTHON_SHA256,
        "sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
        "candidate_policy_sha256": CANDIDATE_POLICY_SHA256,
        "trusted_completion_exit_code": TRUSTED_COMPLETION_EXIT_CODE,
        "candidate_failure_exit_code": CANDIDATE_FAILURE_EXIT_CODE,
        "infrastructure_failure_exit_code": INFRASTRUCTURE_FAILURE_EXIT_CODE,
        "test_failure_exit_code": TEST_FAILURE_EXIT_CODE,
        "setup_failure_exit_code": SETUP_FAILURE_EXIT_CODE,
        "policy_rejection_exit_code": POLICY_REJECTION_EXIT_CODE,
        "resource_limit_exit_code": RESOURCE_LIMIT_EXIT_CODE,
        "candidate_random_seed": CANDIDATE_RANDOM_SEED,
        "python_runtime_descriptor": EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR,
        "python_runtime_descriptor_sha256": (
            EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR_SHA256
        ),
        "memfd_abi": MEMFD_ABI,
        "sandbox_runtime_tree_sha256": SANDBOX_RUNTIME_TREE_SHA256,
        "sandbox_runtime_tree_entries": SANDBOX_RUNTIME_TREE_ENTRIES,
        "sandbox_runtime_tree_files": SANDBOX_RUNTIME_TREE_FILES,
        "sandbox_runtime_tree_directories": SANDBOX_RUNTIME_TREE_DIRECTORIES,
        "sandbox_runtime_tree_bytes": SANDBOX_RUNTIME_TREE_BYTES,
        "elf_closure_audit_sha256": ELF_CLOSURE_AUDIT_SHA256,
        "system_library_members": [
            {
                "source": str(source),
                "destination": destination,
                "sha256": digest,
                "size": size,
            }
            for source, destination, digest, size in SYSTEM_LIBRARY_BINDINGS
        ],
        "clear_environment": True,
        "network_namespace": "isolated",
        "candidate_read_only": True,
        "candidate_direct_pid_1": True,
        "site_packages_visible": False,
        "probe_results": probes,
        "probe_sha256": probe_sha,
        "sandbox_isolation_passed": True,
    }


def _sandbox_sha256() -> str:
    return hashlib.sha256(
        (json.dumps(_sandbox_payload(), indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()


def _setup_receipt(setup_source: str = "") -> dict[str, Any]:
    receipt = {
        "schema": "shohin-pcf1-mbpp-setup-qualification-v1",
        "status": "pass",
        "setup_source_sha256": hashlib.sha256(setup_source.encode()).hexdigest(),
        "candidate_policy_sha256": CANDIDATE_POLICY_SHA256,
        "sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
        "allocation_probe_sha256": _sandbox_payload()["probe_sha256"],
        "setup_qualification_mode": "compile_only_before_candidate",
        "termination_classification": "trusted_tests_completed",
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return receipt


@pytest.fixture(autouse=True)
def _qualified_external_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = {
        "environment_tree": {"sha256": ENVIRONMENT_TREE_SHA256},
        "environment_runtime_sha256": "1" * 64,
        "pip_freeze_sha256": "2" * 64,
        "python": {"executable_sha256": "3" * 64},
    }
    monkeypatch.setattr(
        custody_builder, "validate_environment_receipt", lambda *_: environment
    )
    monkeypatch.setattr(scorer, "validate_environment_receipt", lambda *_: environment)
    monkeypatch.setattr(scorer, "qualify_allocation", _sandbox_payload)
    monkeypatch.setattr(
        scorer, "qualify_mbpp_assessor_setups", lambda _assessors: [_setup_receipt()]
    )


def test_calibration_pair_schema_matches_trainer() -> None:
    assert CALIBRATION_PAIR_SCHEMA == TRAINER_PAIR_SCHEMA


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _identity_order(identities: list[str]) -> str:
    return hashlib.sha256(("\n".join(identities) + "\n").encode()).hexdigest()


def _task(index: int) -> str:
    if index < DOMAIN_TOTALS["math500"]:
        return "math500"
    if index < DOMAIN_TOTALS["math500"] + DOMAIN_TOTALS["bbh_logic"]:
        return "bbh_logic"
    return "mbpp"


def _namespace(**values: Any) -> argparse.Namespace:
    return argparse.Namespace(**values)


def _manifest(root: Path, path: Path) -> tuple[Path, str]:
    entries: list[tuple[str, str]] = []
    for member in sorted(item for item in root.rglob("*") if item.is_file()):
        if member == path:
            continue
        entries.append((member.relative_to(root).as_posix(), sha256_file(member)))
    path.write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in entries),
        encoding="utf-8",
    )
    return path, sha256_file(path)


def _evaluation(
    *,
    arm: str,
    split: str,
    rows: int,
    model_root: str,
    adapter_sha256: str,
    data_sha256: str,
    data_report_sha256: str,
    candidates_path: Path | None = None,
) -> dict[str, Any]:
    candidates_sha = (
        sha256_file(candidates_path)
        if candidates_path is not None
        else _sha(f"{split}-{arm}-candidates")
    )
    calibration = split == "calibration"
    shard_ranges = [(rows * index // 4, rows * (index + 1) // 4) for index in range(4)]
    setup_shards = (
        [
            {
                "shard_index": index,
                "row_start": start,
                "row_end": end,
                "receipts": [_setup_receipt()],
                "receipt_count": 1,
                "receipts_sha256": custody_builder.mbpp_allocation_setup_receipts_sha256(
                    [_setup_receipt()]
                ),
            }
            for index, (start, end) in enumerate(shard_ranges)
        ]
        if calibration
        else []
    )
    setup_shards_sha256 = (
        hashlib.sha256(
            b"".join(
                (
                    json.dumps(shard, sort_keys=True, separators=(",", ":")) + "\n"
                ).encode()
                for shard in setup_shards
            )
        ).hexdigest()
        if calibration
        else None
    )
    return {
        "schema": "shohin-pcf1-merged-evaluation-v1",
        "status": "complete",
        "arm": arm,
        "split": split,
        "model_root": model_root,
        "model_revision": MODEL_REVISION,
        "model_loader": "multimodal",
        "generation_mode": "greedy",
        "max_new_tokens": 768,
        "seed": 2026080816,
        "batch_size": 2,
        "shard_count": 4,
        "adapter_checkpoint_sha256": adapter_sha256,
        "adapter_metadata_sha256": "b" * 64,
        "trainable_parameters": 1234,
        "trainable_parameter_name_sha256": "a" * 64,
        "lora_layer_indices": [30, 31, 32, 33],
        "environment_verified": True,
        "environment_receipt_sha256": ENVIRONMENT_SHA256,
        "environment_tree_sha256": ENVIRONMENT_TREE_SHA256,
        "code_sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
        "code_sandbox_binary_sha256": BWRAP_SHA256,
        "code_sandbox_probe_sha256": _sandbox_sha256() if calibration else None,
        "code_sandbox_probe_result_sha256": (
            _sandbox_payload()["probe_sha256"] if calibration else None
        ),
        "sandbox_receipt_sha256": _sandbox_sha256() if calibration else None,
        "shard_sandbox_probe_sha256s": ([_sandbox_sha256()] * 4 if calibration else []),
        "code_sandbox_status": (
            "passed" if calibration else "not_applicable_no_code_scoring"
        ),
        "code_sandbox_probe_passed": True if calibration else None,
        "inputs": [
            {
                "shard_index": index,
                "row_start": start,
                "row_end": end,
            }
            for index, (start, end) in enumerate(shard_ranges)
        ],
        "mbpp_allocation_setup_status": (
            "passed" if calibration else "not_applicable_no_code_scoring"
        ),
        "mbpp_allocation_setup_receipt_shards": setup_shards,
        "mbpp_allocation_setup_receipt_count": len(setup_shards),
        "mbpp_allocation_setup_receipt_shards_sha256": setup_shards_sha256,
        "data_sha256": data_sha256,
        "data_report_sha256": data_report_sha256,
        "full_row_count": rows,
        "candidates_output": (
            str(candidates_path.resolve()) if candidates_path is not None else None
        ),
        "candidates_sha256": candidates_sha,
        "metrics": {} if split == "calibration" else None,
        "assessment_mode": (
            "calibration_immediate"
            if split == "calibration"
            else "confirmation_deferred"
        ),
        "assessor_board_access_count": 0,
        "runtime_fields": (
            ["source_prompt", "internal_draft.completion"]
            if arm == "self_refinement"
            else ["question"]
        ),
        "counters": {
            "rows": rows,
            "prompt_tokens": rows * 10,
            "generated_tokens": rows,
            "max_token_exhausted": 0,
            "empty_completions": 0,
            "capability_policy_rejections": 0,
        },
        "exact_identity_coverage": True,
        "aggregate_prompt_tokens": rows * 10,
        "aggregate_wall_seconds": 10.0,
        "aggregate_gpu_seconds": 10.0,
        "maximum_peak_gpu_memory_bytes": 1024,
        "sealed_access": dict(SEALED_ACCESS),
    }


def _training_report(
    *, model_root: str, data_sha256: str, warm_start_sha256: str | None
) -> dict[str, Any]:
    revision = warm_start_sha256 is not None
    return {
        "schema": "shohin-hf-product-reasoning-training-v1",
        "status": "complete",
        "model_root": model_root,
        "model_revision": MODEL_REVISION,
        "model_loader": "multimodal",
        "arm": "baseline",
        "data_sha256": data_sha256,
        "updates": 256,
        "batch_size": 1,
        "gradient_accumulation": 8 if revision else 16,
        "max_sequence_length": 4096 if revision else 1024,
        "learning_rate": 2e-5 if revision else 2e-4,
        "lora_layers": 4,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_scope": "token_mixer",
        "seed": 2026080815 if revision else 2026080711,
        "data_seed": 2026080814 if revision else 20260802,
        "warm_start_sha256": warm_start_sha256,
        "trainable_parameters": 1234,
        "trainable_parameter_name_sha256": "a" * 64,
        "lora_layer_indices": [30, 31, 32, 33],
        "selected_rows": 9655 if revision else 100000,
        "environment_verified": True,
        "environment_receipt_sha256": ENVIRONMENT_SHA256,
        "environment_tree_sha256": ENVIRONMENT_TREE_SHA256,
    }


def _accounting(root: Path, name: str) -> tuple[Path, Path]:
    stages = list(
        PRESCORE_ACCOUNTING_STAGES if name == "prescore" else FINAL_ACCOUNTING_STAGES
    )
    job_ids = {stage: str(1000 + index) for index, stage in enumerate(stages)}
    stage_resources = {
        stage: {
            "gpus": int(stage in GPU_STAGES),
            "is_array": stage in ARRAY_TASKS,
            "array_tasks": ARRAY_TASKS.get(stage, 1),
        }
        for stage in stages
    }
    dispatch = _write_json(
        root / f"{name}_dispatch.json",
        {
            "schema": "shohin-pcf1-dispatch-v1",
            "status": "submitted",
            "run_id": RUN_ID,
            "terminal_stage": "final_compare",
            "retry_authorized": False,
            "successor_authorized": False,
            "stop_after_gate": True,
            "job_ids": job_ids,
            "accounting_predecessors": stages,
            "stage_resources": stage_resources,
        },
    )
    jobs: dict[str, dict[str, Any]] = {}
    for stage, job_id in job_ids.items():
        resource = stage_resources[stage]
        records = []
        for index in range(resource["array_tasks"]):
            records.append(
                {
                    "job_id_raw": (
                        f"{job_id}_{index}" if resource["is_array"] else job_id
                    ),
                    "state": "COMPLETED",
                    "partition": "normal",
                    "elapsed_raw": 1,
                    "alloc_tres": (
                        "billing=1,cpu=1,gres/gpu=1"
                        if resource["gpus"]
                        else "billing=1,cpu=1"
                    ),
                    "node_list": "evc01",
                    "exit_code": "0:0",
                    "restarts": 0,
                    "allocated_gpus": resource["gpus"],
                    "allocated_gpu_types": (
                        {"nvidia_h100_pcie": 1} if resource["gpus"] else {}
                    ),
                    "charged_gpu_seconds": resource["gpus"],
                }
            )
        jobs[stage] = {
            "submitted_job_id": job_id,
            "records": records,
            "charged_gpu_seconds": float(resource["gpus"] * resource["array_tasks"]),
        }
    accounting = _write_json(
        root / f"{name}_accounting.json",
        {
            "schema": "shohin-pcf1-slurm-accounting-v1",
            "status": "complete",
            "run_id": RUN_ID,
            "partition": "normal",
            "excluded_nodes": EXCLUDED_NODES,
            "jobs": jobs,
            "required_stages": stages,
            "charged_gpu_seconds": sum(
                float(job["charged_gpu_seconds"]) for job in jobs.values()
            ),
            "all_required_complete": True,
            "retry_count": 0,
            "successor_authorized": False,
            "successor_submitted": False,
        },
    )
    return dispatch, accounting


def _fixture(root: Path) -> dict[str, Any]:
    checkpoint_paths: dict[str, Path] = {}
    for name in ("b1", "revision", "commit"):
        path = root / f"{name}.pt"
        path.write_bytes(f"exact {name} checkpoint\n".encode())
        checkpoint_paths[name] = path
    checkpoint_hashes = {
        name: sha256_file(path) for name, path in checkpoint_paths.items()
    }

    train_ids = sorted(_sha(f"train-{index}") for index in range(5824))
    confirmation_ids = sorted(_sha(f"confirmation-{index}") for index in range(1289))
    train_tasks = {
        identity: TASKS[index % len(TASKS)] for index, identity in enumerate(train_ids)
    }
    confirmation_tasks = {
        identity: _task(index) for index, identity in enumerate(confirmation_ids)
    }
    train_source_rows = [
        {
            "schema": "shohin-pcf1-train-source-v1",
            "identity_sha256": identity,
            "split": "train",
            "task": train_tasks[identity],
            "outcome_class": "base_only" if index < 1277 else "both_wrong",
            "source_prompt": f"source {identity}",
            "response": f"response {identity}",
            "target_kind": "verified_candidate",
            "assessor": {
                "schema": "shohin-pcf1-assessor-v1",
                "identity_sha256": identity,
                "task": train_tasks[identity],
            },
            "runtime_fields": ["source_prompt"],
            "supervisor_only_fields": [
                "response",
                "target_kind",
                "assessor",
                "task",
                "outcome_class",
            ],
        }
        for index, identity in enumerate(train_ids)
    ]
    development_source_rows = [
        {
            "schema": "shohin-pcf1-development-source-v1",
            "identity_sha256": identity,
            "split": "development",
            "task": confirmation_tasks[identity],
            "source_prompt": f"source {identity}",
            "runtime_fields": ["source_prompt"],
            "supervisor_only_fields": ["task"],
        }
        for identity in confirmation_ids
    ]
    train_sources = _write_jsonl(root / "train_sources.jsonl", train_source_rows)
    development_sources = _write_jsonl(
        root / "development_sources.jsonl", development_source_rows
    )
    correctness: dict[str, set[str]] = {
        "revision": set(),
        "unchanged": set(),
        "self_refinement": set(),
    }
    domain_geometry = {
        "math500": (130, 100, 120, 95),
        "bbh_logic": (312, 280, 284, 274),
        "mbpp": (10, 7, 9, 5),
    }
    for task in TASKS:
        identities = [
            identity
            for identity in confirmation_ids
            if confirmation_tasks[identity] == task
        ]
        revision_count, unchanged_count, self_count, overlap = domain_geometry[task]
        correctness["revision"].update(identities[:revision_count])
        correctness["unchanged"].update(identities[:overlap])
        correctness["unchanged"].update(
            identities[revision_count : revision_count + unchanged_count - overlap]
        )
        correctness["self_refinement"].update(identities[:self_count])
    drafts: list[dict[str, Any]] = []
    for split, identities, tasks in (
        ("train", train_ids, train_tasks),
        ("development", confirmation_ids, confirmation_tasks),
    ):
        for identity in identities:
            drafts.append(
                {
                    "schema": "shohin-pcf1-model-draft-v1",
                    "identity_sha256": identity,
                    "split": split,
                    "task": tasks[identity],
                    "completion": f"draft {identity}",
                    "generated_tokens": 1,
                    "max_token_exhausted": False,
                    "prompt_sha256": _sha(f"source {identity}"),
                    "adapter_checkpoint_sha256": checkpoint_hashes["b1"],
                    "model_revision": MODEL_REVISION,
                    "finish_reason": "stop",
                    "wall_seconds": 0.1,
                }
            )
    draft_by_identity = {row["identity_sha256"]: row for row in drafts}
    drafts_path = _write_jsonl(root / "merged_drafts.jsonl", drafts)

    revision_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for index, identity in enumerate(train_ids):
        draft = draft_by_identity[identity]
        outcome = "base_only" if index < 1277 else "both_wrong"
        presentations = 4 if outcome == "base_only" else 1
        question = revision_prompt(f"source {identity}", draft["completion"])
        for presentation in range(presentations):
            revision_rows.append(
                {
                    "schema": "shohin-pcf1-revision-train-v1",
                    "identity_sha256": _sha(f"revision-{identity}-{presentation}"),
                    "source_identity_sha256": identity,
                    "task": train_tasks[identity],
                    "outcome_class": outcome,
                    "presentation": presentation,
                    "question": question,
                    "response": f"response {identity}",
                    "target_kind": "verified_candidate",
                    "model_owned_draft_sha256": hashlib.sha256(
                        draft["completion"].encode()
                    ).hexdigest(),
                    "runtime_fields": ["question"],
                }
            )
        calibration_rows.append(
            {
                "schema": "shohin-pcf1-eval-v1",
                "identity_sha256": identity,
                "split": "calibration",
                "task": train_tasks[identity],
                "question": question,
                "source_prompt": f"source {identity}",
                "internal_draft": draft,
                "assessor": {"schema": "shohin-pcf1-assessor-v1"},
                "candidates": [],
                "runtime_fields": ["question"],
                "internal_draft_visible": True,
                "external_candidate_text_visible": False,
            }
        )
    confirmation_rows = [
        {
            "schema": "shohin-pcf1-eval-v1",
            "identity_sha256": identity,
            "split": "confirmation",
            "task": confirmation_tasks[identity],
            "question": revision_prompt(
                f"source {identity}", draft_by_identity[identity]["completion"]
            ),
            "source_prompt": f"source {identity}",
            "internal_draft": draft_by_identity[identity],
            "candidates": [],
            "runtime_fields": ["question", "source_prompt"],
            "internal_draft_visible": True,
            "external_candidate_text_visible": False,
        }
        for identity in confirmation_ids
    ]
    assessor_rows = [
        {
            "schema": "shohin-pcf1-confirmation-assessor-v1",
            "identity_sha256": identity,
            "split": "confirmation",
            "task": confirmation_tasks[identity],
            "assessor": {
                "identity_sha256": identity,
                "task": confirmation_tasks[identity],
            },
        }
        for identity in confirmation_ids
    ]
    revision_data = _write_jsonl(root / "revision_train.jsonl", revision_rows)
    calibration_data = _write_jsonl(root / "calibration.jsonl", calibration_rows)
    confirmation_data = _write_jsonl(root / "confirmation.jsonl", confirmation_rows)
    confirmation_assessors = _write_jsonl(root / "assessors.jsonl", assessor_rows)
    confirmation_assessor_receipt = _write_json(
        root / "assessor_receipt.json",
        {
            "schema": "shohin-pcf1-confirmation-assessor-receipt-v1",
            "status": "complete",
            "board_sha256": sha256_file(confirmation_assessors),
            "rows": 1289,
            "semantic_access": "final_score_only",
        },
    )
    reference_sandbox_receipt = _write_json(
        root / "reference_sandbox_receipt.json", _sandbox_payload()
    )
    reference_identities = sorted(
        [identity for identity in train_ids if train_tasks[identity] == "mbpp"]
        + [
            identity
            for identity in confirmation_ids
            if confirmation_tasks[identity] == "mbpp"
        ]
    )
    reference_preflight_rows = _write_jsonl(
        root / "mbpp_reference_preflight.jsonl",
        [
            {
                "identity_sha256": identity,
                "split": ("train" if identity in train_tasks else "development"),
                "candidate_source_sha256": _sha(f"candidate-{identity}"),
                "program_sha256": _sha(f"program-{identity}"),
                "setup_source_sha256": _sha(""),
                "setup_qualification_sha256": "8" * 64,
                "candidate_policy_sha256": CANDIDATE_POLICY_SHA256,
                "sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
                "allocation_probe_sha256": _sandbox_payload()["probe_sha256"],
                "reference_assessment_mode": "trusted_reference",
                "generated_candidate_policy_applied": False,
                "termination_classification": "trusted_tests_completed",
            }
            for identity in reference_identities
        ],
    )
    setup_qualification_sha256 = hashlib.sha256(
        (
            json.dumps(
                {
                    "setup_source_sha256": _sha(""),
                    "setup_qualification_sha256": "8" * 64,
                },
                sort_keys=True,
            )
            + "\n"
        ).encode()
    ).hexdigest()

    source_freeze = _write_json(
        root / "source_freeze.json",
        {
            "schema": "shohin-pcf1-data-freeze-report-v1",
            "status": "complete",
            "split_seed": 2026080811,
            "counts": {"train": 5824, "development": 1289, "holdout": 1279},
            "identity_receipts": {
                "train": {
                    "count": 5824,
                    "ordered_identity_sha256": _identity_order(train_ids),
                },
                "development": {
                    "count": 1289,
                    "ordered_identity_sha256": _identity_order(confirmation_ids),
                },
                "holdout": {"count": 1279, "ordered_identity_sha256": "f" * 64},
            },
            "inputs": {
                "pairs_sha256": "45f1d66ce5e87dc2a1f4c3594bdde2bae26e9417e879d16eb4eddb228b696afe",
                "source_bank_sha256s": [
                    "0b6d068b4d71f407cb234579b9278dc640df09139ea906dd0f52a6ab71e05398",
                    "5a96859fd9088cde598b61da60dd2c6cb7281323ee06c034742a1b4e0e237017",
                    "e0ede83257e441050a019f59fb13d9c85bd6cba1d6a755ab86fb7129966ddbe5",
                ],
            },
            "draft_training_reference": {
                "corpus_sha256": "2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549",
                "content_copied": False,
                "path_recorded": False,
                "hash_reference_only": True,
            },
            "revision_training_geometry": {
                "unique_train_identities": 5824,
                "presentations": 9655,
                "single_correct_presentations_per_identity": 4,
                "other_presentations_per_identity": 1,
            },
            "outputs": {
                "train_sources.jsonl": {
                    "sha256": sha256_file(train_sources),
                    "rows": 5824,
                },
                "development_sources.jsonl": {
                    "sha256": sha256_file(development_sources),
                    "rows": 1289,
                },
                "confirmation_assessor_receipt": {
                    "sha256": sha256_file(confirmation_assessor_receipt),
                    "board_sha256": sha256_file(confirmation_assessors),
                    "rows": 1,
                },
                "reference_sandbox_receipt.json": {
                    "sha256": sha256_file(reference_sandbox_receipt),
                    "rows": 1,
                },
                "mbpp_reference_preflight.jsonl": {
                    "sha256": sha256_file(reference_preflight_rows),
                    "rows": len(reference_identities),
                },
            },
            "mbpp_reference_preflight": {
                "schema": "shohin-pcf1-mbpp-reference-preflight-v1",
                "status": "pass",
                "scope": ["train", "development"],
                "rows": len(reference_identities),
                "ordered_identity_sha256": _identity_order(reference_identities),
                "row_receipts_sha256": sha256_file(reference_preflight_rows),
                "unique_setups": 1,
                "setup_pair_receipts_sha256": setup_qualification_sha256,
                "candidate_policy_sha256": CANDIDATE_POLICY_SHA256,
                "sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
                "allocation_probe_sha256": _sandbox_payload()["probe_sha256"],
                "reference_assessment_mode": "trusted_reference",
                "generated_candidate_policy_applied": False,
                "all_references_passed": True,
                "all_sandbox_passed": True,
                "holdout_reference_content_accesses": 0,
                "sandbox_receipt_sha256": sha256_file(reference_sandbox_receipt),
            },
            "source_disjoint": True,
            "sealed_content_materialized": False,
            "protected_board_inputs": 0,
            "public_inputs": 0,
        },
    )
    draft_report = _write_json(
        root / "merged_drafts_report.json",
        {
            "schema": "shohin-pcf1-merged-drafts-v1",
            "status": "complete",
            "model_revision": MODEL_REVISION,
            "model_loader": "multimodal",
            "adapter_checkpoint_sha256": checkpoint_hashes["b1"],
            "environment_receipt_sha256": ENVIRONMENT_SHA256,
            "source_report_sha256": sha256_file(source_freeze),
            "source_counts": {"train": 5824, "development": 1289},
            "generation_mode": "greedy",
            "thinking_enabled": False,
            "max_new_tokens": 768,
            "seed": 2026080818,
            "full_row_count": 7113,
            "rows": 7113,
            "output": str(drafts_path.resolve()),
            "output_sha256": sha256_file(drafts_path),
            "exact_identity_coverage": True,
            "sealed_access": dict(SEALED_ACCESS),
        },
    )
    data_report = _write_json(
        root / "data_report.json",
        {
            "schema": "shohin-pcf1-data-report-v1",
            "status": "complete",
            "split_seed": 2026080811,
            "inputs": {
                "freeze_report_sha256": sha256_file(source_freeze),
                "drafts_sha256": sha256_file(drafts_path),
                "draft_rows": 7113,
            },
            "identity_receipts": {
                "train": {
                    "count": 5824,
                    "ordered_identity_sha256": _identity_order(train_ids),
                },
                "development": {
                    "count": 1289,
                    "ordered_identity_sha256": _identity_order(confirmation_ids),
                },
                "sealed": {
                    "count": 1279,
                    "ordered_identity_sha256": "f" * 64,
                    "content_materialized": False,
                },
            },
            "counts": {
                "train_unique_identities": 5824,
                "revision_train_presentations": 9655,
                "calibration_rows": 5824,
                "confirmation_rows": 1289,
            },
            "revision_presentation_rule": {
                "single_correct": 4,
                "both_correct_or_both_wrong": 1,
            },
            "outputs": {
                "revision_train": {
                    "path": str(revision_data.resolve()),
                    "sha256": sha256_file(revision_data),
                    "rows": 9655,
                },
                "calibration": {
                    "path": str(calibration_data.resolve()),
                    "sha256": sha256_file(calibration_data),
                    "rows": 5824,
                },
                "confirmation": {
                    "path": str(confirmation_data.resolve()),
                    "sha256": sha256_file(confirmation_data),
                    "rows": 1289,
                },
                "confirmation_assessors": {
                    "sha256": sha256_file(confirmation_assessors),
                    "rows": 1289,
                    "semantic_access": "final_score_only",
                },
                "confirmation_assessor_receipt": {
                    "sha256": sha256_file(confirmation_assessor_receipt),
                    "rows": 1,
                },
            },
            "confirmation_assessor_access": {
                "semantic_reads": 0,
                "authorized_reader": "score_pcf1_commit.py",
            },
            "source_disjoint": True,
            "sealed_content_materialized": False,
            "protected_board_inputs": 0,
            "public_inputs": 0,
            "sealed_access": dict(SEALED_ACCESS),
        },
    )

    model_root_path = root / "ministral_model"
    model_root_path.mkdir()
    (model_root_path / "config.json").write_text('{"model_type":"mistral3"}\n')
    (model_root_path / "weights.safetensors").write_bytes(b"model weights\n")
    model_manifest, model_manifest_sha = _manifest(
        model_root_path, root / "model_manifest.sha256"
    )
    model_root = str(model_root_path.resolve())
    runtime_root = root / "runtime"
    runtime_root.mkdir()
    (runtime_root / "evaluate.py").write_text("# frozen runtime\n")
    runtime_manifest, runtime_manifest_sha = _manifest(
        runtime_root, runtime_root / "SHA256SUMS"
    )
    environment_receipt = root / "environment_receipt.json"
    environment_receipt.write_text("qualified environment\n", encoding="utf-8")
    sandbox_receipt = _write_json(root / "sandbox_receipt.json", _sandbox_payload())
    assert sha256_file(sandbox_receipt) == _sandbox_sha256()
    compute_host_receipt = _write_json(
        root / "compute_host_receipt.json",
        {
            "schema": "shohin-pcf1-compute-host-receipt-v1",
            "status": "complete",
            "partition": "normal",
            "node": "evc01",
            "excluded_nodes": sorted(EXCLUDED_NODES),
            "nvidia_smi_invoked_path": "/usr/bin/nvidia-smi",
            "nvidia_smi_resolved_path": "/usr/bin/nvidia-smi",
            "nvidia_smi_sha256": "9" * 64,
            "nvidia_smi_version": "NVIDIA-SMI 550.54.15",
            "visible_gpu_count": 1,
            "gpu_name": "NVIDIA H100 PCIe",
            "driver_version": "550.54.15",
            "pci_bus_id": "00000000:01:00.0",
        },
    )
    compute_host = json.loads(compute_host_receipt.read_text(encoding="utf-8"))
    compute_host_sha256 = sha256_file(compute_host_receipt)
    calibration_sandbox_probes = {
        arm: [
            _write_json(
                root / f"calibration_{arm}_sandbox_probe_{index}.json",
                _sandbox_payload(),
            )
            for index in range(4)
        ]
        for arm in ("revision", "unchanged")
    }

    candidate_paths: dict[str, Path] = {}
    for arm in ("revision", "unchanged", "self_refinement"):
        candidate_paths[arm] = _write_jsonl(
            root / f"{arm}_candidates.jsonl",
            [
                {
                    "schema": "shohin-pcf1-candidate-v1",
                    "arm": arm,
                    "identity_sha256": identity,
                    "task": confirmation_tasks[identity],
                    "completion": f"{arm} completion {identity}",
                    "generated_tokens": 1,
                    "max_token_exhausted": False,
                }
                for identity in confirmation_ids
            ],
        )
    evaluation_paths: dict[str, Path] = {}
    for arm, checkpoint in (
        ("revision", "revision"),
        ("unchanged", "b1"),
        ("self_refinement", "b1"),
    ):
        evaluation_paths[arm] = _write_json(
            root / f"{arm}_report.json",
            _evaluation(
                arm=arm,
                split="confirmation",
                rows=1289,
                model_root=model_root,
                adapter_sha256=checkpoint_hashes[checkpoint],
                data_sha256=sha256_file(confirmation_data),
                data_report_sha256=sha256_file(data_report),
                candidates_path=candidate_paths[arm],
            ),
        )
    calibration_eval_paths = {
        arm: _write_json(
            root / f"calibration_{arm}_report.json",
            _evaluation(
                arm=arm,
                split="calibration",
                rows=5824,
                model_root=model_root,
                adapter_sha256=checkpoint_hashes[
                    "revision" if arm == "revision" else "b1"
                ],
                data_sha256=sha256_file(calibration_data),
                data_report_sha256=sha256_file(data_report),
            ),
        )
        for arm in ("revision", "unchanged")
    }

    def calibration_split(identity: str) -> str:
        digest = hashlib.sha256(f"2026080820\0{identity}".encode()).digest()
        return (
            "calibration_train"
            if int.from_bytes(digest[:8], "big") % 10_000 < 8_000
            else "calibration_development"
        )

    calibration_pairs = sorted(
        [
            {
                "schema": "shohin-pcf1-whole-trajectory-pair-v1",
                "identity_sha256": row["identity_sha256"],
                "split": calibration_split(row["identity_sha256"]),
                "task": row["task"],
                "question": row["question"],
                "outcome_class": "both_wrong",
                "candidates": [
                    {
                        "lineage": "revision",
                        "completion": "revision calibration",
                        "correct": False,
                    },
                    {
                        "lineage": "unchanged",
                        "completion": "unchanged calibration",
                        "correct": False,
                    },
                ],
            }
            for row in calibration_rows
        ],
        key=lambda row: (row["split"], row["identity_sha256"]),
    )
    calibration_pairs_path = _write_jsonl(
        root / "calibration_pairs.jsonl", calibration_pairs
    )
    calibration_counts: dict[str, int] = {}
    for row in calibration_pairs:
        calibration_counts[row["split"]] = calibration_counts.get(row["split"], 0) + 1
    calibration_pair_report = _write_json(
        root / "calibration_pair_report.json",
        {
            "schema": "shohin-pcf1-commit-pair-report-v1",
            "status": "complete",
            "output": str(calibration_pairs_path.resolve()),
            "output_sha256": sha256_file(calibration_pairs_path),
            "seed": 2026080820,
            "counts": calibration_counts,
            "confirmation_rows_loaded": 0,
            "source_disjoint_from_confirmation": True,
            "inputs": {
                "data": str(calibration_data.resolve()),
                "data_sha256": sha256_file(calibration_data),
                "revision_report_sha256": sha256_file(
                    calibration_eval_paths["revision"]
                ),
                "unchanged_report_sha256": sha256_file(
                    calibration_eval_paths["unchanged"]
                ),
                "revision_candidates_sha256": json.loads(
                    calibration_eval_paths["revision"].read_text()
                )["candidates_sha256"],
                "unchanged_candidates_sha256": json.loads(
                    calibration_eval_paths["unchanged"].read_text()
                )["candidates_sha256"],
            },
            "sealed_access": dict(SEALED_ACCESS),
        },
    )
    confirmation_pairs = [
        {
            "schema": "shohin-pcf1-confirmation-pair-v1",
            "identity_sha256": row["identity_sha256"],
            "split": "confirmation",
            "task": row["task"],
            "question": row["question"],
            "candidates": [
                {
                    "lineage": "revision",
                    "completion": f"revision completion {row['identity_sha256']}",
                },
                {
                    "lineage": "unchanged",
                    "completion": f"unchanged completion {row['identity_sha256']}",
                },
            ],
        }
        for row in confirmation_rows
    ]
    confirmation_pairs_path = _write_jsonl(
        root / "confirmation_pairs.jsonl", confirmation_pairs
    )
    confirmation_pair_report = _write_json(
        root / "confirmation_pair_report.json",
        {
            "schema": "shohin-pcf1-confirmation-pair-report-v1",
            "status": "complete",
            "output": str(confirmation_pairs_path.resolve()),
            "output_sha256": sha256_file(confirmation_pairs_path),
            "rows": 1289,
            "labels_or_correctness_fields": 0,
            "source_disjoint_from_calibration": True,
            "inputs": {
                "data": str(confirmation_data.resolve()),
                "data_sha256": sha256_file(confirmation_data),
                "revision_report_sha256": sha256_file(evaluation_paths["revision"]),
                "unchanged_report_sha256": sha256_file(evaluation_paths["unchanged"]),
                "revision_candidates_sha256": sha256_file(candidate_paths["revision"]),
                "unchanged_candidates_sha256": sha256_file(
                    candidate_paths["unchanged"]
                ),
            },
            "sealed_access": dict(SEALED_ACCESS),
        },
    )

    b1_training = _write_json(
        root / "b1_training.json",
        _training_report(
            model_root=model_root,
            data_sha256="2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549",
            warm_start_sha256=None,
        ),
    )
    revision_training = _write_json(
        root / "revision_training.json",
        _training_report(
            model_root=model_root,
            data_sha256=sha256_file(revision_data),
            warm_start_sha256=checkpoint_hashes["b1"],
        ),
    )
    commit_training = _write_json(
        root / "commit_training.json",
        {
            "schema": "shohin-pcf1-commit-training-report-v1",
            "status": "complete",
            "model_root": model_root,
            "model_revision": MODEL_REVISION,
            "model_loader": "multimodal",
            "adapter_checkpoint": str(checkpoint_paths["b1"].resolve()),
            "adapter_checkpoint_sha256": checkpoint_hashes["b1"],
            "checkpoint": str(checkpoint_paths["commit"].resolve()),
            "checkpoint_sha256": checkpoint_hashes["commit"],
            "pairs": str(calibration_pairs_path.resolve()),
            "pairs_sha256": sha256_file(calibration_pairs_path),
            "updates": 128,
            "gradient_accumulation": 8,
            "head_width": 512,
            "max_sequence_length": 3072,
            "backbone_learning_rate": 2e-6,
            "head_learning_rate": 2e-4,
            "tie_loss_weight": 0.25,
            "seed": 2026080822,
            "protected_adapter_sha256_after": checkpoint_hashes["b1"],
            "protected_adapter_unchanged": True,
            "trainable_parameters": 1234,
            "trainable_parameter_name_sha256": "a" * 64,
            "lora_layer_indices": [30, 31, 32, 33],
            "environment_verified": True,
            "environment_receipt_sha256": ENVIRONMENT_SHA256,
            "environment_tree_sha256": ENVIRONMENT_TREE_SHA256,
            "sealed_access": dict(SEALED_ACCESS),
        },
    )
    selections = [
        {
            "schema": "shohin-pcf1-commit-selection-v1",
            "identity_sha256": identity,
            "task": confirmation_tasks[identity],
            "selected_index": (
                1
                if identity in correctness["unchanged"]
                and identity not in correctness["revision"]
                else 0
            ),
            "selected_lineage": (
                "unchanged"
                if identity in correctness["unchanged"]
                and identity not in correctness["revision"]
                else "revision"
            ),
            "order_consistent": True,
            "margin": 1.0,
        }
        for identity in confirmation_ids
    ]
    selections_path = _write_jsonl(root / "selections.jsonl", selections)
    application = _write_json(
        root / "application.json",
        {
            "schema": "shohin-pcf1-commit-application-report-v1",
            "status": "complete",
            "model_root": model_root,
            "model_revision": MODEL_REVISION,
            "model_loader": "multimodal",
            "adapter_checkpoint_sha256": checkpoint_hashes["b1"],
            "commit_checkpoint_sha256": checkpoint_hashes["commit"],
            "pairs_sha256": sha256_file(confirmation_pairs_path),
            "pairs_report_sha256": sha256_file(confirmation_pair_report),
            "selections": str(selections_path.resolve()),
            "selections_sha256": sha256_file(selections_path),
            "rows": 1289,
            "max_sequence_length": 3072,
            "prompt_truncated": 0,
            "malformed": 0,
            "order_consistent": 1289,
            "maximum_swap_error": 0.0,
            "correctness_or_task_label_visible": False,
            "protected_adapter_unchanged": True,
            "environment_verified": True,
            "environment_receipt_sha256": ENVIRONMENT_SHA256,
            "environment_tree_sha256": ENVIRONMENT_TREE_SHA256,
            "sealed_access": dict(SEALED_ACCESS),
        },
    )
    mechanics = _write_json(
        root / "mechanics.json",
        {
            "schema": "shohin-pcf1-mechanics-v1",
            "status": "pass",
            "capability_scored": False,
            "rows": 24,
            "task_counts": {task: 8 for task in TASKS},
            "model_root": model_root,
            "model_revision": MODEL_REVISION,
            "model_loader": "multimodal",
            "model_manifest_sha256": model_manifest_sha,
            "runtime_manifest_sha256": runtime_manifest_sha,
            "environment_receipt_sha256": ENVIRONMENT_SHA256,
            "sandbox_receipt": str(sandbox_receipt.resolve()),
            "sandbox_receipt_sha256": _sandbox_sha256(),
            "code_sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
            "code_sandbox_binary_sha256": BWRAP_SHA256,
            "code_sandbox_probe_sha256": _sandbox_sha256(),
            "code_sandbox_probe_result_sha256": _sandbox_payload()["probe_sha256"],
            "code_sandbox_runtime_tree_sha256": SANDBOX_RUNTIME_TREE_SHA256,
            "sandbox_isolation_passed": True,
            "compute_host_receipt": str(compute_host_receipt.resolve()),
            "compute_host_receipt_sha256": compute_host_sha256,
            "nvidia_smi_invoked_path": compute_host["nvidia_smi_invoked_path"],
            "nvidia_smi_resolved_path": compute_host["nvidia_smi_resolved_path"],
            "nvidia_smi_sha256": compute_host["nvidia_smi_sha256"],
            "nvidia_smi_version": compute_host["nvidia_smi_version"],
            "qualified_gpu_name": compute_host["gpu_name"],
            "qualified_driver_version": compute_host["driver_version"],
            "qualified_pci_bus_id": compute_host["pci_bus_id"],
            "qualified_node": compute_host["node"],
            "checkpoint_restored": True,
            "optimizer_updates": 1,
            "optimizer_presentations": 24,
            "lora_layers": 4,
            "lora_scope": "token_mixer",
            "lora_layer_indices": [30, 31, 32, 33],
            "lora_projection_count": 8,
            "trainable_parameters": 1234,
            "trainable_parameter_name_sha256": "a" * 64,
            "source_only_runtime_fields": ["source_prompt"],
            "task_router_used": False,
            "revision_prompt_parameters": ["source_prompt", "draft"],
            "supervisor_fields_visible_to_model": False,
            "drafts_nonempty": True,
            "revisions_nonempty": True,
            "matched_prompt_ids_identical": True,
            "commit_ab_order_checks": 24,
            "commit_ab_serialization_exact": True,
            "commit_forward_swapped_exact": True,
            "commit_prompt_truncations": 0,
            "sealed_access": dict(SEALED_ACCESS),
        },
    )

    custody_root = root / "custody"
    precompute_args = _namespace(
        run_id=RUN_ID,
        source_freeze_report=source_freeze,
        train_sources=train_sources,
        development_sources=development_sources,
        reference_preflight_rows=reference_preflight_rows,
        reference_sandbox_receipt=reference_sandbox_receipt,
        merged_drafts=drafts_path,
        merged_drafts_report=draft_report,
        revision_training_data=revision_data,
        calibration_data=calibration_data,
        confirmation_data=confirmation_data,
        confirmation_assessor_receipt=confirmation_assessor_receipt,
        data_report=data_report,
        calibration_pairs=calibration_pairs_path,
        calibration_pair_report=calibration_pair_report,
        confirmation_pairs=confirmation_pairs_path,
        confirmation_pair_report=confirmation_pair_report,
        calibration_revision_report=calibration_eval_paths["revision"],
        calibration_unchanged_report=calibration_eval_paths["unchanged"],
        calibration_revision_sandbox_probes=calibration_sandbox_probes["revision"],
        calibration_unchanged_sandbox_probes=calibration_sandbox_probes["unchanged"],
        revision_report=evaluation_paths["revision"],
        unchanged_report=evaluation_paths["unchanged"],
        self_refinement_report=evaluation_paths["self_refinement"],
        b1_checkpoint=checkpoint_paths["b1"],
        b1_training_report=b1_training,
        revision_checkpoint=checkpoint_paths["revision"],
        revision_training_report=revision_training,
        commit_checkpoint=checkpoint_paths["commit"],
        commit_training_report=commit_training,
        commit_application_report=application,
        confirmation_selections=selections_path,
        mechanics_report=mechanics,
        compute_host_receipt=compute_host_receipt,
        model_root=model_root_path,
        model_revision=MODEL_REVISION,
        model_manifest=model_manifest,
        model_manifest_sha256=model_manifest_sha,
        model_config_sha256=sha256_file(model_root_path / "config.json"),
        runtime_root=runtime_root,
        runtime_manifest=runtime_manifest,
        runtime_manifest_sha256=runtime_manifest_sha,
        environment_receipt=environment_receipt,
        environment_receipt_sha256=ENVIRONMENT_SHA256,
        sandbox_receipt=sandbox_receipt,
        sandbox_receipt_sha256=_sandbox_sha256(),
        output=custody_root,
    )
    dispatch, accounting = _accounting(root, "prescore")
    score_root = root / "score_result"
    authorize_args = _namespace(
        run_id=RUN_ID,
        confirmation_data=confirmation_data,
        confirmation_assessor_receipt=confirmation_assessor_receipt,
        revision_report=evaluation_paths["revision"],
        revision_candidates=candidate_paths["revision"],
        unchanged_report=evaluation_paths["unchanged"],
        unchanged_candidates=candidate_paths["unchanged"],
        self_refinement_report=evaluation_paths["self_refinement"],
        self_refinement_candidates=candidate_paths["self_refinement"],
        candidates_root=root,
        confirmation_pairs=confirmation_pairs_path,
        confirmation_pair_report=confirmation_pair_report,
        confirmation_selections=selections_path,
        commit_application_report=application,
        commit_training_report=commit_training,
        mechanics_report=mechanics,
        data_custody=custody_root / "data_custody.json",
        model_custody=custody_root / "model_custody.json",
        runtime_custody=custody_root / "runtime_custody.json",
        prescore_dispatch_receipt=dispatch,
        prescore_accounting_receipt=accounting,
        environment_receipt=environment_receipt,
        environment_receipt_sha256=ENVIRONMENT_SHA256,
        sandbox_receipt=sandbox_receipt,
        sandbox_receipt_sha256=_sandbox_sha256(),
        score_output_root=score_root,
        output=root / "score_authorization.json",
    )
    score_args = _namespace(
        confirmation_data=confirmation_data,
        confirmation_assessors=confirmation_assessors,
        confirmation_assessor_receipt=confirmation_assessor_receipt,
        data_report=data_report,
        revision_report=evaluation_paths["revision"],
        revision_candidates=candidate_paths["revision"],
        unchanged_report=evaluation_paths["unchanged"],
        unchanged_candidates=candidate_paths["unchanged"],
        self_refinement_report=evaluation_paths["self_refinement"],
        self_refinement_candidates=candidate_paths["self_refinement"],
        candidates_root=root,
        confirmation_pairs=confirmation_pairs_path,
        confirmation_pairs_report=confirmation_pair_report,
        selections=selections_path,
        application_report=application,
        training_report=commit_training,
        mechanics_report=mechanics,
        data_custody=custody_root / "data_custody.json",
        model_custody=custody_root / "model_custody.json",
        runtime_custody=custody_root / "runtime_custody.json",
        prescore_dispatch_receipt=dispatch,
        prescore_accounting_receipt=accounting,
        prescore_authorization=authorize_args.output,
        output_root=score_root,
        sandbox_probe_output=root / "score_result.sandbox-probe.json",
        environment_receipt=environment_receipt,
        environment_receipt_sha256=ENVIRONMENT_SHA256,
    )
    final_dispatch, final_accounting = _accounting(root, "final")
    return {
        "precompute_args": precompute_args,
        "authorize_args": authorize_args,
        "score_args": score_args,
        "correctness": correctness,
        "score_root": score_root,
        "custody_root": custody_root,
        "final_dispatch": final_dispatch,
        "final_accounting": final_accounting,
        "paths": {
            "data_report": data_report,
            "confirmation_pairs": confirmation_pairs_path,
            "confirmation_pair_report": confirmation_pair_report,
            "revision_report": evaluation_paths["revision"],
            "unchanged_report": evaluation_paths["unchanged"],
            "self_refinement_report": evaluation_paths["self_refinement"],
            "accounting": accounting,
        },
    }


def test_builds_prescore_custody_and_one_shot_authorization(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    precompute(fixture["precompute_args"])
    result = authorize_score(fixture["authorize_args"])

    assert result["schema"] == "shohin-pcf1-score-authorization-v1"
    assert result["scoring_authorized"] is True
    assert result["one_shot"] is True
    assert result["score_output_root"] == str(
        fixture["score_root"].resolve(strict=False)
    )
    assert result["assessor_board_access_count_before"] == 0
    assert "confirmation_assessors" not in result
    data = json.loads(
        (fixture["custody_root"] / "data_custody.json").read_text(encoding="utf-8")
    )
    assert data["confirmation_rows"] == 1289
    assert data["public_sealed"] is True
    assert data["public_access_count"] == 0
    model = json.loads(
        (fixture["custody_root"] / "model_custody.json").read_text(encoding="utf-8")
    )
    runtime = json.loads(
        (fixture["custody_root"] / "runtime_custody.json").read_text(encoding="utf-8")
    )
    assert model["compute_host_verified"] is True
    assert runtime["compute_host_verified"] is True
    assert (
        runtime["compute_host_receipt_sha256"] == model["compute_host_receipt_sha256"]
    )


def test_prescore_custody_and_authorization_never_open_assessor_board(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    board = fixture["score_args"].confirmation_assessors.resolve()
    board_opens = 0
    original_open = Path.open

    def tracked_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        nonlocal board_opens
        if path.resolve() == board:
            board_opens += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    precompute(fixture["precompute_args"])
    authorize_score(fixture["authorize_args"])
    assert board_opens == 0


@pytest.mark.parametrize(
    "scientific_failure",
    (None, "prompt_truncated", "malformed", "order_inconsistent"),
)
def test_score_normalize_custody_compare_integration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scientific_failure: str | None,
) -> None:
    fixture = _fixture(tmp_path)
    application_path = fixture["precompute_args"].commit_application_report
    application = json.loads(application_path.read_text(encoding="utf-8"))
    if scientific_failure == "prompt_truncated":
        application["prompt_truncated"] = 1
    elif scientific_failure == "malformed":
        application["malformed"] = 1
    elif scientific_failure == "order_inconsistent":
        selections_path = fixture["precompute_args"].confirmation_selections
        selections = [
            json.loads(line)
            for line in selections_path.read_text(encoding="utf-8").splitlines()
        ]
        selections[0]["order_consistent"] = False
        _write_jsonl(selections_path, selections)
        application["selections_sha256"] = sha256_file(selections_path)
        application["order_consistent"] = 1288
    _write_json(application_path, application)
    precompute(fixture["precompute_args"])
    authorize_score(fixture["authorize_args"])

    correctness = fixture["correctness"]

    def fake_score(assessor: dict[str, Any], completion: str) -> dict[str, bool]:
        arm = completion.split(" ", 1)[0]
        return {"correct": str(assessor["identity_sha256"]) in correctness[arm]}

    monkeypatch.setattr(scorer, "score_completion", fake_score)
    score(fixture["score_args"])
    score_report = fixture["score_root"] / "report.json"
    scored = json.loads(score_report.read_text(encoding="utf-8"))
    assert scored["assessor_board_semantic_reads"] == 1
    assert scored["confirmation_open_count"] == 1
    assert scored["outcome_rows"] == 1289

    custody = fixture["custody_root"]
    paths = fixture["paths"]
    normalized = tmp_path / "normalized"
    normalize(
        _namespace(
            learned_commit_report=score_report,
            revision_report=paths["revision_report"],
            unchanged_report=paths["unchanged_report"],
            self_refinement_report=paths["self_refinement_report"],
            data_custody=custody / "data_custody.json",
            model_custody=custody / "model_custody.json",
            runtime_custody=custody / "runtime_custody.json",
            score_consumption=scorer.score_consumption_path(fixture["score_root"]),
            output=normalized,
        )
    )
    compute_path = tmp_path / "compute_custody.json"
    compute(
        _namespace(
            run_id=RUN_ID,
            data_custody=custody / "data_custody.json",
            model_custody=custody / "model_custody.json",
            runtime_custody=custody / "runtime_custody.json",
            normalized_root=normalized,
            dispatch_receipt=fixture["final_dispatch"],
            accounting_receipt=fixture["final_accounting"],
            score_consumption=scorer.score_consumption_path(fixture["score_root"]),
            score_sandbox_probe=fixture["score_args"].sandbox_probe_output,
            environment_receipt=fixture["precompute_args"].environment_receipt,
            environment_receipt_sha256=ENVIRONMENT_SHA256,
            sandbox_receipt=fixture["precompute_args"].sandbox_receipt,
            sandbox_receipt_sha256=_sandbox_sha256(),
            output=compute_path,
        )
    )
    final = compare(
        _namespace(
            learned_commit_report=normalized / "learned_commit.json",
            trained_revision_report=normalized / "trained_revision.json",
            unchanged_report=normalized / "unchanged.json",
            self_refinement_report=normalized / "self_refinement.json",
            data_custody=custody / "data_custody.json",
            model_custody=custody / "model_custody.json",
            runtime_custody=custody / "runtime_custody.json",
            compute_custody=compute_path,
            output=tmp_path / "final.json",
        )
    )
    assert final["scores"] == {
        "learned_commit": {
            "overall": 465,
            "domains": {"math500": 135, "bbh_logic": 318, "mbpp": 12},
        },
        "trained_revision": {
            "overall": 452,
            "domains": {"math500": 130, "bbh_logic": 312, "mbpp": 10},
        },
        "unchanged": {
            "overall": 387,
            "domains": {"math500": 100, "bbh_logic": 280, "mbpp": 7},
        },
        "self_refinement": {
            "overall": 413,
            "domains": {"math500": 120, "bbh_logic": 284, "mbpp": 9},
        },
    }
    if scientific_failure is not None:
        expected_check = {
            "prompt_truncated": "learned_commit_zero_truncation",
            "malformed": "learned_commit_zero_malformed",
            "order_inconsistent": "commit_exact_ab_order_consistency",
        }[scientific_failure]
        assert final["gate_pass"] is False
        assert final["final_result"] == "FAIL"
        assert final["checks"][expected_check] is False
        assert final["checks"]["retry_count_zero"] is True
        assert final["stop_after_gate"] is True
        assert final["automatic_successor_authorized"] is False
        assert final["automatic_successor_submitted"] is False
        assert final["holdout_access_authorized"] is False
        assert not scorer.score_terminal_failure_path(fixture["score_root"]).exists()
        return
    assert final["gate_pass"] is True
    assert final["next_action"] == "stop_and_preserve_evidence"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("prompt_truncated", True),
        ("prompt_truncated", -1),
        ("prompt_truncated", 2579),
        ("malformed", True),
        ("malformed", -1),
        ("malformed", 1290),
        ("order_consistent", True),
        ("order_consistent", -1),
        ("order_consistent", 1290),
    ),
)
def test_rejects_invalid_application_observation_geometry(
    tmp_path: Path, field: str, value: Any
) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["precompute_args"].commit_application_report
    application = json.loads(path.read_text(encoding="utf-8"))
    application[field] = value
    _write_json(path, application)

    with pytest.raises(PCF1CustodyError, match="commit application lineage"):
        precompute(fixture["precompute_args"])


@pytest.mark.parametrize(
    ("field", "value"),
    (("order_consistent", 1), ("margin", True), ("margin", float("nan"))),
)
def test_rejects_invalid_selection_observation(
    tmp_path: Path, field: str, value: Any
) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["precompute_args"].confirmation_selections
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0][field] = value
    _write_jsonl(path, rows)

    with pytest.raises(PCF1CustodyError, match="confirmation selection row"):
        precompute(fixture["precompute_args"])


def test_rejects_inconsistent_selection_order_count(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    selections_path = fixture["precompute_args"].confirmation_selections
    rows = [
        json.loads(line)
        for line in selections_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["order_consistent"] = False
    _write_jsonl(selections_path, rows)
    application_path = fixture["precompute_args"].commit_application_report
    application = json.loads(application_path.read_text(encoding="utf-8"))
    application["selections_sha256"] = sha256_file(selections_path)
    _write_json(application_path, application)

    with pytest.raises(PCF1CustodyError, match="commit application lineage"):
        precompute(fixture["precompute_args"])


def test_rejects_selection_byte_tamper_even_when_semantics_remain_valid(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    selections_path = fixture["precompute_args"].confirmation_selections
    rows = [
        json.loads(line)
        for line in selections_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["margin"] = 2.0
    _write_jsonl(selections_path, rows)

    with pytest.raises(PCF1CustodyError, match="commit application lineage"):
        precompute(fixture["precompute_args"])


def test_rejects_forged_materialization_hash(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    report_path = fixture["paths"]["data_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["outputs"]["confirmation"]["sha256"] = "0" * 64
    _write_json(report_path, report)

    with pytest.raises(PCF1CustodyError, match="materialization output differs"):
        precompute(fixture["precompute_args"])


def test_rejects_forged_confirmation_order(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    pairs_path = fixture["paths"]["confirmation_pairs"]
    pairs = [json.loads(line) for line in pairs_path.read_text().splitlines()]
    pairs[0], pairs[1] = pairs[1], pairs[0]
    _write_jsonl(pairs_path, pairs)
    report_path = fixture["paths"]["confirmation_pair_report"]
    report = json.loads(report_path.read_text())
    report["output_sha256"] = sha256_file(pairs_path)
    _write_json(report_path, report)

    with pytest.raises(PCF1CustodyError, match="pair/order custody"):
        precompute(fixture["precompute_args"])


def test_rejects_forged_mbpp_reference_preflight(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    report_path = fixture["precompute_args"].source_freeze_report
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["mbpp_reference_preflight"]["all_sandbox_passed"] = False
    _write_json(report_path, report)

    with pytest.raises(PCF1CustodyError, match="reference-preflight custody"):
        precompute(fixture["precompute_args"])


@pytest.mark.parametrize("mutation", ("tamper", "drop", "reorder"))
def test_calibration_custody_rejects_setup_receipt_tamper_drop_or_reorder(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _fixture(tmp_path)
    args = fixture["precompute_args"]
    evaluation = json.loads(
        args.calibration_revision_report.read_text(encoding="utf-8")
    )
    setup_shards = evaluation["mbpp_allocation_setup_receipt_shards"]
    if mutation == "tamper":
        setup_shards[0]["receipts"][0]["setup_source_sha256"] = "0" * 64
    elif mutation == "drop":
        setup_shards.pop()
    else:
        setup_shards.reverse()
    calibration_rows = [
        json.loads(line)
        for line in args.calibration_data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    with pytest.raises(PCF1CustodyError, match="setup"):
        custody_builder._calibration_sandbox_probes(
            args.calibration_revision_sandbox_probes,
            evaluation,
            _sandbox_sha256(),
            "calibration revision",
            calibration_rows,
        )


def test_rejects_forged_frozen_settings(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    report_path = fixture["paths"]["revision_report"]
    report = json.loads(report_path.read_text())
    report["seed"] = 2026081122
    _write_json(report_path, report)
    pair_report_path = fixture["paths"]["confirmation_pair_report"]
    pair_report = json.loads(pair_report_path.read_text())
    pair_report["inputs"]["revision_report_sha256"] = sha256_file(report_path)
    _write_json(pair_report_path, pair_report)

    with pytest.raises(PCF1CustodyError, match="evaluation lineage differs"):
        precompute(fixture["precompute_args"])


def test_rejects_forged_b1_selected_row_geometry(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["precompute_args"].b1_training_report
    report = json.loads(path.read_text(encoding="utf-8"))
    report["selected_rows"] = 99_999
    _write_json(path, report)

    with pytest.raises(PCF1CustodyError, match="B1 selected-row geometry"):
        precompute(fixture["precompute_args"])


def test_rejects_forged_compute_host_receipt(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    receipt_path = fixture["precompute_args"].compute_host_receipt
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["node"] = EXCLUDED_NODES[0]
    _write_json(receipt_path, receipt)
    mechanics_path = fixture["precompute_args"].mechanics_report
    mechanics = json.loads(mechanics_path.read_text(encoding="utf-8"))
    mechanics["compute_host_receipt_sha256"] = sha256_file(receipt_path)
    mechanics["qualified_node"] = EXCLUDED_NODES[0]
    _write_json(mechanics_path, mechanics)

    with pytest.raises(PCF1CustodyError, match="compute-host receipt differs"):
        precompute(fixture["precompute_args"])


def test_rejects_forged_authorization_run(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    precompute(fixture["precompute_args"])
    accounting_path = fixture["paths"]["accounting"]
    accounting = json.loads(accounting_path.read_text())
    accounting["run_id"] = "another-pcf1-run"
    _write_json(accounting_path, accounting)

    with pytest.raises(PCF1CustodyError, match="accounting geometry differs"):
        authorize_score(fixture["authorize_args"])


@pytest.mark.parametrize(
    ("field", "value"),
    [("restarts", 1), ("allocated_gpu_types", {"nvidia_a100": 1})],
)
def test_rejects_forged_scheduler_record_geometry(
    tmp_path: Path, field: str, value: Any
) -> None:
    fixture = _fixture(tmp_path)
    precompute(fixture["precompute_args"])
    accounting_path = fixture["paths"]["accounting"]
    accounting = json.loads(accounting_path.read_text(encoding="utf-8"))
    accounting["jobs"]["mechanics"]["records"][0][field] = value
    _write_json(accounting_path, accounting)

    with pytest.raises(PCF1CustodyError, match="scheduler record differs"):
        authorize_score(fixture["authorize_args"])


def test_prescore_outputs_are_write_once(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    precompute(fixture["precompute_args"])
    before = {
        path.name: path.read_bytes() for path in fixture["custody_root"].iterdir()
    }
    with pytest.raises(PCF1CustodyError, match="refusing existing"):
        precompute(fixture["precompute_args"])
    assert {
        path.name: path.read_bytes() for path in fixture["custody_root"].iterdir()
    } == before

    authorize_score(fixture["authorize_args"])
    auth_path = fixture["authorize_args"].output
    auth_before = auth_path.read_bytes()
    with pytest.raises(PCF1CustodyError, match="refusing existing"):
        authorize_score(fixture["authorize_args"])
    assert auth_path.read_bytes() == auth_before
