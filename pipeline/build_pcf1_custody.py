#!/usr/bin/env python3
"""Compile immutable PCF1 data, model, runtime, and compute custody."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from typing import Any, Iterable, Mapping

from build_pcf1_data import (
    ASSESSOR_SCHEMA,
    DEVELOPMENT_SOURCE_SCHEMA,
    TRAIN_SOURCE_SCHEMA,
    revision_prompt,
)
from pcf1_environment import validate_environment_receipt
from pcf1_code_sandbox import (
    CANDIDATE_FAILURE_EXIT_CODE,
    CANDIDATE_POLICY_SHA256,
    CANDIDATE_RANDOM_SEED,
    ELF_CLOSURE_AUDIT_SHA256,
    EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR,
    EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR_SHA256,
    MEMFD_ABI,
    POLICY_REJECTION_EXIT_CODE,
    RESOURCE_LIMIT_EXIT_CODE,
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
    TEST_FAILURE_EXIT_CODE,
    TRUSTED_COMPLETION_EXIT_CODE,
    PCF1SandboxError,
    mbpp_allocation_setup_receipts_sha256,
    validate_mbpp_setup_qualification_receipt,
    validate_sandbox_receipt_payload,
)

DATA_CUSTODY_SCHEMA = "shohin-pcf1-data-custody-v1"
MODEL_CUSTODY_SCHEMA = "shohin-pcf1-model-custody-v1"
RUNTIME_CUSTODY_SCHEMA = "shohin-pcf1-runtime-custody-v1"
COMPUTE_CUSTODY_SCHEMA = "shohin-pcf1-compute-custody-v1"
FREEZE_REPORT_SCHEMA = "shohin-pcf1-data-freeze-report-v1"
DATA_REPORT_SCHEMA = "shohin-pcf1-data-report-v1"
MERGED_DRAFT_REPORT_SCHEMA = "shohin-pcf1-merged-drafts-v1"
DRAFT_SCHEMA = "shohin-pcf1-model-draft-v1"
REVISION_TRAIN_SCHEMA = "shohin-pcf1-revision-train-v1"
EVAL_SCHEMA = "shohin-pcf1-eval-v1"
CALIBRATION_PAIR_SCHEMA = "shohin-pcf1-whole-trajectory-pair-v1"
CALIBRATION_PAIR_REPORT_SCHEMA = "shohin-pcf1-commit-pair-report-v1"
CONFIRMATION_PAIR_SCHEMA = "shohin-pcf1-confirmation-pair-v1"
CONFIRMATION_PAIR_REPORT_SCHEMA = "shohin-pcf1-confirmation-pair-report-v1"
TRAINING_REPORT_SCHEMA = "shohin-hf-product-reasoning-training-v1"
MERGED_EVALUATION_SCHEMA = "shohin-pcf1-merged-evaluation-v1"
COMMIT_TRAINING_SCHEMA = "shohin-pcf1-commit-training-report-v1"
COMMIT_APPLICATION_SCHEMA = "shohin-pcf1-commit-application-report-v1"
COMMIT_RESULT_SCHEMA = "shohin-pcf1-commit-result-v1"
SELECTION_SCHEMA = "shohin-pcf1-commit-selection-v1"
MECHANICS_SCHEMA = "shohin-pcf1-mechanics-v1"
ARM_REPORT_SCHEMA = "shohin-pcf1-arm-report-v1"
DISPATCH_SCHEMA = "shohin-pcf1-dispatch-v1"
ACCOUNTING_SCHEMA = "shohin-pcf1-slurm-accounting-v1"
SCORE_AUTHORIZATION_SCHEMA = "shohin-pcf1-score-authorization-v1"
SCORE_CONSUMPTION_SCHEMA = "shohin-pcf1-score-consumption-v1"
CONFIRMATION_ASSESSOR_RECEIPT_SCHEMA = "shohin-pcf1-confirmation-assessor-receipt-v1"
CANDIDATE_SCHEMA = "shohin-pcf1-candidate-v1"
SANDBOX_RECEIPT_SCHEMA = "shohin-pcf1-code-sandbox-receipt-v1"
COMPUTE_HOST_RECEIPT_SCHEMA = "shohin-pcf1-compute-host-receipt-v1"
SANDBOX_BINARY_SHA256 = (
    "eb767688b8224d8d3dbe1f8cb30ac3dff9ae8b02ff0452eaec9f94874d4e0011"
)
MODEL_REVISION = "81eaece1948f3875421d9a45bc55487d10e2d894"
MODEL_LOADER = "multimodal"
B1_DATA_SHA256 = "2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549"
SOURCE_PAIR_SHA256 = "45f1d66ce5e87dc2a1f4c3594bdde2bae26e9417e879d16eb4eddb228b696afe"
SOURCE_BANK_SHA256S = (
    "0b6d068b4d71f407cb234579b9278dc640df09139ea906dd0f52a6ab71e05398",
    "5a96859fd9088cde598b61da60dd2c6cb7281323ee06c034742a1b4e0e237017",
    "e0ede83257e441050a019f59fb13d9c85bd6cba1d6a755ab86fb7129966ddbe5",
)
SPLIT_SEED = 2026080811
CALIBRATION_SEED = 2026080820
DRAFT_SEED = 2026080818
EVALUATION_SEED = 2026080816
TOTAL_TRAIN = 5824
TOTAL_CONFIRMATION = 1289
TOTAL_SEALED = 1279
TOTAL_DRAFTS = TOTAL_TRAIN + TOTAL_CONFIRMATION
REVISION_PRESENTATIONS = 9655
TASKS = ("math500", "bbh_logic", "mbpp")
SEALED_ACCESS = {"holdout": 0, "product": 0, "public": 0}
FORBIDDEN_PATH_TERMS = ("holdout", "product", "public")
NATIVE_ARMS = ("revision", "unchanged", "self_refinement")
NORMALIZED_ARMS = (
    "learned_commit",
    "trained_revision",
    "unchanged",
    "self_refinement",
)
EXCLUDED_NODES = [
    "evc26",
    "evc29",
    "evc31",
    "evc32",
    "evc33",
    "evc37",
    "evc38",
    "evc46",
]
PRESCORE_ACCOUNTING_STAGES = (
    "prepare_inputs",
    "mechanics",
    "b1_train",
    "draft_generate",
    "draft_merge",
    "materialize",
    "revision_train",
    "calibration_revision_eval",
    "calibration_revision_merge",
    "calibration_unchanged_eval",
    "calibration_unchanged_merge",
    "calibration_pairs",
    "commit_train",
    "confirmation_revision_eval",
    "confirmation_revision_merge",
    "confirmation_unchanged_eval",
    "confirmation_unchanged_merge",
    "confirmation_self_refinement_eval",
    "confirmation_self_refinement_merge",
    "confirmation_pairs",
    "commit_apply",
    "precompute_custody",
)
FINAL_ACCOUNTING_STAGES = PRESCORE_ACCOUNTING_STAGES + (
    "prescore_accounting",
    "authorize_score",
    "commit_score",
    "normalize",
)
ARRAY_TASKS = {
    "draft_generate": 16,
    "calibration_revision_eval": 4,
    "calibration_unchanged_eval": 4,
    "confirmation_revision_eval": 4,
    "confirmation_unchanged_eval": 4,
    "confirmation_self_refinement_eval": 4,
}
GPU_STAGES = {
    "mechanics",
    "b1_train",
    "draft_generate",
    "revision_train",
    "calibration_revision_eval",
    "calibration_unchanged_eval",
    "commit_train",
    "confirmation_revision_eval",
    "confirmation_unchanged_eval",
    "confirmation_self_refinement_eval",
    "commit_apply",
}
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
MANIFEST_LINE = re.compile(r"([0-9a-f]{64}) ([ *])(.+)\Z")


class PCF1CustodyError(RuntimeError):
    """Explicit PCF1 evidence cannot support the requested custody receipt."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: object, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _run_id(value: object) -> str:
    if not isinstance(value, str) or RUN_ID.fullmatch(value) is None:
        raise PCF1CustodyError("PCF1 run_id is invalid")
    return value


def _safe_path(path: Path, label: str, *, resolve: bool = False) -> Path:
    expanded = path.expanduser()
    absolute = Path(os.path.abspath(expanded))
    candidates = [absolute]
    if resolve:
        try:
            candidates.append(absolute.resolve(strict=True))
        except (OSError, RuntimeError) as error:
            raise PCF1CustodyError(f"unresolvable explicit {label}") from error
    for candidate in candidates:
        lowered = str(candidate).casefold()
        term = next((term for term in FORBIDDEN_PATH_TERMS if term in lowered), None)
        if term is not None:
            raise PCF1CustodyError(f"refusing protected {label} path containing {term}")
    return candidates[-1] if resolve else absolute


def _explicit_file(path: Path, label: str) -> Path:
    absolute = _safe_path(path, label)
    if absolute.is_symlink() or not absolute.is_file():
        raise PCF1CustodyError(f"explicit {label} is missing or symbolic")
    _safe_path(absolute, label, resolve=True)
    return absolute


def _explicit_root(path: Path, label: str) -> Path:
    absolute = _safe_path(path, label)
    if absolute.is_symlink() or not absolute.is_dir():
        raise PCF1CustodyError(f"explicit {label} is missing or symbolic")
    return _safe_path(absolute, label, resolve=True)


def _explicit_file_under_root(path: Path, root: Path, label: str) -> Path:
    explicit = _explicit_file(path, label)
    try:
        relative = explicit.relative_to(root)
    except ValueError as error:
        raise PCF1CustodyError(f"explicit {label} escapes its safe root") from error
    if not relative.parts:
        raise PCF1CustodyError(f"explicit {label} equals its safe root")
    current = root
    for part in relative.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise PCF1CustodyError(f"explicit {label} traverses a symbolic directory")
    return explicit


def _recorded_path_matches(value: object, explicit: Path, label: str) -> bool:
    if not isinstance(value, str) or not value:
        raise PCF1CustodyError(f"{label} recorded path is invalid")
    recorded = _safe_path(Path(value), f"recorded {label}")
    return recorded == _safe_path(explicit, label)


def _load_report(
    path: Path, *, schema: str, label: str, status: str = "complete"
) -> tuple[dict[str, Any], str]:
    explicit = _explicit_file(path, label)
    raw = explicit.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PCF1CustodyError(f"invalid {label} JSON") from error
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get("status") != status
    ):
        raise PCF1CustodyError(f"{label} schema or status differs")
    return value, hashlib.sha256(raw).hexdigest()


def _load_jsonl(path: Path, label: str) -> tuple[list[dict[str, Any]], str]:
    explicit = _explicit_file(path, label)
    raw = explicit.read_bytes()
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(raw.decode().splitlines(), start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise PCF1CustodyError(f"non-object {label} row {line_number}")
            rows.append(row)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PCF1CustodyError(f"invalid {label} JSONL") from error
    if not rows:
        raise PCF1CustodyError(f"empty {label}")
    return rows, hashlib.sha256(raw).hexdigest()


def _ordered_identity_sha256(identities: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(identities) + "\n").encode()).hexdigest()


def _require_access_zero(report: Mapping[str, Any], label: str) -> None:
    if report.get("sealed_access") != SEALED_ACCESS:
        raise PCF1CustodyError(f"{label} sealed-access receipt differs")


def _external_runtime_receipts(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    """Validate the allocated environment and qualified code-sandbox receipts."""

    try:
        environment = validate_environment_receipt(
            args.environment_receipt,
            args.environment_receipt_sha256,
            "pipeline/build_pcf1_custody.py",
        )
    except RuntimeError as error:
        raise PCF1CustodyError("PCF1 environment receipt differs") from error
    sandbox, sandbox_sha = _load_report(
        args.sandbox_receipt,
        schema=SANDBOX_RECEIPT_SCHEMA,
        status="pass",
        label="code sandbox receipt",
    )
    try:
        validate_sandbox_receipt_payload(sandbox)
    except PCF1SandboxError as error:
        raise PCF1CustodyError("PCF1 qualified code-sandbox receipt differs") from error
    probe_results = sandbox.get("probe_results")
    expected_system_members = [
        {
            "source": str(source),
            "destination": destination,
            "sha256": digest,
            "size": size,
        }
        for source, destination, digest, size in SYSTEM_LIBRARY_BINDINGS
    ]
    if (
        sandbox_sha != args.sandbox_receipt_sha256
        or sandbox.get("bwrap_path") != "/usr/bin/bwrap"
        or sandbox.get("bwrap_sha256") != SANDBOX_BINARY_SHA256
        or sandbox.get("bwrap_version") != "bubblewrap 0.4.0"
        or sandbox.get("sandbox_config_sha256") != SANDBOX_CONFIG_SHA256
        or sandbox.get("candidate_policy_sha256") != CANDIDATE_POLICY_SHA256
        or sandbox.get("trusted_completion_exit_code") != TRUSTED_COMPLETION_EXIT_CODE
        or sandbox.get("candidate_failure_exit_code") != CANDIDATE_FAILURE_EXIT_CODE
        or sandbox.get("infrastructure_failure_exit_code")
        != INFRASTRUCTURE_FAILURE_EXIT_CODE
        or sandbox.get("test_failure_exit_code") != TEST_FAILURE_EXIT_CODE
        or sandbox.get("setup_failure_exit_code") != SETUP_FAILURE_EXIT_CODE
        or sandbox.get("policy_rejection_exit_code") != POLICY_REJECTION_EXIT_CODE
        or sandbox.get("resource_limit_exit_code") != RESOURCE_LIMIT_EXIT_CODE
        or sandbox.get("candidate_random_seed") != CANDIDATE_RANDOM_SEED
        or sandbox.get("python_runtime_descriptor")
        != EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR
        or sandbox.get("python_runtime_descriptor_sha256")
        != EXPECTED_SANDBOX_RUNTIME_DESCRIPTOR_SHA256
        or sandbox.get("memfd_abi") != MEMFD_ABI
        or sandbox.get("sandbox_runtime_tree_sha256") != SANDBOX_RUNTIME_TREE_SHA256
        or sandbox.get("sandbox_runtime_tree_entries") != SANDBOX_RUNTIME_TREE_ENTRIES
        or sandbox.get("sandbox_runtime_tree_files") != SANDBOX_RUNTIME_TREE_FILES
        or sandbox.get("sandbox_runtime_tree_directories")
        != SANDBOX_RUNTIME_TREE_DIRECTORIES
        or sandbox.get("sandbox_runtime_tree_bytes") != SANDBOX_RUNTIME_TREE_BYTES
        or sandbox.get("elf_closure_audit_sha256") != ELF_CLOSURE_AUDIT_SHA256
        or sandbox.get("system_library_members") != expected_system_members
        or sandbox.get("clear_environment") is not True
        or sandbox.get("network_namespace") != "isolated"
        or sandbox.get("candidate_read_only") is not True
        or sandbox.get("candidate_direct_pid_1") is not True
        or sandbox.get("site_packages_visible") is not False
        or sandbox.get("sandbox_isolation_passed") is not True
        or not _sha256(sandbox.get("probe_sha256"))
        or not isinstance(probe_results, dict)
        or set(probe_results) != SANDBOX_PROBES
        or any(value is not True for value in probe_results.values())
        or sandbox.get("probe_sha256")
        != hashlib.sha256(
            json.dumps(probe_results, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    ):
        raise PCF1CustodyError("PCF1 qualified code-sandbox receipt differs")
    return environment, args.environment_receipt_sha256, sandbox, sandbox_sha


def _exact_manifest(
    *, root: Path, manifest_path: Path, expected_sha256: str, label: str
) -> dict[str, Any]:
    root = _explicit_root(root, f"{label} root")
    manifest = _explicit_file(manifest_path, f"{label} manifest")
    if not _sha256(expected_sha256) or sha256_file(manifest) != expected_sha256:
        raise PCF1CustodyError(f"{label} manifest byte hash differs")
    entries: dict[str, str] = {}
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise PCF1CustodyError(f"{label} manifest is not UTF-8") from error
    for line in lines:
        match = MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise PCF1CustodyError(f"{label} manifest line differs")
        digest, _, rendered = match.groups()
        relative = PurePosixPath(rendered)
        if (
            relative.is_absolute()
            or not rendered
            or rendered != relative.as_posix()
            or any(part in ("", ".", "..") for part in relative.parts)
            or any(term in rendered.casefold() for term in FORBIDDEN_PATH_TERMS)
            or rendered in entries
        ):
            raise PCF1CustodyError(f"{label} manifest entry is unsafe")
        entries[rendered] = digest
    if not entries or list(entries) != sorted(entries):
        raise PCF1CustodyError(f"{label} manifest is empty or unordered")

    excluded: Path | None = None
    try:
        excluded = manifest.relative_to(root)
    except ValueError:
        pass
    observed: dict[str, str] = {}
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directory_names:
            candidate = directory_path / name
            _safe_path(candidate, f"{label} directory")
            if candidate.is_symlink():
                raise PCF1CustodyError(f"{label} tree contains a directory symlink")
        for name in file_names:
            candidate = directory_path / name
            relative = candidate.relative_to(root)
            if excluded is not None and relative == excluded:
                continue
            rendered = relative.as_posix()
            _safe_path(candidate, f"{label} member")
            if candidate.is_symlink():
                raise PCF1CustodyError(f"{label} tree contains a file symlink")
            mode = candidate.stat(follow_symlinks=False).st_mode
            if not stat.S_ISREG(mode):
                raise PCF1CustodyError(f"{label} tree has a non-regular file")
            observed[rendered] = sha256_file(candidate)
    if entries != dict(sorted(observed.items())):
        raise PCF1CustodyError(f"{label} manifest/tree entries differ")
    canonical = "".join(f"{digest}  {path}\n" for path, digest in entries.items())
    return {
        "manifest_sha256": expected_sha256,
        "tree_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "file_count": len(entries),
    }


def _freeze_report(report: Mapping[str, Any]) -> None:
    expected_counts = {
        "train": TOTAL_TRAIN,
        "development": TOTAL_CONFIRMATION,
        "holdout": TOTAL_SEALED,
    }
    if (
        report.get("split_seed") != SPLIT_SEED
        or report.get("counts") != expected_counts
        or report.get("inputs")
        != {
            "pairs_sha256": SOURCE_PAIR_SHA256,
            "source_bank_sha256s": list(SOURCE_BANK_SHA256S),
        }
        or report.get("draft_training_reference")
        != {
            "corpus_sha256": B1_DATA_SHA256,
            "content_copied": False,
            "path_recorded": False,
            "hash_reference_only": True,
        }
        or report.get("revision_training_geometry")
        != {
            "unique_train_identities": TOTAL_TRAIN,
            "presentations": REVISION_PRESENTATIONS,
            "single_correct_presentations_per_identity": 4,
            "other_presentations_per_identity": 1,
        }
        or report.get("source_disjoint") is not True
        or report.get("sealed_content_materialized") is not False
        or report.get("protected_board_inputs") != 0
        or report.get("public_inputs") != 0
    ):
        raise PCF1CustodyError("PCF1 source-freeze custody differs")
    receipts = report.get("identity_receipts")
    if not isinstance(receipts, dict) or any(
        receipts.get(split, {}).get("count") != count
        or not _sha256(receipts.get(split, {}).get("ordered_identity_sha256"))
        for split, count in expected_counts.items()
    ):
        raise PCF1CustodyError("PCF1 source-freeze identity receipt differs")


def _source_views(
    report: Mapping[str, Any], args: argparse.Namespace
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str, str]:
    """Replay the exact safe source views independently of materialized rows."""

    specifications = (
        (
            "train",
            args.train_sources,
            TRAIN_SOURCE_SCHEMA,
            TOTAL_TRAIN,
            {
                "schema",
                "identity_sha256",
                "split",
                "task",
                "outcome_class",
                "source_prompt",
                "response",
                "target_kind",
                "assessor",
                "runtime_fields",
                "supervisor_only_fields",
            },
        ),
        (
            "development",
            args.development_sources,
            DEVELOPMENT_SOURCE_SCHEMA,
            TOTAL_CONFIRMATION,
            {
                "schema",
                "identity_sha256",
                "split",
                "task",
                "source_prompt",
                "runtime_fields",
                "supervisor_only_fields",
            },
        ),
    )
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    digests: dict[str, str] = {}
    outputs = report.get("outputs")
    identities = report.get("identity_receipts")
    if not isinstance(outputs, dict) or not isinstance(identities, dict):
        raise PCF1CustodyError("PCF1 source-view receipts are absent")
    for split, path, schema, expected_rows, allowed in specifications:
        rows, digest = _load_jsonl(path, f"{split} source view")
        by_identity: dict[str, dict[str, Any]] = {}
        for row in rows:
            identity = row.get("identity_sha256")
            if (
                set(row) != allowed
                or row.get("schema") != schema
                or not _sha256(identity)
                or identity in by_identity
                or row.get("split") != split
                or row.get("task") not in TASKS
                or not isinstance(row.get("source_prompt"), str)
                or not row["source_prompt"].strip()
                or row.get("runtime_fields") != ["source_prompt"]
            ):
                raise PCF1CustodyError(f"PCF1 {split} source-view row differs")
            if split == "train":
                assessor = row.get("assessor")
                if (
                    row.get("outcome_class")
                    not in ("base_only", "both_correct", "both_wrong", "expert_only")
                    or not isinstance(row.get("response"), str)
                    or not row["response"].strip()
                    or not isinstance(row.get("target_kind"), str)
                    or not row["target_kind"]
                    or not isinstance(assessor, dict)
                    or assessor.get("schema") != ASSESSOR_SCHEMA
                    or assessor.get("identity_sha256") != identity
                    or assessor.get("task") != row.get("task")
                    or row.get("supervisor_only_fields")
                    != ["response", "target_kind", "assessor", "task", "outcome_class"]
                ):
                    raise PCF1CustodyError("PCF1 train source supervision differs")
            elif row.get("supervisor_only_fields") != ["task"]:
                raise PCF1CustodyError("PCF1 development source firewall differs")
            by_identity[str(identity)] = row
        ordered = list(by_identity)
        receipt = outputs.get(f"{split}_sources.jsonl")
        identity_receipt = identities.get(split)
        if (
            len(rows) != expected_rows
            or ordered != sorted(ordered)
            or not isinstance(receipt, dict)
            or receipt != {"sha256": digest, "rows": expected_rows}
            or identity_receipt
            != {
                "count": expected_rows,
                "ordered_identity_sha256": _ordered_identity_sha256(ordered),
            }
        ):
            raise PCF1CustodyError(f"PCF1 {split} source-view binding differs")
        loaded[split] = by_identity
        digests[split] = digest
    if set(loaded["train"]) & set(loaded["development"]):
        raise PCF1CustodyError("PCF1 safe source views overlap")
    return (
        loaded["train"],
        loaded["development"],
        digests["train"],
        digests["development"],
    )


def _reference_preflight(
    freeze: Mapping[str, Any],
    args: argparse.Namespace,
    train_sources: Mapping[str, Mapping[str, Any]],
    development_sources: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str, Mapping[str, Any]]:
    """Bind trusted isolated execution of every nonsealed MBPP reference."""

    receipt, receipt_sha = _load_report(
        args.reference_sandbox_receipt,
        schema=SANDBOX_RECEIPT_SCHEMA,
        status="pass",
        label="reference sandbox receipt",
    )
    try:
        validate_sandbox_receipt_payload(receipt)
    except PCF1SandboxError as error:
        raise PCF1CustodyError("PCF1 reference sandbox receipt differs") from error
    output = freeze.get("outputs", {}).get("reference_sandbox_receipt.json")
    rows_output = freeze.get("outputs", {}).get("mbpp_reference_preflight.jsonl")
    preflight = freeze.get("mbpp_reference_preflight")
    source_identities = sorted(
        identity
        for sources in (train_sources, development_sources)
        for identity, row in sources.items()
        if row.get("task") == "mbpp"
    )
    rows, rows_sha = _load_jsonl(
        args.reference_preflight_rows, "MBPP reference-preflight rows"
    )
    identities: list[str] = []
    setup_qualifications: set[tuple[str, str]] = set()
    sources = {**train_sources, **development_sources}
    for row in rows:
        identity = str(row.get("identity_sha256"))
        source = sources.get(identity)
        if (
            set(row)
            != {
                "identity_sha256",
                "split",
                "candidate_source_sha256",
                "program_sha256",
                "setup_source_sha256",
                "setup_qualification_sha256",
                "candidate_policy_sha256",
                "sandbox_config_sha256",
                "allocation_probe_sha256",
                "reference_assessment_mode",
                "generated_candidate_policy_applied",
                "termination_classification",
            }
            or source is None
            or source.get("task") != "mbpp"
            or row.get("split") != source.get("split")
            or any(
                not _sha256(row.get(field))
                for field in (
                    "identity_sha256",
                    "candidate_source_sha256",
                    "program_sha256",
                    "setup_source_sha256",
                    "setup_qualification_sha256",
                )
            )
            or row.get("candidate_policy_sha256") != CANDIDATE_POLICY_SHA256
            or row.get("sandbox_config_sha256") != SANDBOX_CONFIG_SHA256
            or row.get("allocation_probe_sha256") != receipt.get("probe_sha256")
            or row.get("termination_classification") != "trusted_tests_completed"
            or row.get("reference_assessment_mode") != "trusted_reference"
            or row.get("generated_candidate_policy_applied") is not False
        ):
            raise PCF1CustodyError("PCF1 MBPP reference-preflight row differs")
        identities.append(identity)
        setup_qualifications.add(
            (
                str(row["setup_source_sha256"]),
                str(row["setup_qualification_sha256"]),
            )
        )
    setup_qualification_lines = b"".join(
        (
            json.dumps(
                {
                    "setup_source_sha256": setup_sha256,
                    "setup_qualification_sha256": qualification_sha256,
                },
                sort_keys=True,
            )
            + "\n"
        ).encode()
        for setup_sha256, qualification_sha256 in sorted(setup_qualifications)
    )
    if (
        output != {"sha256": receipt_sha, "rows": 1}
        or rows_output != {"sha256": rows_sha, "rows": len(source_identities)}
        or identities != source_identities
        or len(set(identities)) != len(identities)
        or not isinstance(preflight, dict)
        or set(preflight)
        != {
            "schema",
            "status",
            "scope",
            "rows",
            "ordered_identity_sha256",
            "row_receipts_sha256",
            "unique_setups",
            "setup_pair_receipts_sha256",
            "candidate_policy_sha256",
            "sandbox_config_sha256",
            "allocation_probe_sha256",
            "reference_assessment_mode",
            "generated_candidate_policy_applied",
            "all_references_passed",
            "all_sandbox_passed",
            "holdout_reference_content_accesses",
            "sandbox_receipt_sha256",
        }
        or preflight.get("schema") != "shohin-pcf1-mbpp-reference-preflight-v1"
        or preflight.get("status") != "pass"
        or preflight.get("scope") != ["train", "development"]
        or preflight.get("rows") != len(source_identities)
        or preflight.get("ordered_identity_sha256")
        != _ordered_identity_sha256(identities)
        or not _sha256(preflight.get("row_receipts_sha256"))
        or preflight.get("row_receipts_sha256") != rows_sha
        or preflight.get("unique_setups") != len(setup_qualifications)
        or preflight.get("setup_pair_receipts_sha256")
        != hashlib.sha256(setup_qualification_lines).hexdigest()
        or preflight.get("candidate_policy_sha256") != CANDIDATE_POLICY_SHA256
        or preflight.get("sandbox_config_sha256") != SANDBOX_CONFIG_SHA256
        or preflight.get("allocation_probe_sha256") != receipt.get("probe_sha256")
        or preflight.get("sandbox_receipt_sha256") != receipt_sha
        or preflight.get("reference_assessment_mode") != "trusted_reference"
        or preflight.get("generated_candidate_policy_applied") is not False
        or preflight.get("all_references_passed") is not True
        or preflight.get("all_sandbox_passed") is not True
        or preflight.get("holdout_reference_content_accesses") != 0
    ):
        raise PCF1CustodyError("PCF1 MBPP reference-preflight custody differs")
    return receipt_sha, rows_sha, preflight


def _draft_rows(
    rows: list[dict[str, Any]],
    report: Mapping[str, Any],
    digest: str,
    args: argparse.Namespace,
    sources: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], str]:
    if len(rows) != TOTAL_DRAFTS:
        raise PCF1CustodyError("PCF1 merged draft cardinality differs")
    by_identity: dict[str, dict[str, Any]] = {}
    split_counts: Counter[str] = Counter()
    adapters: set[str] = set()
    for row in rows:
        identity = row.get("identity_sha256")
        source = sources.get(str(identity))
        split = row.get("split")
        generated = row.get("generated_tokens")
        exhausted = row.get("max_token_exhausted")
        if (
            set(row)
            != {
                "schema",
                "identity_sha256",
                "split",
                "task",
                "completion",
                "generated_tokens",
                "max_token_exhausted",
                "prompt_sha256",
                "adapter_checkpoint_sha256",
                "model_revision",
                "finish_reason",
                "wall_seconds",
            }
            or row.get("schema") != DRAFT_SCHEMA
            or not _sha256(identity)
            or identity in by_identity
            or source is None
            or split not in ("train", "development")
            or split != source.get("split")
            or row.get("task") not in TASKS
            or row.get("task") != source.get("task")
            or not isinstance(row.get("completion"), str)
            or not row["completion"].strip()
            or row.get("prompt_sha256")
            != hashlib.sha256(str(source.get("source_prompt")).encode()).hexdigest()
            or not _sha256(row.get("adapter_checkpoint_sha256"))
            or row.get("model_revision") != MODEL_REVISION
            or isinstance(generated, bool)
            or not isinstance(generated, int)
            or generated <= 0
            or not isinstance(exhausted, bool)
            or row.get("finish_reason") != ("length" if exhausted else "stop")
        ):
            raise PCF1CustodyError("PCF1 merged draft row custody differs")
        by_identity[identity] = row
        split_counts[split] += 1
        adapters.add(row["adapter_checkpoint_sha256"])
    if split_counts != {"train": TOTAL_TRAIN, "development": TOTAL_CONFIRMATION}:
        raise PCF1CustodyError("PCF1 merged draft split geometry differs")
    if len(adapters) != 1:
        raise PCF1CustodyError("PCF1 merged draft adapter lineage differs")
    if set(by_identity) != set(sources):
        raise PCF1CustodyError("PCF1 merged draft/source-view coverage differs")
    if (
        report.get("model_revision") != MODEL_REVISION
        or report.get("model_loader") != MODEL_LOADER
        or report.get("adapter_checkpoint_sha256") != next(iter(adapters))
        or report.get("environment_receipt_sha256") != args.environment_receipt_sha256
        or report.get("source_report_sha256") != sha256_file(args.source_freeze_report)
        or report.get("source_counts")
        != {"train": TOTAL_TRAIN, "development": TOTAL_CONFIRMATION}
        or report.get("generation_mode") != "greedy"
        or report.get("thinking_enabled") is not False
        or report.get("max_new_tokens") != 768
        or report.get("seed") != DRAFT_SEED
        or report.get("full_row_count") != TOTAL_DRAFTS
        or report.get("rows") != TOTAL_DRAFTS
        or report.get("exact_identity_coverage") is not True
        or not _recorded_path_matches(
            report.get("output"), args.merged_drafts, "merged drafts"
        )
        or report.get("output_sha256") != digest
    ):
        raise PCF1CustodyError("PCF1 merged draft report binding differs")
    _require_access_zero(report, "merged draft report")
    return by_identity, next(iter(adapters))


def _materialized_rows(
    *,
    revision_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    confirmation_rows: list[dict[str, Any]],
    drafts: Mapping[str, dict[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    draft_splits = {
        split: {
            identity for identity, row in drafts.items() if row.get("split") == split
        }
        for split in ("train", "development")
    }
    revision_counts: Counter[str] = Counter()
    revision_outcomes: dict[str, set[str]] = {}
    revision_order: list[tuple[str, int]] = []
    for row in revision_rows:
        source_identity = row.get("source_identity_sha256")
        draft = drafts.get(str(source_identity))
        source = sources.get(str(source_identity))
        outcome = row.get("outcome_class")
        presentation = row.get("presentation")
        if (
            row.get("schema") != REVISION_TRAIN_SCHEMA
            or draft is None
            or source is None
            or draft.get("split") != "train"
            or row.get("task") != draft.get("task")
            or row.get("task") != source.get("task")
            or row.get("outcome_class") != source.get("outcome_class")
            or row.get("response") != source.get("response")
            or row.get("target_kind") != source.get("target_kind")
            or outcome not in ("base_only", "both_correct", "both_wrong", "expert_only")
            or isinstance(presentation, bool)
            or not isinstance(presentation, int)
            or presentation < 0
            or row.get("model_owned_draft_sha256")
            != hashlib.sha256(str(draft["completion"]).strip().encode()).hexdigest()
            or row.get("question")
            != revision_prompt(
                str(source.get("source_prompt")), str(draft["completion"]).strip()
            )
            or row.get("runtime_fields") != ["question"]
        ):
            raise PCF1CustodyError("PCF1 revision training row custody differs")
        revision_counts[str(source_identity)] += 1
        revision_outcomes.setdefault(str(source_identity), set()).add(str(outcome))
        revision_order.append((str(source_identity), presentation))
    if (
        len(revision_rows) != REVISION_PRESENTATIONS
        or set(revision_counts) != draft_splits["train"]
        or revision_order != sorted(revision_order)
        or any(len(outcomes) != 1 for outcomes in revision_outcomes.values())
        or any(
            count
            != (
                4
                if next(iter(revision_outcomes[identity]))
                in ("base_only", "expert_only")
                else 1
            )
            for identity, count in revision_counts.items()
        )
    ):
        raise PCF1CustodyError("PCF1 revision presentation geometry differs")

    def evaluation_identities(
        rows: list[dict[str, Any]], expected_split: str, source_split: str
    ) -> list[str]:
        identities: list[str] = []
        for row in rows:
            identity = row.get("identity_sha256")
            draft = drafts.get(str(identity))
            source = sources.get(str(identity))
            expected_internal = (
                {**draft, "completion": str(draft["completion"]).strip()}
                if draft is not None
                else None
            )
            if (
                row.get("schema") != EVAL_SCHEMA
                or row.get("split") != expected_split
                or draft is None
                or source is None
                or draft.get("split") != source_split
                or row.get("task") != draft.get("task")
                or row.get("task") != source.get("task")
                or row.get("source_prompt") != source.get("source_prompt")
                or row.get("question")
                != revision_prompt(
                    str(source.get("source_prompt")), str(draft["completion"]).strip()
                )
                or row.get("internal_draft") != expected_internal
                or row.get("runtime_fields")
                != (
                    ["question"]
                    if expected_split == "calibration"
                    else ["question", "source_prompt"]
                )
                or row.get("internal_draft_visible") is not True
                or row.get("external_candidate_text_visible") is not False
            ):
                raise PCF1CustodyError("PCF1 materialized evaluation row differs")
            identities.append(str(identity))
        if identities != sorted(identities) or len(set(identities)) != len(identities):
            raise PCF1CustodyError("PCF1 materialized identity order differs")
        return identities

    calibration = evaluation_identities(calibration_rows, "calibration", "train")
    confirmation = evaluation_identities(
        confirmation_rows, "confirmation", "development"
    )
    if (
        set(calibration) != draft_splits["train"]
        or set(confirmation) != draft_splits["development"]
    ):
        raise PCF1CustodyError("PCF1 materialized identity coverage differs")
    return calibration, confirmation


def _materialization_report(
    report: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    receipts: Mapping[str, tuple[str, int]],
    calibration_ids: list[str],
    confirmation_ids: list[str],
    assessor_sha256: str,
    assessor_receipt_sha256: str,
) -> None:
    if (
        report.get("split_seed") != SPLIT_SEED
        or report.get("inputs")
        != {
            "freeze_report_sha256": sha256_file(args.source_freeze_report),
            "drafts_sha256": sha256_file(args.merged_drafts),
            "draft_rows": TOTAL_DRAFTS,
        }
        or report.get("counts")
        != {
            "train_unique_identities": TOTAL_TRAIN,
            "revision_train_presentations": REVISION_PRESENTATIONS,
            "calibration_rows": TOTAL_TRAIN,
            "confirmation_rows": TOTAL_CONFIRMATION,
        }
        or report.get("revision_presentation_rule")
        != {"single_correct": 4, "both_correct_or_both_wrong": 1}
        or report.get("confirmation_assessor_access")
        != {"semantic_reads": 0, "authorized_reader": "score_pcf1_commit.py"}
        or report.get("source_disjoint") is not True
        or report.get("sealed_content_materialized") is not False
        or report.get("protected_board_inputs") != 0
        or report.get("public_inputs") != 0
    ):
        raise PCF1CustodyError("PCF1 materialization report differs")
    _require_access_zero(report, "materialization report")
    identities = report.get("identity_receipts")
    if (
        not isinstance(identities, dict)
        or identities.get("train")
        != {
            "count": TOTAL_TRAIN,
            "ordered_identity_sha256": _ordered_identity_sha256(calibration_ids),
        }
        or identities.get("development")
        != {
            "count": TOTAL_CONFIRMATION,
            "ordered_identity_sha256": _ordered_identity_sha256(confirmation_ids),
        }
        or identities.get("sealed", {}).get("count") != TOTAL_SEALED
        or identities.get("sealed", {}).get("content_materialized") is not False
    ):
        raise PCF1CustodyError("PCF1 materialization identity receipt differs")
    outputs = report.get("outputs")
    specifications = {
        "revision_train": (args.revision_training_data, *receipts["revision_train"]),
        "calibration": (args.calibration_data, *receipts["calibration"]),
        "confirmation": (args.confirmation_data, *receipts["confirmation"]),
    }
    if not isinstance(outputs, dict):
        raise PCF1CustodyError("PCF1 materialization outputs are absent")
    for name, (path, digest, count) in specifications.items():
        receipt = outputs.get(name)
        if (
            not isinstance(receipt, dict)
            or not _recorded_path_matches(receipt.get("path"), path, name)
            or receipt.get("sha256") != digest
            or receipt.get("rows") != count
            or (
                name == "confirmation_assessors"
                and receipt.get("semantic_access") != "final_score_only"
            )
        ):
            raise PCF1CustodyError(f"PCF1 materialization output differs: {name}")
    assessor = outputs.get("confirmation_assessors")
    assessor_receipt = outputs.get("confirmation_assessor_receipt")
    if (
        not isinstance(assessor, dict)
        or assessor
        != {
            "sha256": assessor_sha256,
            "rows": TOTAL_CONFIRMATION,
            "semantic_access": "final_score_only",
        }
        or not isinstance(assessor_receipt, dict)
        or assessor_receipt != {"sha256": assessor_receipt_sha256, "rows": 1}
    ):
        raise PCF1CustodyError("PCF1 materialization assessor receipts differ")


def _pair_rows(
    *,
    source_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    pair_schema: str,
    split: str,
) -> list[str]:
    sources = {str(row["identity_sha256"]): row for row in source_rows}
    identities: list[str] = []
    for pair in pair_rows:
        identity = pair.get("identity_sha256")
        source = sources.get(str(identity))
        candidates = pair.get("candidates")
        if (
            pair.get("schema") != pair_schema
            or source is None
            or pair.get("task") != source.get("task")
            or pair.get("question") != source.get("question")
            or not isinstance(candidates, list)
            or len(candidates) != 2
            or [candidate.get("lineage") for candidate in candidates]
            != ["revision", "unchanged"]
            or any(
                not isinstance(candidate.get("completion"), str)
                or (split != "confirmation" and not candidate["completion"].strip())
                for candidate in candidates
            )
        ):
            raise PCF1CustodyError(f"PCF1 {split} pair content/order differs")
        if split == "confirmation":
            if pair.get("split") != "confirmation" or any(
                any(field in candidate for field in ("correct", "gold", "answer"))
                for candidate in candidates
            ):
                raise PCF1CustodyError("PCF1 confirmation pair exposes a label")
        else:
            digest = hashlib.sha256(f"{CALIBRATION_SEED}\0{identity}".encode()).digest()
            expected = (
                "calibration_train"
                if int.from_bytes(digest[:8], "big") % 10_000 < 8_000
                else "calibration_development"
            )
            if pair.get("split") != expected:
                raise PCF1CustodyError("PCF1 calibration pair split differs")
        identities.append(str(identity))
    if len(identities) != len(sources) or len(set(identities)) != len(identities):
        raise PCF1CustodyError(f"PCF1 {split} pair identity coverage differs")
    if set(identities) != set(sources):
        raise PCF1CustodyError(f"PCF1 {split} pair/source identities differ")
    if split == "confirmation" and identities != [
        str(row["identity_sha256"]) for row in source_rows
    ]:
        raise PCF1CustodyError("PCF1 confirmation pair/order custody differs")
    if split == "calibration" and [
        (str(row.get("split")), str(row.get("identity_sha256"))) for row in pair_rows
    ] != sorted(
        (str(row.get("split")), str(row.get("identity_sha256"))) for row in pair_rows
    ):
        raise PCF1CustodyError("PCF1 calibration pair order differs")
    return identities


def _build_data(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    run_id = _run_id(args.run_id)
    freeze, freeze_sha = _load_report(
        args.source_freeze_report,
        schema=FREEZE_REPORT_SCHEMA,
        label="source freeze report",
    )
    _freeze_report(freeze)
    assessor_receipt, assessor_receipt_sha = _load_report(
        args.confirmation_assessor_receipt,
        schema=CONFIRMATION_ASSESSOR_RECEIPT_SCHEMA,
        label="confirmation assessor receipt",
    )
    assessor_sha = assessor_receipt.get("board_sha256")
    if (
        not _sha256(assessor_sha)
        or assessor_receipt.get("rows") != TOTAL_CONFIRMATION
        or assessor_receipt.get("semantic_access") != "final_score_only"
    ):
        raise PCF1CustodyError("PCF1 confirmation assessor receipt differs")
    freeze_assessor = freeze.get("outputs", {}).get("confirmation_assessor_receipt")
    if (
        not isinstance(freeze_assessor, dict)
        or freeze_assessor.get("sha256") != assessor_receipt_sha
        or freeze_assessor.get("board_sha256") != assessor_sha
        or freeze_assessor.get("rows") != 1
    ):
        raise PCF1CustodyError("PCF1 source-freeze assessor receipt differs")
    train_sources, development_sources, train_sources_sha, development_sources_sha = (
        _source_views(freeze, args)
    )
    reference_sandbox_sha, reference_rows_sha, reference_preflight = (
        _reference_preflight(freeze, args, train_sources, development_sources)
    )
    sources: dict[str, Mapping[str, Any]] = {
        **train_sources,
        **development_sources,
    }
    drafts, drafts_sha = _load_jsonl(args.merged_drafts, "merged drafts")
    draft_report, draft_report_sha = _load_report(
        args.merged_drafts_report,
        schema=MERGED_DRAFT_REPORT_SCHEMA,
        label="merged draft report",
    )
    drafts_by_identity, draft_adapter = _draft_rows(
        drafts, draft_report, drafts_sha, args, sources
    )
    revision_rows, revision_sha = _load_jsonl(
        args.revision_training_data, "revision training data"
    )
    calibration_rows, calibration_sha = _load_jsonl(
        args.calibration_data, "calibration data"
    )
    confirmation_rows, confirmation_sha = _load_jsonl(
        args.confirmation_data, "confirmation data"
    )
    if (
        len(revision_rows) != REVISION_PRESENTATIONS
        or len(calibration_rows) != TOTAL_TRAIN
        or len(confirmation_rows) != TOTAL_CONFIRMATION
    ):
        raise PCF1CustodyError("PCF1 materialized cardinality differs")
    calibration_ids, confirmation_ids = _materialized_rows(
        revision_rows=revision_rows,
        calibration_rows=calibration_rows,
        confirmation_rows=confirmation_rows,
        drafts=drafts_by_identity,
        sources=sources,
    )
    materialization, materialization_sha = _load_report(
        args.data_report, schema=DATA_REPORT_SCHEMA, label="materialization report"
    )
    _materialization_report(
        materialization,
        args=args,
        receipts={
            "revision_train": (revision_sha, REVISION_PRESENTATIONS),
            "calibration": (calibration_sha, TOTAL_TRAIN),
            "confirmation": (confirmation_sha, TOTAL_CONFIRMATION),
        },
        calibration_ids=calibration_ids,
        confirmation_ids=confirmation_ids,
        assessor_sha256=assessor_sha,
        assessor_receipt_sha256=assessor_receipt_sha,
    )

    calibration_pairs, calibration_pairs_sha = _load_jsonl(
        args.calibration_pairs, "calibration pairs"
    )
    _pair_rows(
        source_rows=calibration_rows,
        pair_rows=calibration_pairs,
        pair_schema=CALIBRATION_PAIR_SCHEMA,
        split="calibration",
    )
    calibration_pair_report, calibration_pair_report_sha = _load_report(
        args.calibration_pair_report,
        schema=CALIBRATION_PAIR_REPORT_SCHEMA,
        label="calibration pair report",
    )
    calibration_inputs = calibration_pair_report.get("inputs")
    calibration_counts = calibration_pair_report.get("counts")
    if (
        calibration_pair_report.get("seed") != CALIBRATION_SEED
        or not isinstance(calibration_counts, dict)
        or sum(calibration_counts.values()) != TOTAL_TRAIN
        or calibration_pair_report.get("confirmation_rows_loaded") != 0
        or calibration_pair_report.get("source_disjoint_from_confirmation") is not True
        or calibration_pair_report.get("output_sha256") != calibration_pairs_sha
        or not _recorded_path_matches(
            calibration_pair_report.get("output"),
            args.calibration_pairs,
            "calibration pairs",
        )
        or not isinstance(calibration_inputs, dict)
        or calibration_inputs.get("data_sha256") != calibration_sha
        or not _recorded_path_matches(
            calibration_inputs.get("data"), args.calibration_data, "calibration data"
        )
        or calibration_inputs.get("revision_report_sha256")
        != sha256_file(args.calibration_revision_report)
        or calibration_inputs.get("unchanged_report_sha256")
        != sha256_file(args.calibration_unchanged_report)
    ):
        raise PCF1CustodyError("PCF1 calibration-pair report binding differs")
    _require_access_zero(calibration_pair_report, "calibration pair report")

    confirmation_pairs, confirmation_pairs_sha = _load_jsonl(
        args.confirmation_pairs, "confirmation pairs"
    )
    pair_ids = _pair_rows(
        source_rows=confirmation_rows,
        pair_rows=confirmation_pairs,
        pair_schema=CONFIRMATION_PAIR_SCHEMA,
        split="confirmation",
    )
    confirmation_pair_report, confirmation_pair_report_sha = _load_report(
        args.confirmation_pair_report,
        schema=CONFIRMATION_PAIR_REPORT_SCHEMA,
        label="confirmation pair report",
    )
    confirmation_inputs = confirmation_pair_report.get("inputs")
    if (
        confirmation_pair_report.get("rows") != TOTAL_CONFIRMATION
        or confirmation_pair_report.get("labels_or_correctness_fields") != 0
        or confirmation_pair_report.get("source_disjoint_from_calibration") is not True
        or confirmation_pair_report.get("output_sha256") != confirmation_pairs_sha
        or not _recorded_path_matches(
            confirmation_pair_report.get("output"),
            args.confirmation_pairs,
            "confirmation pairs",
        )
        or not isinstance(confirmation_inputs, dict)
        or confirmation_inputs.get("data_sha256") != confirmation_sha
        or not _recorded_path_matches(
            confirmation_inputs.get("data"),
            args.confirmation_data,
            "confirmation data",
        )
        or confirmation_inputs.get("revision_report_sha256")
        != sha256_file(args.revision_report)
        or confirmation_inputs.get("unchanged_report_sha256")
        != sha256_file(args.unchanged_report)
    ):
        raise PCF1CustodyError("PCF1 confirmation-pair report binding differs")
    _require_access_zero(confirmation_pair_report, "confirmation pair report")

    custody = {
        "schema": DATA_CUSTODY_SCHEMA,
        "status": "complete",
        "run_id": run_id,
        "custody_verified": True,
        "source_disjoint": True,
        "confirmation_rows": TOTAL_CONFIRMATION,
        "identity_order_sha256": _ordered_identity_sha256(pair_ids),
        "data_sha256": confirmation_sha,
        "confirmation_assessors_sha256": assessor_sha,
        "confirmation_assessor_receipt_sha256": assessor_receipt_sha,
        "data_report_sha256": materialization_sha,
        "source_freeze_report_sha256": freeze_sha,
        "train_sources_sha256": train_sources_sha,
        "development_sources_sha256": development_sources_sha,
        "source_lineage_verified": True,
        "mbpp_reference_preflight_verified": True,
        "mbpp_reference_rows": reference_preflight["rows"],
        "mbpp_reference_identity_order_sha256": reference_preflight[
            "ordered_identity_sha256"
        ],
        "mbpp_reference_row_receipts_sha256": reference_preflight[
            "row_receipts_sha256"
        ],
        "mbpp_reference_preflight_rows_sha256": reference_rows_sha,
        "mbpp_reference_unique_setups": reference_preflight["unique_setups"],
        "mbpp_source_preflight_setup_pair_receipts_sha256": reference_preflight[
            "setup_pair_receipts_sha256"
        ],
        "reference_sandbox_receipt_sha256": reference_sandbox_sha,
        "merged_drafts_sha256": drafts_sha,
        "merged_drafts_report_sha256": draft_report_sha,
        "draft_adapter_checkpoint_sha256": draft_adapter,
        "revision_training_data_sha256": revision_sha,
        "calibration_data_sha256": calibration_sha,
        "calibration_pairs_sha256": calibration_pairs_sha,
        "calibration_pair_report_sha256": calibration_pair_report_sha,
        "confirmation_pairs_sha256": confirmation_pairs_sha,
        "confirmation_pair_report_sha256": confirmation_pair_report_sha,
        "holdout_sealed": True,
        "product_sealed": True,
        "public_sealed": True,
        "holdout_access_count": 0,
        "product_access_count": 0,
        "public_access_count": 0,
    }
    context = {
        "materialization_sha": materialization_sha,
        "calibration_sha": calibration_sha,
        "confirmation_sha": confirmation_sha,
        "assessor_sha": assessor_sha,
        "assessor_receipt_sha": assessor_receipt_sha,
        "calibration_pairs_sha": calibration_pairs_sha,
        "confirmation_pairs_sha": confirmation_pairs_sha,
        "confirmation_ids": pair_ids,
        "draft_adapter": draft_adapter,
        "train_sources_sha": train_sources_sha,
        "development_sources_sha": development_sources_sha,
        "revision_sha": revision_sha,
        "calibration_pair_report": calibration_pair_report,
        "confirmation_pair_report": confirmation_pair_report,
        "calibration_rows": calibration_rows,
    }
    return custody, context


def _model_binding(report: Mapping[str, Any], model_root: str, label: str) -> None:
    if (
        report.get("model_root") != model_root
        or report.get("model_revision") != MODEL_REVISION
        or report.get("model_loader") != MODEL_LOADER
    ):
        raise PCF1CustodyError(f"{label} model binding differs")


def _calibration_sandbox_probes(
    paths: list[Path],
    evaluation: Mapping[str, Any],
    sandbox_receipt_sha256: str,
    label: str,
    calibration_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(paths) != 4:
        raise PCF1CustodyError(f"PCF1 {label} sandbox-probe geometry differs")
    resolved: set[Path] = set()
    inodes: set[tuple[int, int]] = set()
    hashes: list[str] = []
    sandbox_probe_results: list[str] = []
    for index, path in enumerate(paths):
        explicit = _explicit_file(path, f"{label} shard {index} sandbox probe")
        identity = (explicit.stat().st_dev, explicit.stat().st_ino)
        if explicit in resolved or identity in inodes:
            raise PCF1CustodyError(f"PCF1 {label} sandbox-probe identity differs")
        resolved.add(explicit)
        inodes.add(identity)
        receipt, receipt_sha = _load_report(
            explicit,
            schema=SANDBOX_RECEIPT_SCHEMA,
            status="pass",
            label=f"{label} shard {index} sandbox probe",
        )
        try:
            validate_sandbox_receipt_payload(receipt)
        except PCF1SandboxError as error:
            raise PCF1CustodyError(
                f"PCF1 {label} sandbox-probe receipt differs"
            ) from error
        if receipt_sha != sandbox_receipt_sha256:
            raise PCF1CustodyError(f"PCF1 {label} sandbox-probe hash differs")
        hashes.append(receipt_sha)
        sandbox_probe_results.append(str(receipt["probe_sha256"]))
    if evaluation.get("shard_sandbox_probe_sha256s") != hashes:
        raise PCF1CustodyError(f"PCF1 {label} merged sandbox-probe binding differs")
    input_receipts = evaluation.get("inputs")
    setup_shards = evaluation.get("mbpp_allocation_setup_receipt_shards")
    if (
        evaluation.get("mbpp_allocation_setup_status") != "passed"
        or not isinstance(input_receipts, list)
        or not isinstance(setup_shards, list)
        or len(input_receipts) != 4
        or len(setup_shards) != 4
    ):
        raise PCF1CustodyError(f"PCF1 {label} setup qualification differs")
    inputs_by_shard = {
        receipt.get("shard_index"): receipt
        for receipt in input_receipts
        if isinstance(receipt, dict)
    }
    if set(inputs_by_shard) != set(range(4)):
        raise PCF1CustodyError(f"PCF1 {label} setup shard geometry differs")
    ordered_shards = setup_shards
    if [
        shard.get("shard_index") if isinstance(shard, dict) else None
        for shard in ordered_shards
    ] != list(range(4)):
        raise PCF1CustodyError(f"PCF1 {label} setup shard order differs")
    total_receipts = 0
    for index, shard in enumerate(ordered_shards):
        input_receipt = inputs_by_shard[index]
        start = input_receipt.get("row_start")
        end = input_receipt.get("row_end")
        receipts = shard.get("receipts") if isinstance(shard, dict) else None
        expected_setup_hashes: list[str] = []
        seen: set[str] = set()
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(calibration_rows)
            or not isinstance(receipts, list)
            or set(shard)
            != {
                "shard_index",
                "row_start",
                "row_end",
                "receipts",
                "receipt_count",
                "receipts_sha256",
            }
            or shard.get("shard_index") != index
            or shard.get("row_start") != start
            or shard.get("row_end") != end
        ):
            raise PCF1CustodyError(f"PCF1 {label} setup shard differs")
        for row in calibration_rows[start:end]:
            if row.get("task") != "mbpp":
                continue
            assessor = row.get("assessor")
            setup = (
                assessor.get("test_setup_code", "")
                if isinstance(assessor, dict)
                else None
            )
            if not isinstance(setup, str):
                raise PCF1CustodyError(f"PCF1 {label} assessor setup differs")
            setup_sha256 = hashlib.sha256(setup.encode()).hexdigest()
            if setup_sha256 not in seen:
                seen.add(setup_sha256)
                expected_setup_hashes.append(setup_sha256)
        if (
            shard.get("receipt_count") != len(receipts)
            or len(receipts) != len(expected_setup_hashes)
            or shard.get("receipts_sha256")
            != mbpp_allocation_setup_receipts_sha256(receipts)
        ):
            raise PCF1CustodyError(f"PCF1 {label} setup receipt coverage differs")
        for setup_receipt, setup_sha256 in zip(
            receipts, expected_setup_hashes, strict=True
        ):
            if not isinstance(setup_receipt, dict):
                raise PCF1CustodyError(f"PCF1 {label} setup receipt differs")
            try:
                validate_mbpp_setup_qualification_receipt(
                    setup_receipt,
                    allocation_probe_sha256=sandbox_probe_results[index],
                    setup_source_sha256=setup_sha256,
                )
            except PCF1SandboxError as error:
                raise PCF1CustodyError(f"PCF1 {label} setup receipt differs") from error
        total_receipts += len(receipts)
    aggregate_sha256 = hashlib.sha256(
        b"".join(
            (json.dumps(shard, sort_keys=True, separators=(",", ":")) + "\n").encode()
            for shard in ordered_shards
        )
    ).hexdigest()
    if (
        evaluation.get("mbpp_allocation_setup_receipt_count") != total_receipts
        or evaluation.get("mbpp_allocation_setup_receipt_shards_sha256")
        != aggregate_sha256
    ):
        raise PCF1CustodyError(f"PCF1 {label} setup aggregate differs")
    return {
        "sandbox_probe_sha256s": hashes,
        "setup_receipt_count": total_receipts,
        "setup_receipt_shards_sha256": aggregate_sha256,
    }


def _training_reports(
    *,
    b1: Mapping[str, Any],
    revision: Mapping[str, Any],
    commit: Mapping[str, Any],
    hashes: Mapping[str, str],
    data: Mapping[str, Any],
    args: argparse.Namespace,
    environment: Mapping[str, Any],
) -> None:
    b1_expected = {
        "arm": "baseline",
        "data_sha256": B1_DATA_SHA256,
        "updates": 256,
        "batch_size": 1,
        "gradient_accumulation": 16,
        "max_sequence_length": 1024,
        "learning_rate": 2e-4,
        "lora_layers": 4,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_scope": "token_mixer",
        "seed": 2026080711,
        "data_seed": 20260802,
    }
    revision_expected = {
        "arm": "baseline",
        "data_sha256": data["revision_training_data_sha256"],
        "updates": 256,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "max_sequence_length": 4096,
        "learning_rate": 2e-5,
        "lora_layers": 4,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_scope": "token_mixer",
        "seed": 2026080815,
        "data_seed": 2026080814,
        "warm_start_sha256": hashes["b1"],
    }
    if (
        any(b1.get(key) != value for key, value in b1_expected.items())
        or b1.get("warm_start_sha256") is not None
    ):
        raise PCF1CustodyError("PCF1 B1 training settings differ")
    if any(revision.get(key) != value for key, value in revision_expected.items()):
        raise PCF1CustodyError("PCF1 revision training settings differ")
    if b1.get("selected_rows") != 100_000:
        raise PCF1CustodyError("PCF1 B1 selected-row geometry differs")
    if revision.get("selected_rows") != REVISION_PRESENTATIONS:
        raise PCF1CustodyError("PCF1 revision selected-row geometry differs")
    for label, report in (("B1", b1), ("revision", revision)):
        if (
            not isinstance(report.get("trainable_parameters"), int)
            or report.get("trainable_parameters", 0) <= 0
            or not _sha256(report.get("trainable_parameter_name_sha256"))
            or report.get("lora_layer_indices") != [30, 31, 32, 33]
            or report.get("environment_verified") is not True
            or report.get("environment_receipt_sha256")
            != args.environment_receipt_sha256
            or report.get("environment_tree_sha256")
            != environment.get("environment_tree", {}).get("sha256")
        ):
            raise PCF1CustodyError(f"PCF1 {label} training admission differs")
    commit_expected = {
        "adapter_checkpoint_sha256": hashes["b1"],
        "checkpoint_sha256": hashes["commit"],
        "updates": 128,
        "gradient_accumulation": 8,
        "head_width": 512,
        "max_sequence_length": 3072,
        "backbone_learning_rate": 2e-6,
        "head_learning_rate": 2e-4,
        "tie_loss_weight": 0.25,
        "seed": 2026080822,
        "pairs_sha256": data["calibration_pairs_sha256"],
        "protected_adapter_sha256_after": hashes["b1"],
        "protected_adapter_unchanged": True,
        "trainable_parameters": b1["trainable_parameters"],
        "trainable_parameter_name_sha256": b1["trainable_parameter_name_sha256"],
        "lora_layer_indices": [30, 31, 32, 33],
    }
    if any(commit.get(key) != value for key, value in commit_expected.items()):
        raise PCF1CustodyError("PCF1 commit training settings differ")
    if (
        commit.get("environment_verified") is not True
        or commit.get("environment_receipt_sha256") != args.environment_receipt_sha256
    ):
        raise PCF1CustodyError("PCF1 commit training environment differs")
    if (
        not _recorded_path_matches(
            commit.get("adapter_checkpoint"), args.b1_checkpoint, "B1 checkpoint"
        )
        or not _recorded_path_matches(
            commit.get("checkpoint"), args.commit_checkpoint, "commit checkpoint"
        )
        or not _recorded_path_matches(
            commit.get("pairs"), args.calibration_pairs, "calibration pairs"
        )
    ):
        raise PCF1CustodyError("PCF1 commit training path binding differs")
    _require_access_zero(commit, "commit training report")


def _evaluation_report(
    *,
    report: Mapping[str, Any],
    arm: str,
    split: str,
    rows: int,
    adapter_sha256: str,
    data_sha256: str,
    data_report_sha256: str,
    model_root: str,
) -> None:
    _model_binding(report, model_root, f"{split} {arm} evaluation")
    counters = report.get("counters")
    prompt_tokens = (
        counters.get("prompt_tokens") if isinstance(counters, dict) else None
    )
    capability_policy_rejections = (
        counters.get("capability_policy_rejections")
        if isinstance(counters, dict)
        else None
    )
    trainable_parameters = report.get("trainable_parameters")
    trainable_name_sha256 = report.get("trainable_parameter_name_sha256")
    aggregate_wall_seconds = report.get("aggregate_wall_seconds")
    aggregate_gpu_seconds = report.get("aggregate_gpu_seconds")
    peak_gpu_memory_bytes = report.get("maximum_peak_gpu_memory_bytes")
    deferred = split == "confirmation"
    metrics_invalid = (
        report.get("metrics") is not None
        if deferred
        else not isinstance(report.get("metrics"), dict)
    )
    if (
        report.get("arm") != arm
        or report.get("split") != split
        or report.get("adapter_checkpoint_sha256") != adapter_sha256
        or report.get("data_sha256") != data_sha256
        or report.get("data_report_sha256") != data_report_sha256
        or report.get("generation_mode") != "greedy"
        or report.get("max_new_tokens") != 768
        or report.get("seed") != EVALUATION_SEED
        or report.get("batch_size") != 2
        or not isinstance(report.get("shard_count"), int)
        or report.get("shard_count", 0) <= 0
        or report.get("full_row_count") != rows
        or report.get("exact_identity_coverage") is not True
        or not _sha256(report.get("candidates_sha256"))
        or not isinstance(counters, dict)
        or counters.get("rows") != rows
        or isinstance(prompt_tokens, bool)
        or not isinstance(prompt_tokens, int)
        or prompt_tokens <= 0
        or report.get("aggregate_prompt_tokens") != prompt_tokens
        or isinstance(capability_policy_rejections, bool)
        or not isinstance(capability_policy_rejections, int)
        or capability_policy_rejections < 0
        or (deferred and capability_policy_rejections != 0)
        or isinstance(trainable_parameters, bool)
        or not isinstance(trainable_parameters, int)
        or trainable_parameters <= 0
        or not _sha256(trainable_name_sha256)
        or not _sha256(report.get("adapter_metadata_sha256"))
        or report.get("lora_layer_indices") != [30, 31, 32, 33]
        or isinstance(aggregate_wall_seconds, bool)
        or not isinstance(aggregate_wall_seconds, (int, float))
        or aggregate_wall_seconds < 0
        or aggregate_gpu_seconds != aggregate_wall_seconds
        or isinstance(peak_gpu_memory_bytes, bool)
        or not isinstance(peak_gpu_memory_bytes, int)
        or peak_gpu_memory_bytes < 0
        or metrics_invalid
        or report.get("assessment_mode")
        != ("confirmation_deferred" if deferred else "calibration_immediate")
        or report.get("assessor_board_access_count") != 0
        or report.get("runtime_fields")
        != (
            ["source_prompt", "internal_draft.completion"]
            if arm == "self_refinement"
            else ["question"]
        )
        or (
            deferred
            and (
                report.get("mbpp_allocation_setup_status")
                != "not_applicable_no_code_scoring"
                or report.get("mbpp_allocation_setup_receipt_shards") != []
                or report.get("mbpp_allocation_setup_receipt_count") != 0
                or report.get("mbpp_allocation_setup_receipt_shards_sha256") is not None
            )
        )
    ):
        raise PCF1CustodyError(f"PCF1 {split} {arm} evaluation lineage differs")
    _require_access_zero(report, f"{split} {arm} evaluation")


def _compute_host_receipt(
    path: Path, mechanics: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    receipt, digest = _load_report(
        path,
        schema=COMPUTE_HOST_RECEIPT_SCHEMA,
        label="compute-host receipt",
    )
    expected_keys = {
        "schema",
        "status",
        "partition",
        "node",
        "excluded_nodes",
        "nvidia_smi_invoked_path",
        "nvidia_smi_resolved_path",
        "nvidia_smi_sha256",
        "nvidia_smi_version",
        "visible_gpu_count",
        "gpu_name",
        "driver_version",
        "pci_bus_id",
    }
    invoked = Path(str(receipt.get("nvidia_smi_invoked_path", "")))
    resolved = Path(str(receipt.get("nvidia_smi_resolved_path", "")))
    if (
        set(receipt) != expected_keys
        or receipt.get("partition") != "normal"
        or not isinstance(receipt.get("node"), str)
        or not receipt["node"]
        or receipt.get("node") in EXCLUDED_NODES
        or receipt.get("excluded_nodes") != sorted(EXCLUDED_NODES)
        or not invoked.is_absolute()
        or not resolved.is_absolute()
        or not _sha256(receipt.get("nvidia_smi_sha256"))
        or not isinstance(receipt.get("nvidia_smi_version"), str)
        or not receipt["nvidia_smi_version"]
        or receipt.get("visible_gpu_count") != 1
        or receipt.get("gpu_name") != "NVIDIA H100 PCIe"
        or not isinstance(receipt.get("driver_version"), str)
        or not receipt["driver_version"]
        or not isinstance(receipt.get("pci_bus_id"), str)
        or not receipt["pci_bus_id"]
        or mechanics.get("compute_host_receipt_sha256") != digest
    ):
        raise PCF1CustodyError("PCF1 compute-host receipt differs")
    return receipt, digest


def _mechanics_report(
    report: Mapping[str, Any],
    *,
    model_root: str,
    model_manifest_sha256: str,
    runtime_manifest_sha256: str,
    sandbox: Mapping[str, Any],
    sandbox_receipt_sha256: str,
    environment_receipt_sha256: str,
    compute_host: Mapping[str, Any],
    compute_host_sha256: str,
    compute_host_path: Path,
) -> None:
    if (
        report.get("capability_scored") is not False
        or report.get("rows") != 24
        or report.get("task_counts") != {task: 8 for task in TASKS}
        or report.get("model_root") != model_root
        or report.get("model_revision") != MODEL_REVISION
        or report.get("model_loader") != MODEL_LOADER
        or report.get("model_manifest_sha256") != model_manifest_sha256
        or report.get("runtime_manifest_sha256") != runtime_manifest_sha256
        or report.get("environment_receipt_sha256") != environment_receipt_sha256
        or report.get("sandbox_receipt_sha256") != sandbox_receipt_sha256
        or report.get("code_sandbox_config_sha256")
        != sandbox.get("sandbox_config_sha256")
        or report.get("code_sandbox_binary_sha256") != SANDBOX_BINARY_SHA256
        or report.get("code_sandbox_probe_sha256") != sandbox_receipt_sha256
        or report.get("code_sandbox_probe_result_sha256") != sandbox.get("probe_sha256")
        or report.get("code_sandbox_runtime_tree_sha256") != SANDBOX_RUNTIME_TREE_SHA256
        or report.get("sandbox_isolation_passed") is not True
        or not _recorded_path_matches(
            report.get("compute_host_receipt"),
            compute_host_path,
            "compute-host receipt",
        )
        or report.get("compute_host_receipt_sha256") != compute_host_sha256
        or report.get("nvidia_smi_invoked_path")
        != compute_host.get("nvidia_smi_invoked_path")
        or report.get("nvidia_smi_resolved_path")
        != compute_host.get("nvidia_smi_resolved_path")
        or report.get("nvidia_smi_sha256") != compute_host.get("nvidia_smi_sha256")
        or report.get("nvidia_smi_version") != compute_host.get("nvidia_smi_version")
        or report.get("qualified_gpu_name") != compute_host.get("gpu_name")
        or report.get("qualified_driver_version") != compute_host.get("driver_version")
        or report.get("qualified_pci_bus_id") != compute_host.get("pci_bus_id")
        or report.get("qualified_node") != compute_host.get("node")
        or report.get("checkpoint_restored") is not True
        or report.get("optimizer_updates") != 1
        or report.get("optimizer_presentations") != 24
        or report.get("lora_layers") != 4
        or report.get("lora_scope") != "token_mixer"
        or report.get("lora_layer_indices") != [30, 31, 32, 33]
        or not isinstance(report.get("lora_projection_count"), int)
        or report.get("lora_projection_count", 0) <= 0
        or not _sha256(report.get("trainable_parameter_name_sha256"))
        or not isinstance(report.get("trainable_parameters"), int)
        or report.get("trainable_parameters", 0) <= 0
        or report.get("source_only_runtime_fields") != ["source_prompt"]
        or report.get("task_router_used") is not False
        or report.get("revision_prompt_parameters") != ["source_prompt", "draft"]
        or report.get("supervisor_fields_visible_to_model") is not False
        or report.get("drafts_nonempty") is not True
        or report.get("revisions_nonempty") is not True
        or report.get("matched_prompt_ids_identical") is not True
        or report.get("commit_ab_order_checks") != 24
        or report.get("commit_ab_serialization_exact") is not True
        or report.get("commit_forward_swapped_exact") is not True
        or report.get("commit_prompt_truncations") != 0
    ):
        raise PCF1CustodyError("PCF1 mechanics/admission evidence differs")
    _require_access_zero(report, "mechanics report")


def _build_model(
    args: argparse.Namespace,
    data: Mapping[str, Any],
    context: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    mechanics: Mapping[str, Any],
    mechanics_sha: str,
    environment: Mapping[str, Any],
    sandbox: Mapping[str, Any],
    sandbox_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_root = str(_explicit_root(args.model_root, "model root"))
    compute_host, compute_host_sha = _compute_host_receipt(
        args.compute_host_receipt, mechanics
    )
    if args.model_revision != MODEL_REVISION:
        raise PCF1CustodyError("PCF1 model revision differs from the pinned host")
    checkpoints = {
        "b1": _explicit_file(args.b1_checkpoint, "B1 checkpoint"),
        "revision": _explicit_file(args.revision_checkpoint, "revision checkpoint"),
        "commit": _explicit_file(args.commit_checkpoint, "commit checkpoint"),
    }
    hashes = {name: sha256_file(path) for name, path in checkpoints.items()}
    if data.get("draft_adapter_checkpoint_sha256") != hashes["b1"]:
        raise PCF1CustodyError("PCF1 source-only draft/B1 checkpoint lineage differs")

    b1, b1_report_sha = _load_report(
        args.b1_training_report,
        schema=TRAINING_REPORT_SCHEMA,
        label="B1 training report",
    )
    revision_training, revision_training_sha = _load_report(
        args.revision_training_report,
        schema=TRAINING_REPORT_SCHEMA,
        label="revision training report",
    )
    commit_training, commit_training_sha = _load_report(
        args.commit_training_report,
        schema=COMMIT_TRAINING_SCHEMA,
        label="commit training report",
    )
    for label, report in (
        ("B1 training report", b1),
        ("revision training report", revision_training),
        ("commit training report", commit_training),
    ):
        _model_binding(report, model_root, label)
    _training_reports(
        b1=b1,
        revision=revision_training,
        commit=commit_training,
        hashes=hashes,
        data=data,
        args=args,
        environment=environment,
    )

    reports: dict[str, tuple[dict[str, Any], str]] = {}
    for name, path in (
        ("calibration_revision", args.calibration_revision_report),
        ("calibration_unchanged", args.calibration_unchanged_report),
        ("revision", args.revision_report),
        ("unchanged", args.unchanged_report),
        ("self_refinement", args.self_refinement_report),
    ):
        reports[name] = _load_report(
            path, schema=MERGED_EVALUATION_SCHEMA, label=f"{name} evaluation report"
        )
    _evaluation_report(
        report=reports["calibration_revision"][0],
        arm="revision",
        split="calibration",
        rows=TOTAL_TRAIN,
        adapter_sha256=hashes["revision"],
        data_sha256=data["calibration_data_sha256"],
        data_report_sha256=data["data_report_sha256"],
        model_root=model_root,
    )
    _evaluation_report(
        report=reports["calibration_unchanged"][0],
        arm="unchanged",
        split="calibration",
        rows=TOTAL_TRAIN,
        adapter_sha256=hashes["b1"],
        data_sha256=data["calibration_data_sha256"],
        data_report_sha256=data["data_report_sha256"],
        model_root=model_root,
    )
    calibration_sandbox_probe_hashes = {
        "revision": _calibration_sandbox_probes(
            args.calibration_revision_sandbox_probes,
            reports["calibration_revision"][0],
            sandbox_receipt_sha256,
            "calibration revision",
            context["calibration_rows"],
        ),
        "unchanged": _calibration_sandbox_probes(
            args.calibration_unchanged_sandbox_probes,
            reports["calibration_unchanged"][0],
            sandbox_receipt_sha256,
            "calibration unchanged",
            context["calibration_rows"],
        ),
    }
    expected_confirmation = {
        "revision": hashes["revision"],
        "unchanged": hashes["b1"],
        "self_refinement": hashes["b1"],
    }
    for arm in NATIVE_ARMS:
        _evaluation_report(
            report=reports[arm][0],
            arm=arm,
            split="confirmation",
            rows=TOTAL_CONFIRMATION,
            adapter_sha256=expected_confirmation[arm],
            data_sha256=data["data_sha256"],
            data_report_sha256=data["data_report_sha256"],
            model_root=model_root,
        )
    adapter_training = {
        "revision": revision_training,
        "unchanged": b1,
        "self_refinement": b1,
        "calibration_revision": revision_training,
        "calibration_unchanged": b1,
    }
    for name, (evaluation, _) in reports.items():
        training = adapter_training[name]
        calibration = name.startswith("calibration_")
        if (
            evaluation.get("trainable_parameters")
            != training.get("trainable_parameters")
            or evaluation.get("trainable_parameter_name_sha256")
            != training.get("trainable_parameter_name_sha256")
            or evaluation.get("environment_verified") is not True
            or evaluation.get("environment_receipt_sha256")
            != args.environment_receipt_sha256
            or evaluation.get("environment_tree_sha256")
            != environment.get("environment_tree", {}).get("sha256")
            or evaluation.get("code_sandbox_config_sha256")
            != sandbox.get("sandbox_config_sha256")
            or evaluation.get("code_sandbox_binary_sha256") != SANDBOX_BINARY_SHA256
            or (
                calibration
                and (
                    evaluation.get("code_sandbox_status") != "passed"
                    or evaluation.get("code_sandbox_probe_passed") is not True
                    or evaluation.get("code_sandbox_probe_sha256")
                    != sandbox_receipt_sha256
                    or evaluation.get("code_sandbox_probe_result_sha256")
                    != sandbox.get("probe_sha256")
                    or evaluation.get("sandbox_receipt_sha256")
                    != sandbox_receipt_sha256
                )
            )
            or (
                not calibration
                and (
                    evaluation.get("code_sandbox_status")
                    != "not_applicable_no_code_scoring"
                    or evaluation.get("code_sandbox_probe_passed") is not None
                    or evaluation.get("code_sandbox_probe_sha256") is not None
                    or evaluation.get("code_sandbox_probe_result_sha256") is not None
                    or evaluation.get("sandbox_receipt_sha256") is not None
                )
            )
        ):
            raise PCF1CustodyError(f"PCF1 {name} evaluation/training trainables differ")
    cal_pair = context["calibration_pair_report"]
    conf_pair = context["confirmation_pair_report"]
    if (
        cal_pair.get("inputs", {}).get("revision_candidates_sha256")
        != reports["calibration_revision"][0].get("candidates_sha256")
        or cal_pair.get("inputs", {}).get("unchanged_candidates_sha256")
        != reports["calibration_unchanged"][0].get("candidates_sha256")
        or conf_pair.get("inputs", {}).get("revision_candidates_sha256")
        != reports["revision"][0].get("candidates_sha256")
        or conf_pair.get("inputs", {}).get("unchanged_candidates_sha256")
        != reports["unchanged"][0].get("candidates_sha256")
    ):
        raise PCF1CustodyError("PCF1 pair/evaluation candidate binding differs")

    application, application_sha = _load_report(
        args.commit_application_report,
        schema=COMMIT_APPLICATION_SCHEMA,
        label="commit application report",
    )
    selections, selections_sha = _load_jsonl(
        args.confirmation_selections, "confirmation selections"
    )
    selection_ids: list[str] = []
    for row in selections:
        selected_index = row.get("selected_index")
        margin = row.get("margin")
        if (
            row.get("schema") != SELECTION_SCHEMA
            or not _sha256(row.get("identity_sha256"))
            or row.get("task") not in TASKS
            or isinstance(selected_index, bool)
            or selected_index not in (0, 1)
            or row.get("selected_lineage") != ("revision", "unchanged")[selected_index]
            or not isinstance(row.get("order_consistent"), bool)
            or isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not math.isfinite(float(margin))
        ):
            raise PCF1CustodyError("PCF1 confirmation selection row differs")
        selection_ids.append(str(row["identity_sha256"]))
    if selection_ids != context["confirmation_ids"]:
        raise PCF1CustodyError("PCF1 confirmation selection order differs")
    prompt_truncated = application.get("prompt_truncated")
    malformed = application.get("malformed")
    order_consistent = application.get("order_consistent")
    _model_binding(application, model_root, "commit application report")
    if (
        application.get("adapter_checkpoint_sha256") != hashes["b1"]
        or application.get("commit_checkpoint_sha256") != hashes["commit"]
        or application.get("pairs_sha256") != data["confirmation_pairs_sha256"]
        or application.get("pairs_report_sha256")
        != data["confirmation_pair_report_sha256"]
        or application.get("selections_sha256") != selections_sha
        or not _recorded_path_matches(
            application.get("selections"),
            args.confirmation_selections,
            "confirmation selections",
        )
        or application.get("rows") != TOTAL_CONFIRMATION
        or application.get("max_sequence_length") != 3072
        or isinstance(prompt_truncated, bool)
        or not isinstance(prompt_truncated, int)
        or not 0 <= prompt_truncated <= 2 * TOTAL_CONFIRMATION
        or isinstance(malformed, bool)
        or not isinstance(malformed, int)
        or not 0 <= malformed <= TOTAL_CONFIRMATION
        or isinstance(order_consistent, bool)
        or not isinstance(order_consistent, int)
        or not 0 <= order_consistent <= TOTAL_CONFIRMATION
        or order_consistent != sum(int(row["order_consistent"]) for row in selections)
        or application.get("correctness_or_task_label_visible") is not False
        or application.get("protected_adapter_unchanged") is not True
        or application.get("environment_verified") is not True
        or application.get("environment_receipt_sha256")
        != args.environment_receipt_sha256
        or application.get("environment_tree_sha256")
        != environment.get("environment_tree", {}).get("sha256")
    ):
        raise PCF1CustodyError("PCF1 commit application lineage differs")
    _require_access_zero(application, "commit application report")

    _mechanics_report(
        mechanics,
        model_root=model_root,
        model_manifest_sha256=model_manifest["manifest_sha256"],
        runtime_manifest_sha256=args.runtime_manifest_sha256,
        sandbox=sandbox,
        sandbox_receipt_sha256=sandbox_receipt_sha256,
        environment_receipt_sha256=args.environment_receipt_sha256,
        compute_host=compute_host,
        compute_host_sha256=compute_host_sha,
        compute_host_path=args.compute_host_receipt,
    )
    evaluation_accounting = {
        name: {
            "prompt_tokens": report["counters"]["prompt_tokens"],
            "generated_tokens": report["counters"]["generated_tokens"],
            "wall_seconds": report["aggregate_wall_seconds"],
            "peak_gpu_memory_bytes": report["maximum_peak_gpu_memory_bytes"],
            "trainable_parameters": report["trainable_parameters"],
            "trainable_parameter_name_sha256": report[
                "trainable_parameter_name_sha256"
            ],
            "lora_layer_indices": report["lora_layer_indices"],
            "capability_policy_rejections": report["counters"][
                "capability_policy_rejections"
            ],
        }
        for name, (report, _) in reports.items()
    }
    custody = {
        "schema": MODEL_CUSTODY_SCHEMA,
        "status": "complete",
        "run_id": args.run_id,
        "custody_verified": True,
        "model_root": model_root,
        "model_revision": MODEL_REVISION,
        "model_manifest_sha256": model_manifest["manifest_sha256"],
        "model_tree_sha256": model_manifest["tree_sha256"],
        "model_file_count": model_manifest["file_count"],
        "model_config_sha256": args.model_config_sha256,
        "adapter_lora_layer_indices": [30, 31, 32, 33],
        "evaluation_accounting": evaluation_accounting,
        "calibration_sandbox_probe_sha256s": calibration_sandbox_probe_hashes,
        "mbpp_calibration_setup_qualifications_verified": True,
        "mbpp_calibration_allocation_setup_receipts": (
            calibration_sandbox_probe_hashes
        ),
        "compute_host_verified": True,
        "compute_host_receipt_sha256": compute_host_sha,
        "compute_host": compute_host,
        "b1_compute_host_gate_verified": True,
        "data_custody_sha256": None,
        "commit_training_report_sha256": commit_training_sha,
        "checkpoint_sha256s": {
            "trained_revision": hashes["revision"],
            "unchanged": hashes["b1"],
            "self_refinement": hashes["b1"],
            "learned_commit_host": hashes["b1"],
            "learned_commit": hashes["commit"],
        },
        "training_report_sha256s": {
            "b1": b1_report_sha,
            "revision": revision_training_sha,
            "commit": commit_training_sha,
        },
        "native_report_sha256s": {
            **{name: digest for name, (_, digest) in reports.items()},
            "commit_application": application_sha,
            "confirmation_selections": selections_sha,
            "mechanics": mechanics_sha,
        },
        "holdout_sealed": True,
        "product_sealed": True,
        "public_sealed": True,
        "holdout_access_count": 0,
        "product_access_count": 0,
        "public_access_count": 0,
    }
    return custody, {
        "reports": reports,
        "commit_application": application,
        "commit_training": commit_training,
        "evaluation_accounting": evaluation_accounting,
        "calibration_sandbox_probe_sha256s": calibration_sandbox_probe_hashes,
        "mbpp_calibration_setup_qualifications_verified": True,
        "mbpp_calibration_allocation_setup_receipts": (
            calibration_sandbox_probe_hashes
        ),
        "compute_host_receipt_sha256": compute_host_sha,
        "compute_host": compute_host,
    }


def _build_runtime(
    args: argparse.Namespace,
    model_context: Mapping[str, Any],
    runtime_manifest: Mapping[str, Any],
    mechanics_sha: str,
    environment: Mapping[str, Any],
    environment_sha: str,
    sandbox: Mapping[str, Any],
    sandbox_sha: str,
) -> dict[str, Any]:
    reports = model_context["reports"]
    evaluation_accounting = model_context["evaluation_accounting"]
    calibration_sandbox_probe_hashes = model_context[
        "calibration_sandbox_probe_sha256s"
    ]
    compute_host = model_context["compute_host"]
    compute_host_sha256 = model_context["compute_host_receipt_sha256"]
    revision = reports["revision"][0]
    settings = {
        key: revision[key]
        for key in (
            "model_loader",
            "generation_mode",
            "max_new_tokens",
            "seed",
            "batch_size",
            "shard_count",
        )
    }
    if settings != {
        "model_loader": MODEL_LOADER,
        "generation_mode": "greedy",
        "max_new_tokens": 768,
        "seed": EVALUATION_SEED,
        "batch_size": 2,
        "shard_count": revision["shard_count"],
    } or any(
        any(report.get(key) != value for key, value in settings.items())
        for name, (report, _) in reports.items()
        if name in NATIVE_ARMS
    ):
        raise PCF1CustodyError("PCF1 matched confirmation settings differ")
    commit = model_context["commit_application"]
    commit_settings = {
        "model_loader": commit.get("model_loader"),
        "max_sequence_length": commit.get("max_sequence_length"),
    }
    if commit_settings != {
        "model_loader": MODEL_LOADER,
        "max_sequence_length": 3072,
    }:
        raise PCF1CustodyError("PCF1 frozen commit settings differ")
    return {
        "schema": RUNTIME_CUSTODY_SCHEMA,
        "status": "complete",
        "run_id": args.run_id,
        "custody_verified": True,
        "model_revision": MODEL_REVISION,
        "runtime_sha256": runtime_manifest["manifest_sha256"],
        "runtime_manifest_sha256": runtime_manifest["manifest_sha256"],
        "runtime_tree_sha256": runtime_manifest["tree_sha256"],
        "runtime_file_count": runtime_manifest["file_count"],
        "evaluation_settings": settings,
        "commit_settings": commit_settings,
        "evaluation_accounting": evaluation_accounting,
        "evaluation_accounting_sha256": hashlib.sha256(
            json.dumps(
                evaluation_accounting, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "calibration_sandbox_probe_sha256s": calibration_sandbox_probe_hashes,
        "mbpp_calibration_setup_qualifications_verified": True,
        "mbpp_calibration_allocation_setup_receipts": (
            calibration_sandbox_probe_hashes
        ),
        "compute_host_verified": True,
        "compute_host_receipt_sha256": compute_host_sha256,
        "compute_host": compute_host,
        "b1_compute_host_gate_verified": True,
        "mechanics_report_sha256": mechanics_sha,
        "environment_verified": True,
        "environment_receipt_sha256": environment_sha,
        "environment_tree_sha256": environment["environment_tree"]["sha256"],
        "environment_runtime_sha256": environment["environment_runtime_sha256"],
        "environment_pip_freeze_sha256": environment["pip_freeze_sha256"],
        "environment_python_sha256": environment["python"]["executable_sha256"],
        "code_sandbox_verified": True,
        "code_sandbox_config_sha256": sandbox["sandbox_config_sha256"],
        "code_sandbox_binary_sha256": sandbox["bwrap_sha256"],
        "code_sandbox_probe_sha256": sandbox_sha,
        "code_sandbox_probe_result_sha256": sandbox["probe_sha256"],
        "code_sandbox_receipt_sha256": sandbox_sha,
        "code_sandbox_runtime_tree_sha256": sandbox["sandbox_runtime_tree_sha256"],
        "holdout_sealed": True,
        "product_sealed": True,
        "public_sealed": True,
        "holdout_access_count": 0,
        "product_access_count": 0,
        "public_access_count": 0,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(encoded).hexdigest()


def _publish_directory(output: Path, payloads: Mapping[str, Mapping[str, Any]]) -> None:
    output = _safe_path(output, "custody output")
    if output.exists() or output.is_symlink():
        raise PCF1CustodyError(f"refusing existing PCF1 custody output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise PCF1CustodyError("refusing existing PCF1 custody temporary output")
    temporary.mkdir()
    try:
        for name, payload in payloads.items():
            _write_json(temporary / f"{name}.json", payload)
        directory = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        os.rename(temporary, output)
        parent = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def precompute(args: argparse.Namespace) -> dict[str, Any]:
    _run_id(args.run_id)
    if args.model_revision != MODEL_REVISION:
        raise PCF1CustodyError("PCF1 model revision differs")
    environment, environment_sha, sandbox, sandbox_sha = _external_runtime_receipts(
        args
    )
    model_manifest = _exact_manifest(
        root=args.model_root,
        manifest_path=args.model_manifest,
        expected_sha256=args.model_manifest_sha256,
        label="model",
    )
    config_path = _explicit_root(args.model_root, "model root") / "config.json"
    if (
        not _sha256(args.model_config_sha256)
        or not config_path.is_file()
        or sha256_file(config_path) != args.model_config_sha256
    ):
        raise PCF1CustodyError("PCF1 model config hash differs")
    runtime_manifest = _exact_manifest(
        root=args.runtime_root,
        manifest_path=args.runtime_manifest,
        expected_sha256=args.runtime_manifest_sha256,
        label="runtime",
    )
    mechanics, mechanics_sha = _load_report(
        args.mechanics_report,
        schema=MECHANICS_SCHEMA,
        status="pass",
        label="mechanics report",
    )
    data, data_context = _build_data(args)
    model, model_context = _build_model(
        args,
        data,
        data_context,
        model_manifest,
        mechanics,
        mechanics_sha,
        environment,
        sandbox,
        sandbox_sha,
    )
    runtime = _build_runtime(
        args,
        model_context,
        runtime_manifest,
        mechanics_sha,
        environment,
        environment_sha,
        sandbox,
        sandbox_sha,
    )

    # Bind the exact bytes that will be emitted without creating a circular hash.
    encoded_data = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()
    model["data_custody_sha256"] = hashlib.sha256(encoded_data).hexdigest()
    _publish_directory(
        args.output,
        {
            "data_custody": data,
            "model_custody": model,
            "runtime_custody": runtime,
        },
    )
    return {
        "status": "complete",
        "run_id": args.run_id,
        "output": str(_safe_path(args.output, "custody output")),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    output = _safe_path(path, "compute custody output")
    if output.exists() or output.is_symlink():
        raise PCF1CustodyError(f"refusing existing PCF1 custody output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        _write_json(temporary, payload)
        os.link(temporary, output)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise PCF1CustodyError(
            f"refusing existing PCF1 custody output: {output}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _accounting(
    dispatch: Mapping[str, Any],
    accounting: Mapping[str, Any],
    run_id: str,
    expected_stages: tuple[str, ...],
) -> None:
    if (
        dispatch.get("run_id") != run_id
        or dispatch.get("terminal_stage") != "final_compare"
        or dispatch.get("retry_authorized") is not False
        or dispatch.get("successor_authorized") is not False
        or dispatch.get("stop_after_gate") is not True
    ):
        raise PCF1CustodyError("PCF1 dispatch custody differs")
    job_ids = dispatch.get("job_ids")
    predecessors = dispatch.get("accounting_predecessors")
    stage_resources = dispatch.get("stage_resources")
    jobs = accounting.get("jobs")
    expected_resources = {
        stage: {
            "gpus": int(stage in GPU_STAGES),
            "is_array": stage in ARRAY_TASKS,
            "array_tasks": ARRAY_TASKS.get(stage, 1),
        }
        for stage in expected_stages
    }
    if (
        accounting.get("run_id") != run_id
        or accounting.get("partition") != "normal"
        or accounting.get("excluded_nodes") != EXCLUDED_NODES
        or accounting.get("all_required_complete") is not True
        or accounting.get("retry_count") != 0
        or accounting.get("successor_authorized") is not False
        or accounting.get("successor_submitted") is not False
        or not isinstance(job_ids, dict)
        or set(job_ids) != set(expected_stages)
        or predecessors != list(expected_stages)
        or accounting.get("required_stages") != list(expected_stages)
        or stage_resources != expected_resources
        or not isinstance(jobs, dict)
        or set(jobs) != set(expected_stages)
        or any(stage not in job_ids for stage in predecessors)
    ):
        raise PCF1CustodyError("PCF1 scheduler accounting geometry differs")
    charged = 0.0
    for stage in predecessors:
        job = jobs[stage]
        records = job.get("records") if isinstance(job, dict) else None
        job_charge = job.get("charged_gpu_seconds") if isinstance(job, dict) else None
        resource = expected_resources[stage]
        if (
            not isinstance(job, dict)
            or job.get("submitted_job_id") != job_ids[stage]
            or not isinstance(records, list)
            or not records
            or isinstance(job_charge, bool)
            or not isinstance(job_charge, (int, float))
            or job_charge < 0
            or len(records) != resource["array_tasks"]
        ):
            raise PCF1CustodyError(f"PCF1 scheduler stage differs: {stage}")
        submitted = str(job_ids[stage])
        expected_job_ids = (
            [f"{submitted}_{index}" for index in range(resource["array_tasks"])]
            if resource["is_array"]
            else [submitted]
        )
        record_charge = 0.0
        for record, expected_job_id in zip(records, expected_job_ids, strict=True):
            elapsed_raw = (
                record.get("elapsed_raw") if isinstance(record, dict) else None
            )
            allocated_gpus = (
                record.get("allocated_gpus") if isinstance(record, dict) else None
            )
            charged_gpu_seconds = (
                record.get("charged_gpu_seconds") if isinstance(record, dict) else None
            )
            expected_gpu_types = (
                {"nvidia_h100_pcie": 1} if resource["gpus"] == 1 else {}
            )
            if (
                not isinstance(record, dict)
                or record.get("job_id_raw") != expected_job_id
                or str(record.get("state", "")).split()[0] != "COMPLETED"
                or record.get("partition") not in ("", "normal")
                or isinstance(elapsed_raw, bool)
                or not isinstance(elapsed_raw, int)
                or elapsed_raw < 0
                or not isinstance(record.get("alloc_tres"), str)
                or not isinstance(record.get("node_list"), str)
                or record.get("exit_code") != "0:0"
                or record.get("restarts") != 0
                or allocated_gpus != resource["gpus"]
                or record.get("allocated_gpu_types") != expected_gpu_types
                or isinstance(charged_gpu_seconds, bool)
                or not isinstance(charged_gpu_seconds, (int, float))
                or float(charged_gpu_seconds) != elapsed_raw * resource["gpus"]
            ):
                raise PCF1CustodyError(f"PCF1 scheduler record differs: {stage}")
            record_charge += float(charged_gpu_seconds)
        if abs(record_charge - float(job_charge)) > 1e-6:
            raise PCF1CustodyError(f"PCF1 scheduler stage charge differs: {stage}")
        charged += float(job_charge)
    total = accounting.get("charged_gpu_seconds")
    if (
        isinstance(total, bool)
        or not isinstance(total, (int, float))
        or total <= 0
        or abs(float(total) - charged) > 1e-6
    ):
        raise PCF1CustodyError("PCF1 scheduler charged GPU accounting differs")


def authorize_score(args: argparse.Namespace) -> dict[str, Any]:
    """Authorize exactly one semantic read after all label-free evidence exists."""

    run_id = _run_id(args.run_id)
    environment, environment_sha, sandbox, sandbox_sha = _external_runtime_receipts(
        args
    )
    score_output_root = _safe_path(args.score_output_root, "score output root")
    canonical_score_output_root = score_output_root.resolve(strict=False)
    _safe_path(canonical_score_output_root, "resolved score output root")
    candidates_root = _explicit_root(args.candidates_root, "candidate root")
    if candidates_root in {Path("/"), Path.home().resolve()}:
        raise PCF1CustodyError("PCF1 candidate root is too broad")
    consumption_path = canonical_score_output_root.with_name(
        f"{canonical_score_output_root.name}.score-authorization-consumed.json"
    )
    if (
        score_output_root.exists()
        or score_output_root.is_symlink()
        or consumption_path.exists()
        or consumption_path.is_symlink()
    ):
        raise PCF1CustodyError("PCF1 score output root is not fresh")
    custody_specs = {
        "data_custody": (args.data_custody, DATA_CUSTODY_SCHEMA),
        "model_custody": (args.model_custody, MODEL_CUSTODY_SCHEMA),
        "runtime_custody": (args.runtime_custody, RUNTIME_CUSTODY_SCHEMA),
    }
    custody: dict[str, dict[str, Any]] = {}
    custody_hashes: dict[str, str] = {}
    for role, (path, schema) in custody_specs.items():
        report, digest = _load_report(path, schema=schema, label=role)
        if report.get("run_id") != run_id or report.get("custody_verified") is not True:
            raise PCF1CustodyError(f"PCF1 {role} run/verification differs")
        custody[role] = report
        custody_hashes[f"{role}_sha256"] = digest
    data_custody = custody["data_custody"]
    model_custody = custody["model_custody"]
    runtime_custody = custody["runtime_custody"]
    if (
        data_custody.get("holdout_sealed") is not True
        or data_custody.get("product_sealed") is not True
        or data_custody.get("public_sealed") is not True
        or data_custody.get("holdout_access_count") != 0
        or data_custody.get("product_access_count") != 0
        or data_custody.get("public_access_count") != 0
        or runtime_custody.get("environment_verified") is not True
        or runtime_custody.get("environment_receipt_sha256") != environment_sha
        or runtime_custody.get("environment_tree_sha256")
        != environment["environment_tree"]["sha256"]
        or runtime_custody.get("code_sandbox_verified") is not True
        or runtime_custody.get("code_sandbox_config_sha256")
        != sandbox["sandbox_config_sha256"]
        or runtime_custody.get("code_sandbox_binary_sha256") != SANDBOX_BINARY_SHA256
        or runtime_custody.get("code_sandbox_probe_sha256") != sandbox_sha
        or runtime_custody.get("code_sandbox_probe_result_sha256")
        != sandbox["probe_sha256"]
        or runtime_custody.get("code_sandbox_receipt_sha256") != sandbox_sha
        or model_custody.get("compute_host_verified") is not True
        or runtime_custody.get("compute_host_verified") is not True
        or model_custody.get("b1_compute_host_gate_verified") is not True
        or runtime_custody.get("b1_compute_host_gate_verified") is not True
        or not _sha256(model_custody.get("compute_host_receipt_sha256"))
        or runtime_custody.get("compute_host_receipt_sha256")
        != model_custody.get("compute_host_receipt_sha256")
        or runtime_custody.get("compute_host") != model_custody.get("compute_host")
        or model_custody.get("mbpp_calibration_setup_qualifications_verified")
        is not True
        or runtime_custody.get("mbpp_calibration_setup_qualifications_verified")
        is not True
        or runtime_custody.get("mbpp_calibration_allocation_setup_receipts")
        != model_custody.get("mbpp_calibration_allocation_setup_receipts")
    ):
        raise PCF1CustodyError("PCF1 protected data custody differs")

    confirmation_rows, confirmation_sha = _load_jsonl(
        args.confirmation_data, "authorization confirmation data"
    )
    confirmation_ids = [str(row.get("identity_sha256")) for row in confirmation_rows]
    if (
        len(confirmation_rows) != TOTAL_CONFIRMATION
        or any(
            row.get("schema") != EVAL_SCHEMA
            or row.get("split") != "confirmation"
            or not _sha256(row.get("identity_sha256"))
            or row.get("task") not in TASKS
            or "assessor" in row
            for row in confirmation_rows
        )
        or len(set(confirmation_ids)) != TOTAL_CONFIRMATION
        or _ordered_identity_sha256(confirmation_ids)
        != data_custody.get("identity_order_sha256")
        or confirmation_sha != data_custody.get("data_sha256")
    ):
        raise PCF1CustodyError("PCF1 authorization confirmation data differs")
    assessor_receipt, assessor_receipt_sha = _load_report(
        args.confirmation_assessor_receipt,
        schema=CONFIRMATION_ASSESSOR_RECEIPT_SCHEMA,
        label="authorization assessor receipt",
    )
    assessor_sha = assessor_receipt.get("board_sha256")
    if (
        not _sha256(assessor_sha)
        or assessor_sha != data_custody.get("confirmation_assessors_sha256")
        or assessor_receipt_sha
        != data_custody.get("confirmation_assessor_receipt_sha256")
        or assessor_receipt.get("rows") != TOTAL_CONFIRMATION
        or assessor_receipt.get("semantic_access") != "final_score_only"
    ):
        raise PCF1CustodyError("PCF1 authorization assessor hash differs")

    arm_reports: dict[str, str] = {}
    arm_candidates: dict[str, str] = {}
    for arm, report_path, candidates_path in (
        ("revision", args.revision_report, args.revision_candidates),
        ("unchanged", args.unchanged_report, args.unchanged_candidates),
        (
            "self_refinement",
            args.self_refinement_report,
            args.self_refinement_candidates,
        ),
    ):
        candidates_path = _explicit_file_under_root(
            candidates_path, candidates_root, f"authorization {arm} candidates"
        )
        report, report_sha = _load_report(
            report_path,
            schema=MERGED_EVALUATION_SCHEMA,
            label=f"authorization {arm} report",
        )
        candidates, candidates_sha = _load_jsonl(
            candidates_path, f"authorization {arm} candidates"
        )
        candidate_ids: list[str] = []
        if (
            report.get("arm") != arm
            or report.get("split") != "confirmation"
            or report.get("metrics") is not None
            or report.get("assessment_mode") != "confirmation_deferred"
            or report.get("assessor_board_access_count") != 0
            or report.get("full_row_count") != TOTAL_CONFIRMATION
            or report.get("exact_identity_coverage") is not True
            or report.get("candidates_sha256") != candidates_sha
            or not _recorded_path_matches(
                report.get("candidates_output"), candidates_path, f"{arm} candidates"
            )
        ):
            raise PCF1CustodyError(f"PCF1 authorization {arm} report differs")
        _require_access_zero(report, f"authorization {arm} report")
        for source, candidate in zip(confirmation_rows, candidates, strict=True):
            allowed = {
                "schema",
                "arm",
                "identity_sha256",
                "task",
                "completion",
                "generated_tokens",
                "max_token_exhausted",
            }
            if (
                set(candidate) != allowed
                or candidate.get("schema") != CANDIDATE_SCHEMA
                or candidate.get("arm") != arm
                or candidate.get("identity_sha256") != source.get("identity_sha256")
                or candidate.get("task") != source.get("task")
                or not isinstance(candidate.get("completion"), str)
                or isinstance(candidate.get("generated_tokens"), bool)
                or not isinstance(candidate.get("generated_tokens"), int)
                or not isinstance(candidate.get("max_token_exhausted"), bool)
            ):
                raise PCF1CustodyError(
                    f"PCF1 authorization {arm} candidate/order differs"
                )
            candidate_ids.append(str(candidate["identity_sha256"]))
        if candidate_ids != confirmation_ids:
            raise PCF1CustodyError(f"PCF1 authorization {arm} order differs")
        arm_reports[arm] = report_sha
        arm_candidates[arm] = candidates_sha

    pairs, pairs_sha = _load_jsonl(args.confirmation_pairs, "authorization pairs")
    if (
        len(pairs) != TOTAL_CONFIRMATION
        or [str(row.get("identity_sha256")) for row in pairs] != confirmation_ids
        or pairs_sha != data_custody.get("confirmation_pairs_sha256")
    ):
        raise PCF1CustodyError("PCF1 authorization pair order differs")
    pair_report, pair_report_sha = _load_report(
        args.confirmation_pair_report,
        schema=CONFIRMATION_PAIR_REPORT_SCHEMA,
        label="authorization pair report",
    )
    if pair_report.get(
        "output_sha256"
    ) != pairs_sha or pair_report_sha != data_custody.get(
        "confirmation_pair_report_sha256"
    ):
        raise PCF1CustodyError("PCF1 authorization pair receipt differs")

    selections, selections_sha = _load_jsonl(
        args.confirmation_selections, "authorization selections"
    )
    if [str(row.get("identity_sha256")) for row in selections] != confirmation_ids:
        raise PCF1CustodyError("PCF1 authorization selection order differs")
    application, application_sha = _load_report(
        args.commit_application_report,
        schema=COMMIT_APPLICATION_SCHEMA,
        label="authorization application report",
    )
    if (
        application.get("selections_sha256") != selections_sha
        or application.get("pairs_sha256") != pairs_sha
        or application.get("rows") != TOTAL_CONFIRMATION
        or application.get("assessor_board_access_count", 0) != 0
        or application.get("correctness_or_task_label_visible") is not False
    ):
        raise PCF1CustodyError("PCF1 authorization application differs")
    _require_access_zero(application, "authorization application report")
    training, training_sha = _load_report(
        args.commit_training_report,
        schema=COMMIT_TRAINING_SCHEMA,
        label="authorization commit training report",
    )
    mechanics, mechanics_sha = _load_report(
        args.mechanics_report,
        schema=MECHANICS_SCHEMA,
        status="pass",
        label="authorization mechanics report",
    )
    _require_access_zero(training, "authorization commit training report")
    _require_access_zero(mechanics, "authorization mechanics report")
    model_native = custody["model_custody"].get("native_report_sha256s", {})
    if (
        custody["model_custody"].get("commit_training_report_sha256") != training_sha
        or model_native.get("commit_application") != application_sha
        or model_native.get("confirmation_selections") != selections_sha
        or model_native.get("mechanics") != mechanics_sha
        or any(model_native.get(arm) != arm_reports[arm] for arm in NATIVE_ARMS)
    ):
        raise PCF1CustodyError("PCF1 authorization model custody differs")

    dispatch, dispatch_sha = _load_report(
        args.prescore_dispatch_receipt,
        schema=DISPATCH_SCHEMA,
        status="submitted",
        label="pre-score dispatch receipt",
    )
    accounting, accounting_sha = _load_report(
        args.prescore_accounting_receipt,
        schema=ACCOUNTING_SCHEMA,
        label="pre-score scheduler accounting receipt",
    )
    _accounting(dispatch, accounting, run_id, PRESCORE_ACCOUNTING_STAGES)

    result = {
        "schema": SCORE_AUTHORIZATION_SCHEMA,
        "status": "complete",
        "run_id": run_id,
        "scoring_authorized": True,
        "one_shot": True,
        "score_output_root": str(canonical_score_output_root),
        "candidates_root": str(candidates_root),
        "rows": TOTAL_CONFIRMATION,
        "identity_order_sha256": data_custody["identity_order_sha256"],
        "confirmation_data_sha256": confirmation_sha,
        "confirmation_assessors_sha256": assessor_sha,
        "confirmation_assessor_receipt_sha256": assessor_receipt_sha,
        "data_report_sha256": data_custody["data_report_sha256"],
        "arm_report_sha256s": arm_reports,
        "arm_candidates_sha256s": arm_candidates,
        "confirmation_pairs_sha256": pairs_sha,
        "confirmation_pair_report_sha256": pair_report_sha,
        "pair_report_sha256": pair_report_sha,
        "commit_application_report_sha256": application_sha,
        "selections_sha256": selections_sha,
        "commit_training_report_sha256": training_sha,
        "mechanics_report_sha256": mechanics_sha,
        "environment_receipt_sha256": environment_sha,
        "environment_tree_sha256": environment["environment_tree"]["sha256"],
        "code_sandbox_config_sha256": sandbox["sandbox_config_sha256"],
        "code_sandbox_binary_sha256": SANDBOX_BINARY_SHA256,
        "code_sandbox_probe_sha256": sandbox_sha,
        "code_sandbox_probe_result_sha256": sandbox["probe_sha256"],
        "code_sandbox_receipt_sha256": sandbox_sha,
        "compute_host_receipt_sha256": model_custody["compute_host_receipt_sha256"],
        "mbpp_calibration_setup_qualifications_verified": True,
        "mbpp_calibration_allocation_setup_receipts": runtime_custody[
            "mbpp_calibration_allocation_setup_receipts"
        ],
        **custody_hashes,
        "prescore_dispatch_receipt_sha256": dispatch_sha,
        "prescore_accounting_receipt_sha256": accounting_sha,
        "prescore_charged_gpu_seconds": accounting["charged_gpu_seconds"],
        "assessor_board_access_count_before": 0,
        "holdout_sealed": True,
        "product_sealed": True,
        "public_sealed": True,
        "holdout_access_count": 0,
        "product_access_count": 0,
        "public_access_count": 0,
        "sealed_access": SEALED_ACCESS,
    }
    _atomic_json(args.output, result)
    return result


def compute(args: argparse.Namespace) -> dict[str, Any]:
    run_id = _run_id(args.run_id)
    environment, environment_sha, sandbox, sandbox_sha = _external_runtime_receipts(
        args
    )
    custody_specs = {
        "data_custody": (args.data_custody, DATA_CUSTODY_SCHEMA),
        "model_custody": (args.model_custody, MODEL_CUSTODY_SCHEMA),
        "runtime_custody": (args.runtime_custody, RUNTIME_CUSTODY_SCHEMA),
    }
    custody: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for role, (path, schema) in custody_specs.items():
        report, digest = _load_report(path, schema=schema, label=role)
        if report.get("run_id") != run_id or report.get("custody_verified") is not True:
            raise PCF1CustodyError(f"PCF1 {role} run/verification differs")
        custody[role] = report
        hashes[f"{role}_sha256"] = digest
    if (
        custody["model_custody"].get("data_custody_sha256")
        != hashes["data_custody_sha256"]
        or custody["runtime_custody"].get("evaluation_accounting")
        != custody["model_custody"].get("evaluation_accounting")
        or custody["runtime_custody"].get("evaluation_accounting_sha256")
        != hashlib.sha256(
            json.dumps(
                custody["model_custody"].get("evaluation_accounting"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        or custody["runtime_custody"].get("environment_verified") is not True
        or custody["runtime_custody"].get("environment_receipt_sha256")
        != environment_sha
        or custody["runtime_custody"].get("environment_tree_sha256")
        != environment["environment_tree"]["sha256"]
        or custody["runtime_custody"].get("code_sandbox_verified") is not True
        or custody["runtime_custody"].get("code_sandbox_config_sha256")
        != sandbox["sandbox_config_sha256"]
        or custody["runtime_custody"].get("code_sandbox_binary_sha256")
        != SANDBOX_BINARY_SHA256
        or custody["runtime_custody"].get("code_sandbox_probe_sha256") != sandbox_sha
        or custody["runtime_custody"].get("code_sandbox_probe_result_sha256")
        != sandbox["probe_sha256"]
        or custody["runtime_custody"].get("code_sandbox_receipt_sha256") != sandbox_sha
        or custody["model_custody"].get("compute_host_verified") is not True
        or custody["runtime_custody"].get("compute_host_verified") is not True
        or custody["model_custody"].get("b1_compute_host_gate_verified") is not True
        or custody["runtime_custody"].get("b1_compute_host_gate_verified") is not True
        or not _sha256(custody["model_custody"].get("compute_host_receipt_sha256"))
        or custody["runtime_custody"].get("compute_host_receipt_sha256")
        != custody["model_custody"].get("compute_host_receipt_sha256")
        or custody["runtime_custody"].get("compute_host")
        != custody["model_custody"].get("compute_host")
        or custody["model_custody"].get(
            "mbpp_calibration_setup_qualifications_verified"
        )
        is not True
        or custody["runtime_custody"].get(
            "mbpp_calibration_setup_qualifications_verified"
        )
        is not True
        or custody["runtime_custody"].get("mbpp_calibration_allocation_setup_receipts")
        != custody["model_custody"].get("mbpp_calibration_allocation_setup_receipts")
    ):
        raise PCF1CustodyError("PCF1 model/data custody binding differs")

    consumption, consumption_sha = _load_report(
        args.score_consumption,
        schema=SCORE_CONSUMPTION_SCHEMA,
        label="score consumption marker",
    )
    score_sandbox, score_sandbox_sha = _load_report(
        args.score_sandbox_probe,
        schema=SANDBOX_RECEIPT_SCHEMA,
        status="pass",
        label="final score sandbox probe",
    )
    if (
        consumption.get("claim_state") != "consumed"
        or consumption.get("run_id") != run_id
        or consumption.get("identity_order_sha256")
        != custody["data_custody"].get("identity_order_sha256")
        or consumption.get("rows") != TOTAL_CONFIRMATION
        or consumption.get("semantic_read_budget") != 1
        or not _sha256(consumption.get("score_authorization_sha256"))
        or not _sha256(consumption.get("confirmation_assessors_sha256"))
        or not isinstance(consumption.get("score_output_root"), str)
        or not consumption["score_output_root"]
        or score_sandbox != sandbox
        or score_sandbox_sha != sandbox_sha
        or consumption.get("sandbox_probe_sha256") != score_sandbox_sha
        or consumption.get("sandbox_probe_result_sha256") != sandbox["probe_sha256"]
        or consumption.get("sandbox_receipt_sha256") != score_sandbox_sha
    ):
        raise PCF1CustodyError("PCF1 score consumption marker differs")

    normalized = _explicit_root(args.normalized_root, "normalized root")
    arm_hashes: dict[str, str] = {}
    final_setup_receipts_sha256: str | None = None
    final_setup_receipt_count: int | None = None
    for arm in NORMALIZED_ARMS:
        report, digest = _load_report(
            normalized / f"{arm}.json",
            schema=ARM_REPORT_SCHEMA,
            label=f"normalized {arm} report",
        )
        if (
            report.get("arm") != arm
            or report.get("run_id") != run_id
            or report.get("split") != "development"
            or report.get("full_row_count") != TOTAL_CONFIRMATION
            or report.get("identity_order_sha256")
            != custody["data_custody"].get("identity_order_sha256")
            or report.get("data_sha256") != custody["data_custody"].get("data_sha256")
            or report.get("model_revision")
            != custody["model_custody"].get("model_revision")
            or report.get("runtime_sha256")
            != custody["runtime_custody"].get("runtime_sha256")
            or report.get("custody") != hashes
            or report.get("score_consumption_sha256") != consumption_sha
            or report.get("score_consumption_state") != "consumed"
            or report.get("environment_receipt_sha256") != environment_sha
            or report.get("environment_tree_sha256")
            != environment["environment_tree"]["sha256"]
            or report.get("code_sandbox_config_sha256")
            != sandbox["sandbox_config_sha256"]
            or report.get("code_sandbox_binary_sha256") != SANDBOX_BINARY_SHA256
            or report.get("code_sandbox_probe_sha256") != score_sandbox_sha
            or report.get("sandbox_receipt_sha256") != score_sandbox_sha
            or report.get("one_open_verified") is not True
            or report.get("mbpp_final_score_setup_qualifications_verified") is not True
            or not _sha256(
                report.get("mbpp_final_score_allocation_setup_receipts_sha256")
            )
            or isinstance(
                report.get("mbpp_final_score_allocation_setup_receipt_count"), bool
            )
            or not isinstance(
                report.get("mbpp_final_score_allocation_setup_receipt_count"), int
            )
            or report.get("mbpp_final_score_allocation_setup_receipt_count", 0) <= 0
        ):
            raise PCF1CustodyError(f"normalized {arm} custody binding differs")
        setup_digest = str(report["mbpp_final_score_allocation_setup_receipts_sha256"])
        setup_count = int(report["mbpp_final_score_allocation_setup_receipt_count"])
        if final_setup_receipts_sha256 is None:
            final_setup_receipts_sha256 = setup_digest
            final_setup_receipt_count = setup_count
        elif (
            setup_digest != final_setup_receipts_sha256
            or setup_count != final_setup_receipt_count
        ):
            raise PCF1CustodyError("normalized MBPP setup custody differs")
        setup_receipts = report.get("mbpp_final_score_allocation_setup_receipts")
        if arm != "learned_commit":
            if setup_receipts is not None:
                raise PCF1CustodyError("normalized arm exposes final setup receipts")
        else:
            if (
                not isinstance(setup_receipts, list)
                or len(setup_receipts) != setup_count
                or mbpp_allocation_setup_receipts_sha256(setup_receipts) != setup_digest
            ):
                raise PCF1CustodyError("normalized final setup receipts differ")
            seen_setup_hashes: set[str] = set()
            for setup_receipt in setup_receipts:
                if not isinstance(setup_receipt, dict):
                    raise PCF1CustodyError("normalized final setup receipt differs")
                try:
                    validate_mbpp_setup_qualification_receipt(
                        setup_receipt,
                        allocation_probe_sha256=str(score_sandbox["probe_sha256"]),
                    )
                except PCF1SandboxError as error:
                    raise PCF1CustodyError(
                        "normalized final setup receipt differs"
                    ) from error
                setup_hash = str(setup_receipt["setup_source_sha256"])
                if setup_hash in seen_setup_hashes:
                    raise PCF1CustodyError(
                        "normalized final setup receipt is duplicated"
                    )
                seen_setup_hashes.add(setup_hash)
        arm_hashes[arm] = digest

    dispatch, dispatch_sha = _load_report(
        args.dispatch_receipt,
        schema=DISPATCH_SCHEMA,
        status="submitted",
        label="dispatch receipt",
    )
    accounting, accounting_sha = _load_report(
        args.accounting_receipt,
        schema=ACCOUNTING_SCHEMA,
        label="scheduler accounting receipt",
    )
    _accounting(dispatch, accounting, run_id, FINAL_ACCOUNTING_STAGES)
    result = {
        "schema": COMPUTE_CUSTODY_SCHEMA,
        "status": "complete",
        "run_id": run_id,
        "custody_verified": True,
        **hashes,
        "arm_report_sha256s": arm_hashes,
        "dispatch_receipt_sha256": dispatch_sha,
        "scheduler_accounting_receipt_sha256": accounting_sha,
        "score_consumption_sha256": consumption_sha,
        "score_consumption_state": "consumed",
        "one_open_verified": True,
        "environment_verified": True,
        "environment_receipt_sha256": environment_sha,
        "environment_tree_sha256": environment["environment_tree"]["sha256"],
        "code_sandbox_verified": True,
        "code_sandbox_config_sha256": sandbox["sandbox_config_sha256"],
        "code_sandbox_binary_sha256": SANDBOX_BINARY_SHA256,
        "code_sandbox_probe_sha256": sandbox_sha,
        "code_sandbox_probe_result_sha256": sandbox["probe_sha256"],
        "code_sandbox_receipt_sha256": sandbox_sha,
        "score_sandbox_receipt_sha256": score_sandbox_sha,
        "mbpp_final_score_setup_qualifications_verified": True,
        "mbpp_final_score_allocation_setup_receipt_count": (final_setup_receipt_count),
        "mbpp_final_score_allocation_setup_receipts_sha256": (
            final_setup_receipts_sha256
        ),
        "mbpp_calibration_setup_qualifications_verified": True,
        "mbpp_calibration_allocation_setup_receipts": custody["runtime_custody"][
            "mbpp_calibration_allocation_setup_receipts"
        ],
        "compute_host_verified": True,
        "compute_host_receipt_sha256": custody["model_custody"][
            "compute_host_receipt_sha256"
        ],
        "b1_compute_host_gate_verified": True,
        "evaluation_accounting_sha256": custody["runtime_custody"][
            "evaluation_accounting_sha256"
        ],
        "charged_gpu_seconds": accounting["charged_gpu_seconds"],
        "partition": "normal",
        "excluded_nodes": EXCLUDED_NODES,
        "retry_count": 0,
        "successor_authorized": False,
        "successor_submitted": False,
        "accounting_verified": True,
        "scheduler_accounting_verified": True,
        "charged_resource_accounting_verified": True,
        "holdout_sealed": True,
        "product_sealed": True,
        "public_sealed": True,
        "holdout_access_count": 0,
        "product_access_count": 0,
        "public_access_count": 0,
        "sealed_access": SEALED_ACCESS,
    }
    _atomic_json(args.output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    pre = modes.add_parser("precompute")
    pre.add_argument("--run-id", required=True)
    pre.add_argument("--source-freeze-report", type=Path, required=True)
    pre.add_argument("--train-sources", type=Path, required=True)
    pre.add_argument("--development-sources", type=Path, required=True)
    pre.add_argument("--reference-preflight-rows", type=Path, required=True)
    pre.add_argument("--reference-sandbox-receipt", type=Path, required=True)
    pre.add_argument("--merged-drafts", type=Path, required=True)
    pre.add_argument("--merged-drafts-report", type=Path, required=True)
    pre.add_argument("--revision-training-data", type=Path, required=True)
    pre.add_argument("--calibration-data", type=Path, required=True)
    pre.add_argument("--confirmation-data", type=Path, required=True)
    pre.add_argument("--confirmation-assessor-receipt", type=Path, required=True)
    pre.add_argument("--data-report", type=Path, required=True)
    pre.add_argument("--calibration-pairs", type=Path, required=True)
    pre.add_argument("--calibration-pair-report", type=Path, required=True)
    pre.add_argument("--confirmation-pairs", type=Path, required=True)
    pre.add_argument("--confirmation-pair-report", type=Path, required=True)
    pre.add_argument("--calibration-revision-report", type=Path, required=True)
    pre.add_argument("--calibration-unchanged-report", type=Path, required=True)
    pre.add_argument(
        "--calibration-revision-sandbox-probe",
        dest="calibration_revision_sandbox_probes",
        action="append",
        type=Path,
        required=True,
    )
    pre.add_argument(
        "--calibration-unchanged-sandbox-probe",
        dest="calibration_unchanged_sandbox_probes",
        action="append",
        type=Path,
        required=True,
    )
    pre.add_argument("--revision-report", type=Path, required=True)
    pre.add_argument("--unchanged-report", type=Path, required=True)
    pre.add_argument("--self-refinement-report", type=Path, required=True)
    pre.add_argument("--b1-checkpoint", type=Path, required=True)
    pre.add_argument("--b1-training-report", type=Path, required=True)
    pre.add_argument("--revision-checkpoint", type=Path, required=True)
    pre.add_argument("--revision-training-report", type=Path, required=True)
    pre.add_argument("--commit-checkpoint", type=Path, required=True)
    pre.add_argument("--commit-training-report", type=Path, required=True)
    pre.add_argument("--commit-application-report", type=Path, required=True)
    pre.add_argument("--confirmation-selections", type=Path, required=True)
    pre.add_argument("--mechanics-report", type=Path, required=True)
    pre.add_argument("--compute-host-receipt", type=Path, required=True)
    pre.add_argument("--model-root", type=Path, required=True)
    pre.add_argument("--model-revision", default=MODEL_REVISION)
    pre.add_argument("--model-manifest", type=Path, required=True)
    pre.add_argument("--model-manifest-sha256", required=True)
    pre.add_argument("--model-config-sha256", required=True)
    pre.add_argument("--runtime-root", type=Path, required=True)
    pre.add_argument("--runtime-manifest", type=Path, required=True)
    pre.add_argument("--runtime-manifest-sha256", required=True)
    pre.add_argument("--environment-receipt", type=Path, required=True)
    pre.add_argument("--environment-receipt-sha256", required=True)
    pre.add_argument("--sandbox-receipt", type=Path, required=True)
    pre.add_argument("--sandbox-receipt-sha256", required=True)
    pre.add_argument("--output", type=Path, required=True)

    authorize = modes.add_parser("authorize-score")
    authorize.add_argument("--run-id", required=True)
    authorize.add_argument("--confirmation-data", type=Path, required=True)
    authorize.add_argument("--confirmation-assessor-receipt", type=Path, required=True)
    authorize.add_argument("--revision-report", type=Path, required=True)
    authorize.add_argument("--revision-candidates", type=Path, required=True)
    authorize.add_argument("--unchanged-report", type=Path, required=True)
    authorize.add_argument("--unchanged-candidates", type=Path, required=True)
    authorize.add_argument("--self-refinement-report", type=Path, required=True)
    authorize.add_argument("--self-refinement-candidates", type=Path, required=True)
    authorize.add_argument("--candidates-root", type=Path, required=True)
    authorize.add_argument("--confirmation-pairs", type=Path, required=True)
    authorize.add_argument("--confirmation-pair-report", type=Path, required=True)
    authorize.add_argument("--confirmation-selections", type=Path, required=True)
    authorize.add_argument("--commit-application-report", type=Path, required=True)
    authorize.add_argument("--commit-training-report", type=Path, required=True)
    authorize.add_argument("--mechanics-report", type=Path, required=True)
    authorize.add_argument("--data-custody", type=Path, required=True)
    authorize.add_argument("--model-custody", type=Path, required=True)
    authorize.add_argument("--runtime-custody", type=Path, required=True)
    authorize.add_argument("--prescore-dispatch-receipt", type=Path, required=True)
    authorize.add_argument("--prescore-accounting-receipt", type=Path, required=True)
    authorize.add_argument("--environment-receipt", type=Path, required=True)
    authorize.add_argument("--environment-receipt-sha256", required=True)
    authorize.add_argument("--sandbox-receipt", type=Path, required=True)
    authorize.add_argument("--sandbox-receipt-sha256", required=True)
    authorize.add_argument("--score-output-root", type=Path, required=True)
    authorize.add_argument("--output", type=Path, required=True)

    final = modes.add_parser("compute")
    final.add_argument("--run-id", required=True)
    final.add_argument("--data-custody", type=Path, required=True)
    final.add_argument("--model-custody", type=Path, required=True)
    final.add_argument("--runtime-custody", type=Path, required=True)
    final.add_argument("--normalized-root", type=Path, required=True)
    final.add_argument("--dispatch-receipt", type=Path, required=True)
    final.add_argument("--accounting-receipt", type=Path, required=True)
    final.add_argument("--score-consumption", type=Path, required=True)
    final.add_argument("--score-sandbox-probe", type=Path, required=True)
    final.add_argument("--environment-receipt", type=Path, required=True)
    final.add_argument("--environment-receipt-sha256", required=True)
    final.add_argument("--sandbox-receipt", type=Path, required=True)
    final.add_argument("--sandbox-receipt-sha256", required=True)
    final.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    builders = {
        "precompute": precompute,
        "authorize-score": authorize_score,
        "compute": compute,
    }
    result = builders[args.mode](args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
