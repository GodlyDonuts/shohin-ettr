#!/usr/bin/env python3
"""Normalize native PCF1 reports for the frozen final comparison.

The normalizer reads only the eight JSON reports named on its command line.
It never follows a path recorded inside a report.  Its four outputs are an
atomic, write-once directory and carry hashes of the exact custody-report
bytes used to normalize them.  Compute custody is intentionally downstream
of this program because it must bind the hashes of these outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from pcf1_code_sandbox import (
    PCF1SandboxError,
    mbpp_allocation_setup_receipts_sha256,
    validate_mbpp_setup_qualification_receipt,
)

ARM_SCHEMA = "shohin-pcf1-arm-report-v1"
MERGED_EVALUATION_SCHEMA = "shohin-pcf1-merged-evaluation-v1"
COMMIT_SCHEMA = "shohin-pcf1-commit-result-v1"
CONSUMPTION_SCHEMA = "shohin-pcf1-score-consumption-v1"
CUSTODY_SCHEMAS = {
    "data_custody": "shohin-pcf1-data-custody-v1",
    "model_custody": "shohin-pcf1-model-custody-v1",
    "runtime_custody": "shohin-pcf1-runtime-custody-v1",
}
TOTAL_ROWS = 1289
DOMAINS = ("math500", "bbh_logic", "mbpp")
PINNED_MODEL_REVISION = "81eaece1948f3875421d9a45bc55487d10e2d894"
NATIVE_TO_NORMALIZED = {
    "revision": "trained_revision",
    "unchanged": "unchanged",
    "self_refinement": "self_refinement",
}
OUTPUT_NAMES = {
    "learned_commit": "learned_commit.json",
    "trained_revision": "trained_revision.json",
    "unchanged": "unchanged.json",
    "self_refinement": "self_refinement.json",
}


class PCF1NormalizationError(RuntimeError):
    """Native PCF1 evidence cannot be normalized without changing its meaning."""


def _path_arg(args: argparse.Namespace, *names: str) -> Path:
    for name in names:
        value = getattr(args, name, None)
        if value is not None:
            return Path(value)
    raise PCF1NormalizationError(f"missing PCF1 normalization input: {names[0]}")


def _load_complete(
    path: Path, *, schema: str, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    rendered = f"{path}\n{path.resolve(strict=False)}".casefold()
    if any(term in rendered for term in ("holdout", "product", "public")):
        raise PCF1NormalizationError(f"protected PCF1 {label} report: {path}")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PCF1NormalizationError(
            f"unreadable PCF1 {label} report: {path}"
        ) from error
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get("status") != "complete"
    ):
        raise PCF1NormalizationError(f"incomplete PCF1 {label} report: {path}")
    return value, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PCF1NormalizationError(f"invalid PCF1 count: {label}")
    return value


def _sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _final_setup_receipts(report: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    receipts = report.get("mbpp_allocation_setup_receipts")
    count = report.get("mbpp_allocation_setup_receipt_count")
    digest = report.get("mbpp_allocation_setup_receipts_sha256")
    allocation_probe_sha256 = report.get("code_sandbox_probe_result_sha256")
    if (
        report.get("mbpp_allocation_setup_status") != "passed"
        or not isinstance(receipts, list)
        or not receipts
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count != len(receipts)
        or not _sha256(digest)
        or not _sha256(allocation_probe_sha256)
        or digest != mbpp_allocation_setup_receipts_sha256(receipts)
    ):
        raise PCF1NormalizationError("PCF1 final MBPP setup qualification differs")
    setup_hashes: set[str] = set()
    for receipt in receipts:
        if not isinstance(receipt, dict):
            raise PCF1NormalizationError("PCF1 final MBPP setup receipt differs")
        try:
            validate_mbpp_setup_qualification_receipt(
                receipt,
                allocation_probe_sha256=str(allocation_probe_sha256),
            )
        except PCF1SandboxError as error:
            raise PCF1NormalizationError(
                "PCF1 final MBPP setup receipt differs"
            ) from error
        setup_hash = str(receipt["setup_source_sha256"])
        if setup_hash in setup_hashes:
            raise PCF1NormalizationError("PCF1 final MBPP setup receipt is duplicated")
        setup_hashes.add(setup_hash)
    return receipts, str(digest)


def _metric(report: dict[str, Any], domain: str, correct_field: str) -> dict[str, int]:
    metric = report.get("metrics", {}).get(domain)
    if not isinstance(metric, dict):
        raise PCF1NormalizationError(f"missing PCF1 native metric: {domain}")
    correct = _integer(metric.get(correct_field), f"{domain}.{correct_field}")
    total = _integer(metric.get("total"), f"{domain}.total")
    if correct > total:
        raise PCF1NormalizationError(f"impossible PCF1 native metric: {domain}")
    return {"generated_correct": correct, "total": total}


def _deferred_arm_metric(
    score_report: dict[str, Any], arm: str, domain: str
) -> dict[str, int]:
    arm_metrics = score_report.get("arm_metrics")
    metrics = arm_metrics.get(arm) if isinstance(arm_metrics, dict) else None
    metric = metrics.get(domain) if isinstance(metrics, dict) else None
    if not isinstance(metric, dict):
        raise PCF1NormalizationError(f"missing PCF1 deferred metric: {arm}.{domain}")
    correct = _integer(
        metric.get("generated_correct"), f"{arm}.{domain}.generated_correct"
    )
    total = _integer(metric.get("total"), f"{arm}.{domain}.total")
    if correct > total:
        raise PCF1NormalizationError(f"impossible PCF1 deferred metric: {arm}.{domain}")
    return {"generated_correct": correct, "total": total}


def _commit_metric(report: dict[str, Any], domain: str) -> dict[str, int]:
    confirmation = report.get("confirmation")
    metric = confirmation.get(domain) if isinstance(confirmation, dict) else None
    required = (
        "total",
        "revision_correct",
        "unchanged_correct",
        "selected_correct",
        "order_consistent",
        "revision_correct_retained",
        "unchanged_correct_retained",
    )
    if not isinstance(metric, dict):
        raise PCF1NormalizationError(f"missing PCF1 commit metric: {domain}")
    values = {name: _integer(metric.get(name), f"{domain}.{name}") for name in required}
    if any(values[name] > values["total"] for name in required if name != "total"):
        raise PCF1NormalizationError(f"impossible PCF1 commit metric: {domain}")
    return values


def _sealed_access_zero(report: dict[str, Any]) -> bool:
    return report.get("sealed_access") == {
        "holdout": 0,
        "product": 0,
        "public": 0,
    }


def _write_json(path: Path, value: dict[str, Any]) -> str:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(encoded).hexdigest()


def _publish_directory(
    output: Path, reports: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    if output.exists() or output.is_symlink():
        raise PCF1NormalizationError(f"refusing existing PCF1 output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise PCF1NormalizationError(
            f"refusing existing PCF1 temporary output: {temporary}"
        )
    temporary.mkdir()
    receipts: dict[str, dict[str, Any]] = {}
    try:
        for arm, report in reports.items():
            name = OUTPUT_NAMES[arm]
            digest = _write_json(temporary / name, report)
            receipts[arm] = {
                "path": str((output / name).resolve()),
                "sha256": digest,
            }
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(temporary, output)
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        if temporary.exists() and temporary.is_dir():
            shutil.rmtree(temporary)
        raise
    return receipts


def normalize(args: argparse.Namespace) -> dict[str, Any]:
    native_paths = {
        "learned_commit": _path_arg(args, "learned_commit_report", "commit_report"),
        "revision": _path_arg(args, "revision_report", "trained_revision_report"),
        "unchanged": _path_arg(args, "unchanged_report"),
        "self_refinement": _path_arg(args, "self_refinement_report"),
    }
    custody_paths = {
        "data_custody": _path_arg(args, "data_custody", "data_custody_report"),
        "model_custody": _path_arg(args, "model_custody", "model_custody_report"),
        "runtime_custody": _path_arg(args, "runtime_custody", "runtime_custody_report"),
    }
    output = _path_arg(args, "output", "output_dir")
    consumption_path = _path_arg(args, "score_consumption")
    output_rendered = f"{output}\n{output.resolve(strict=False)}".casefold()
    if any(term in output_rendered for term in ("holdout", "product", "public")):
        raise PCF1NormalizationError(f"protected PCF1 normalized output: {output}")

    native: dict[str, dict[str, Any]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for arm, path in native_paths.items():
        schema = COMMIT_SCHEMA if arm == "learned_commit" else MERGED_EVALUATION_SCHEMA
        report, receipt = _load_complete(path, schema=schema, label=arm)
        native[arm] = report
        receipts[arm] = receipt
    for role, path in custody_paths.items():
        report, receipt = _load_complete(path, schema=CUSTODY_SCHEMAS[role], label=role)
        native[role] = report
        receipts[role] = receipt
    consumption, consumption_receipt = _load_complete(
        consumption_path,
        schema=CONSUMPTION_SCHEMA,
        label="score consumption",
    )
    receipts["score_consumption"] = consumption_receipt

    evaluations = {arm: native[arm] for arm in NATIVE_TO_NORMALIZED}
    for arm, report in evaluations.items():
        if (
            report.get("arm") != arm
            or report.get("split") != "confirmation"
            or report.get("full_row_count") != TOTAL_ROWS
            or report.get("exact_identity_coverage") is not True
            or report.get("metrics") is not None
            or report.get("assessment_mode") != "confirmation_deferred"
            or report.get("assessor_board_access_count") != 0
            or not _sealed_access_zero(report)
            or not _sha256(report.get("candidates_sha256"))
            or not _sha256(report.get("adapter_checkpoint_sha256"))
            or not _sha256(report.get("adapter_metadata_sha256"))
            or isinstance(report.get("trainable_parameters"), bool)
            or not isinstance(report.get("trainable_parameters"), int)
            or report.get("trainable_parameters", 0) <= 0
            or not _sha256(report.get("trainable_parameter_name_sha256"))
            or report.get("lora_layer_indices") != [30, 31, 32, 33]
            or report.get("runtime_fields")
            != (
                ["source_prompt", "internal_draft.completion"]
                if arm == "self_refinement"
                else ["question"]
            )
            or report.get("environment_verified") is not True
            or not _sha256(report.get("environment_receipt_sha256"))
            or not _sha256(report.get("environment_tree_sha256"))
            or not _sha256(report.get("code_sandbox_config_sha256"))
            or not _sha256(report.get("code_sandbox_binary_sha256"))
            or report.get("code_sandbox_status") != "not_applicable_no_code_scoring"
            or report.get("code_sandbox_probe_passed") is not None
            or report.get("code_sandbox_probe_sha256") is not None
            or report.get("code_sandbox_probe_result_sha256") is not None
            or report.get("sandbox_receipt_sha256") is not None
        ):
            raise PCF1NormalizationError(f"PCF1 native evaluation differs: {arm}")
        counters = report.get("counters")
        if (
            not isinstance(counters, dict)
            or _integer(counters.get("rows"), f"{arm}.rows") != TOTAL_ROWS
            or _integer(counters.get("prompt_tokens"), f"{arm}.prompt_tokens") <= 0
            or report.get("aggregate_prompt_tokens") != counters.get("prompt_tokens")
            or isinstance(report.get("aggregate_wall_seconds"), bool)
            or not isinstance(report.get("aggregate_wall_seconds"), (int, float))
            or report.get("aggregate_wall_seconds", -1) < 0
            or report.get("aggregate_gpu_seconds")
            != report.get("aggregate_wall_seconds")
            or isinstance(report.get("maximum_peak_gpu_memory_bytes"), bool)
            or not isinstance(report.get("maximum_peak_gpu_memory_bytes"), int)
            or report.get("maximum_peak_gpu_memory_bytes", -1) < 0
        ):
            raise PCF1NormalizationError(f"PCF1 native coverage differs: {arm}")

    common_keys = (
        "model_root",
        "model_revision",
        "model_loader",
        "data_sha256",
        "data_report_sha256",
        "generation_mode",
        "max_new_tokens",
        "seed",
        "batch_size",
        "shard_count",
        "full_row_count",
        "trainable_parameters",
        "trainable_parameter_name_sha256",
        "lora_layer_indices",
        "environment_receipt_sha256",
        "environment_tree_sha256",
        "code_sandbox_config_sha256",
        "code_sandbox_binary_sha256",
    )
    revision = evaluations["revision"]
    if any(
        report.get(key) != revision.get(key)
        for report in evaluations.values()
        for key in common_keys
    ):
        raise PCF1NormalizationError("PCF1 matched native evaluations differ")

    data = native["data_custody"]
    model = native["model_custody"]
    runtime = native["runtime_custody"]
    commit = native["learned_commit"]
    final_setup_receipts, final_setup_receipts_sha256 = _final_setup_receipts(commit)
    arm_malformed = commit.get("arm_malformed")
    arm_policy_rejected = commit.get("arm_capability_policy_rejected")
    if (
        not isinstance(arm_malformed, dict)
        or set(arm_malformed) != set(NATIVE_TO_NORMALIZED)
        or not isinstance(arm_policy_rejected, dict)
        or set(arm_policy_rejected) != set(NATIVE_TO_NORMALIZED)
        or any(
            _integer(arm_malformed[arm], f"{arm}.malformed")
            < _integer(arm_policy_rejected[arm], f"{arm}.policy_rejected")
            for arm in NATIVE_TO_NORMALIZED
        )
        or any(
            not (
                max(
                    _integer(
                        evaluations[arm]["counters"].get("empty_completions"),
                        f"{arm}.empty_completions",
                    ),
                    _integer(arm_policy_rejected[arm], f"{arm}.policy_rejected"),
                )
                <= _integer(arm_malformed[arm], f"{arm}.malformed")
                <= _integer(
                    evaluations[arm]["counters"].get("empty_completions"),
                    f"{arm}.empty_completions",
                )
                + _integer(arm_policy_rejected[arm], f"{arm}.policy_rejected")
            )
            for arm in NATIVE_TO_NORMALIZED
        )
        or _integer(
            commit.get("confirmation_malformed_candidates"),
            "confirmation_malformed_candidates",
        )
        != sum(int(arm_malformed[arm]) for arm in NATIVE_TO_NORMALIZED)
        or _integer(
            commit.get("confirmation_capability_policy_rejections"),
            "confirmation_capability_policy_rejections",
        )
        != sum(int(arm_policy_rejected[arm]) for arm in NATIVE_TO_NORMALIZED)
        or _integer(
            commit.get("selected_capability_policy_rejections"),
            "selected_capability_policy_rejections",
        )
        > _integer(
            commit.get("confirmation_malformed_selections"),
            "confirmation_malformed_selections",
        )
    ):
        raise PCF1NormalizationError("PCF1 capability-policy counts differ")
    run_ids = [report.get("run_id") for report in (data, model, runtime)]
    if (
        not all(isinstance(value, str) and value for value in run_ids)
        or len(set(run_ids)) != 1
    ):
        raise PCF1NormalizationError("PCF1 custody run identifiers differ")
    run_id = str(run_ids[0])
    expected_confirmation_accounting = {
        arm: {
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
        for arm, report in evaluations.items()
    }
    model_accounting = model.get("evaluation_accounting")
    runtime_accounting = runtime.get("evaluation_accounting")
    runtime_accounting_sha256 = runtime.get("evaluation_accounting_sha256")
    if (
        not isinstance(model_accounting, dict)
        or runtime_accounting != model_accounting
        or not _sha256(runtime_accounting_sha256)
        or runtime_accounting_sha256
        != hashlib.sha256(
            json.dumps(
                runtime_accounting, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        or any(
            model_accounting.get(arm) != receipt
            for arm, receipt in expected_confirmation_accounting.items()
        )
    ):
        raise PCF1NormalizationError("PCF1 evaluation accounting custody differs")
    expected_score_root = native_paths["learned_commit"].resolve().parent
    commit_inputs = commit.get("inputs")
    if (
        consumption.get("claim_state") != "consumed"
        or consumption.get("run_id") != run_id
        or consumption.get("score_output_root") != str(expected_score_root)
        or consumption.get("score_authorization_sha256")
        != commit.get("score_authorization_sha256")
        or consumption.get("confirmation_assessors_sha256")
        != (
            commit_inputs.get("confirmation_assessors_sha256")
            if isinstance(commit_inputs, dict)
            else None
        )
        or consumption.get("identity_order_sha256") != data.get("identity_order_sha256")
        or consumption.get("rows") != TOTAL_ROWS
        or consumption.get("semantic_read_budget") != 1
        or consumption.get("sandbox_probe_sha256")
        != commit.get("code_sandbox_probe_sha256")
        or consumption.get("sandbox_receipt_sha256")
        != commit.get("sandbox_receipt_sha256")
        or consumption.get("sandbox_probe_result_sha256")
        != commit.get("code_sandbox_probe_result_sha256")
        or commit.get("score_consumption") != str(consumption_path.resolve())
        or commit.get("score_consumption_sha256") != consumption_receipt["sha256"]
        or commit.get("score_consumption_state") != "consumed"
        or commit.get("authorization_consumed") is not True
    ):
        raise PCF1NormalizationError("PCF1 score consumption binding differs")

    data_hash_fields = (
        "data_sha256",
        "data_report_sha256",
        "identity_order_sha256",
        "confirmation_pairs_sha256",
    )
    if (
        not all(_sha256(data.get(name)) for name in data_hash_fields)
        or data.get("confirmation_rows") != TOTAL_ROWS
        or data.get("data_sha256") != revision.get("data_sha256")
        or data.get("data_report_sha256") != revision.get("data_report_sha256")
        or commit.get("pairs_sha256") != data.get("confirmation_pairs_sha256")
    ):
        raise PCF1NormalizationError("PCF1 data custody binding differs")

    checkpoint_hashes = model.get("checkpoint_sha256s")
    expected_checkpoint_hashes = {
        "trained_revision": evaluations["revision"].get("adapter_checkpoint_sha256"),
        "unchanged": evaluations["unchanged"].get("adapter_checkpoint_sha256"),
        "self_refinement": evaluations["self_refinement"].get(
            "adapter_checkpoint_sha256"
        ),
        "learned_commit_host": commit.get("adapter_checkpoint_sha256"),
        "learned_commit": commit.get("checkpoint_sha256"),
    }
    if (
        not isinstance(checkpoint_hashes, dict)
        or checkpoint_hashes != expected_checkpoint_hashes
        or not all(_sha256(value) for value in checkpoint_hashes.values())
        or checkpoint_hashes["unchanged"] != checkpoint_hashes["self_refinement"]
        or checkpoint_hashes["unchanged"] != checkpoint_hashes["learned_commit_host"]
        or model.get("model_revision") != revision.get("model_revision")
        or model.get("model_revision") != PINNED_MODEL_REVISION
        or model.get("model_root") != revision.get("model_root")
        or commit.get("model_revision") != model.get("model_revision")
        or commit.get("model_root") != model.get("model_root")
        or not _sha256(model.get("commit_training_report_sha256"))
        or commit.get("training_report_sha256")
        != model.get("commit_training_report_sha256")
        or commit.get("protected_adapter_unchanged") is not True
        or not _sha256(commit.get("selections_sha256"))
        or not _sealed_access_zero(commit)
        or commit.get("assessor_board_semantic_reads") != 1
        or commit.get("confirmation_open_count") != 1
        or commit.get("assessment_calls") != TOTAL_ROWS * 3
        or commit.get("outcome_rows") != TOTAL_ROWS
        or not _sha256(commit.get("outcomes_sha256"))
        or not _sha256(commit.get("score_authorization_sha256"))
        or not _sha256(commit.get("score_consumption_sha256"))
        or commit.get("environment_verified") is not True
        or commit.get("environment_receipt_sha256")
        != runtime.get("environment_receipt_sha256")
        or commit.get("environment_tree_sha256")
        != runtime.get("environment_tree_sha256")
        or commit.get("code_sandbox_config_sha256")
        != runtime.get("code_sandbox_config_sha256")
        or commit.get("code_sandbox_binary_sha256")
        != runtime.get("code_sandbox_binary_sha256")
        or commit.get("code_sandbox_probe_sha256")
        != runtime.get("code_sandbox_probe_sha256")
        or not _sha256(commit.get("sandbox_receipt_sha256"))
        or commit.get("code_sandbox_probe_passed") is not True
        or not _sha256(commit.get("code_sandbox_probe_result_sha256"))
        or commit.get("mbpp_allocation_setup_receipts_sha256")
        != final_setup_receipts_sha256
    ):
        raise PCF1NormalizationError("PCF1 model custody binding differs")

    evaluation_settings = {
        key: revision.get(key)
        for key in (
            "model_loader",
            "generation_mode",
            "max_new_tokens",
            "seed",
            "batch_size",
            "shard_count",
        )
    }
    commit_settings = {
        key: commit.get(key) for key in ("model_loader", "max_sequence_length")
    }
    if (
        not _sha256(runtime.get("runtime_sha256"))
        or runtime.get("model_revision") != model.get("model_revision")
        or runtime.get("evaluation_settings") != evaluation_settings
        or runtime.get("commit_settings") != commit_settings
        or runtime.get("environment_verified") is not True
        or runtime.get("code_sandbox_verified") is not True
        or runtime.get("mbpp_calibration_setup_qualifications_verified") is not True
        or runtime.get("mbpp_calibration_allocation_setup_receipts")
        != model.get("mbpp_calibration_allocation_setup_receipts")
        or not all(
            _sha256(runtime.get(field))
            for field in (
                "environment_receipt_sha256",
                "environment_tree_sha256",
                "code_sandbox_config_sha256",
                "code_sandbox_binary_sha256",
                "code_sandbox_probe_sha256",
            )
        )
    ):
        raise PCF1NormalizationError("PCF1 runtime custody binding differs")

    expected_arm_report_hashes = {arm: receipts[arm]["sha256"] for arm in evaluations}
    expected_arm_candidate_hashes = {
        arm: evaluations[arm].get("candidates_sha256") for arm in evaluations
    }
    if (
        not isinstance(commit_inputs, dict)
        or commit_inputs.get("arm_report_sha256s") != expected_arm_report_hashes
        or commit_inputs.get("arm_candidates_sha256s") != expected_arm_candidate_hashes
    ):
        raise PCF1NormalizationError("PCF1 one-shot score/arm binding differs")
    evaluation_metrics = {
        arm: {
            domain: _deferred_arm_metric(commit, arm, domain)
            for domain in ("overall", *DOMAINS)
        }
        for arm in evaluations
    }
    commit_metrics = {
        domain: _commit_metric(commit, domain) for domain in ("overall", *DOMAINS)
    }
    totals = {
        tuple(
            evaluation_metrics[arm][domain]["total"] for domain in ("overall", *DOMAINS)
        )
        for arm in evaluations
    }
    totals.add(
        tuple(commit_metrics[domain]["total"] for domain in ("overall", *DOMAINS))
    )
    expected_totals = (
        TOTAL_ROWS,
        *[commit_metrics[domain]["total"] for domain in DOMAINS],
    )
    if (
        len(totals) != 1
        or next(iter(totals)) != expected_totals
        or sum(expected_totals[1:]) != TOTAL_ROWS
    ):
        raise PCF1NormalizationError("PCF1 native metric geometry differs")
    for domain in ("overall", *DOMAINS):
        if (
            commit_metrics[domain]["revision_correct"]
            != evaluation_metrics["revision"][domain]["generated_correct"]
            or commit_metrics[domain]["unchanged_correct"]
            != evaluation_metrics["unchanged"][domain]["generated_correct"]
        ):
            raise PCF1NormalizationError(
                f"PCF1 commit/candidate count binding differs: {domain}"
            )

    custody_bindings = {
        "data_custody_sha256": receipts["data_custody"]["sha256"],
        "model_custody_sha256": receipts["model_custody"]["sha256"],
        "runtime_custody_sha256": receipts["runtime_custody"]["sha256"],
    }
    common = {
        "schema": ARM_SCHEMA,
        "status": "complete",
        "run_id": run_id,
        "split": "development",
        "full_row_count": TOTAL_ROWS,
        "identity_order_sha256": data["identity_order_sha256"],
        "data_sha256": data["data_sha256"],
        "model_revision": model["model_revision"],
        "runtime_sha256": runtime["runtime_sha256"],
        "custody": custody_bindings,
        "score_consumption_sha256": consumption_receipt["sha256"],
        "score_consumption_state": "consumed",
        "environment_receipt_sha256": runtime["environment_receipt_sha256"],
        "environment_tree_sha256": runtime["environment_tree_sha256"],
        "code_sandbox_config_sha256": runtime["code_sandbox_config_sha256"],
        "code_sandbox_binary_sha256": runtime["code_sandbox_binary_sha256"],
        "code_sandbox_probe_sha256": runtime["code_sandbox_probe_sha256"],
        "sandbox_receipt_sha256": commit["sandbox_receipt_sha256"],
        "one_open_verified": True,
        "mbpp_final_score_setup_qualifications_verified": True,
        "mbpp_final_score_allocation_setup_receipt_count": len(final_setup_receipts),
        "mbpp_final_score_allocation_setup_receipts_sha256": (
            final_setup_receipts_sha256
        ),
    }
    normalized: dict[str, dict[str, Any]] = {}
    for native_arm, normalized_arm in NATIVE_TO_NORMALIZED.items():
        report = evaluations[native_arm]
        counters = report["counters"]
        normalized[normalized_arm] = {
            **common,
            "arm": normalized_arm,
            "metrics": evaluation_metrics[native_arm],
            "truncation_count": _integer(
                counters.get("max_token_exhausted"),
                f"{native_arm}.max_token_exhausted",
            ),
            "malformed_count": _integer(
                arm_malformed[native_arm], f"{native_arm}.malformed"
            ),
            "empty_completion_count": _integer(
                counters.get("empty_completions"),
                f"{native_arm}.empty_completions",
            ),
            "capability_policy_rejection_count": _integer(
                arm_policy_rejected[native_arm],
                f"{native_arm}.capability_policy_rejected",
            ),
            "native_report": receipts[native_arm],
            "resource_accounting": expected_confirmation_accounting[native_arm],
        }

    overall = commit_metrics["overall"]
    normalized["learned_commit"] = {
        **common,
        "arm": "learned_commit",
        "metrics": {
            domain: {
                "selected_correct": commit_metrics[domain]["selected_correct"],
                "total": commit_metrics[domain]["total"],
            }
            for domain in ("overall", *DOMAINS)
        },
        "truncation_count": _integer(
            commit.get("confirmation_prompt_truncated"),
            "confirmation_prompt_truncated",
        ),
        "malformed_count": _integer(
            commit.get("confirmation_malformed_selections"),
            "confirmation_malformed_selections",
        ),
        "capability_policy_rejection_count": _integer(
            commit.get("selected_capability_policy_rejections"),
            "selected_capability_policy_rejections",
        ),
        "retention": {
            "revision_correct": {
                "retained": overall["revision_correct_retained"],
                "total": overall["revision_correct"],
            },
            "unchanged_correct": {
                "retained": overall["unchanged_correct_retained"],
                "total": overall["unchanged_correct"],
            },
        },
        "order_consistency": {
            "consistent": overall["order_consistent"],
            "total": overall["total"],
        },
        "native_report": receipts["learned_commit"],
        "mbpp_final_score_allocation_setup_receipts": final_setup_receipts,
    }
    output_receipts = _publish_directory(output, normalized)
    return {
        "status": "complete",
        "run_id": run_id,
        "output": str(output.resolve()),
        "reports": output_receipts,
        "compute_custody_created": False,
        "next_action": "bind_normalized_report_hashes_in_compute_custody",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--learned-commit-report", "--commit-report", type=Path, required=True
    )
    parser.add_argument(
        "--revision-report", "--trained-revision-report", type=Path, required=True
    )
    parser.add_argument("--unchanged-report", type=Path, required=True)
    parser.add_argument("--self-refinement-report", type=Path, required=True)
    parser.add_argument(
        "--data-custody", "--data-custody-report", type=Path, required=True
    )
    parser.add_argument(
        "--model-custody", "--model-custody-report", type=Path, required=True
    )
    parser.add_argument(
        "--runtime-custody", "--runtime-custody-report", type=Path, required=True
    )
    parser.add_argument("--score-consumption", type=Path, required=True)
    parser.add_argument("--output", "--output-dir", type=Path, required=True)
    result = normalize(parser.parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
