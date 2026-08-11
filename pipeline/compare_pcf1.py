#!/usr/bin/env python3
"""Reduce the frozen PCF1 reports to the one final publication gate.

This program is intentionally only a JSON reducer.  It does not inspect model
checkpoints, benchmark rows, or any path named inside an input report.  The
SHA-256 recorded for each input is computed over the exact bytes that were
parsed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

OUTPUT_SCHEMA = "shohin-pcf1-final-comparison-v1"
ARM_SCHEMA = "shohin-pcf1-arm-report-v1"
CUSTODY_SCHEMAS = {
    "data_custody": "shohin-pcf1-data-custody-v1",
    "model_custody": "shohin-pcf1-model-custody-v1",
    "runtime_custody": "shohin-pcf1-runtime-custody-v1",
    "compute_custody": "shohin-pcf1-compute-custody-v1",
}
TOTAL_ROWS = 1289
DOMAINS = ("math500", "bbh_logic", "mbpp")
PINNED_MODEL_REVISION = "81eaece1948f3875421d9a45bc55487d10e2d894"
ARMS = (
    "learned_commit",
    "trained_revision",
    "unchanged",
    "self_refinement",
)


class PCF1ComparisonError(RuntimeError):
    """An input cannot support the frozen PCF1 comparison contract."""


def _path_arg(args: argparse.Namespace, *names: str) -> Path:
    for name in names:
        value = getattr(args, name, None)
        if value is not None:
            return Path(value)
    raise PCF1ComparisonError(f"missing PCF1 input argument: {names[0]}")


def _load_complete(
    path: Path, *, schema: str, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PCF1ComparisonError(f"unreadable PCF1 {label} report: {path}") from error
    if (
        not isinstance(value, dict)
        or value.get("schema") != schema
        or value.get("status") != "complete"
    ):
        raise PCF1ComparisonError(f"incomplete PCF1 {label} report: {path}")
    receipt = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }
    return value, receipt


def _nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _finite_nonnegative_number(value: Any) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        return None
    return float(value)


def _hex_digest(value: Any, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _metric(report: dict[str, Any], domain: str) -> dict[str, int]:
    value = report.get("metrics", {}).get(domain)
    if not isinstance(value, dict):
        raise PCF1ComparisonError(f"missing PCF1 metric: {domain}")
    total = _nonnegative_integer(value.get("total"))
    correct_values = [
        _nonnegative_integer(value.get(name))
        for name in ("correct", "generated_correct", "selected_correct")
        if name in value
    ]
    if (
        total is None
        or not correct_values
        or any(item is None for item in correct_values)
        or len(set(correct_values)) != 1
        or correct_values[0] > total
    ):
        raise PCF1ComparisonError(f"malformed PCF1 metric: {domain}")
    return {"correct": int(correct_values[0]), "total": total}


def _arm_metrics(report: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {domain: _metric(report, domain) for domain in ("overall", *DOMAINS)}


def _retention(commit: dict[str, Any], label: str) -> tuple[int | None, int | None]:
    value = commit.get("retention", {}).get(label)
    if not isinstance(value, dict):
        return None, None
    return (
        _nonnegative_integer(value.get("retained")),
        _nonnegative_integer(value.get("total")),
    )


def _at_least_95_percent(retained: int | None, total: int | None) -> bool:
    return (
        retained is not None
        and total is not None
        and total > 0
        and retained <= total
        and retained * 100 >= total * 95
    )


def _rate(retained: int | None, total: int | None) -> float | None:
    if retained is None or total is None or total <= 0 or retained > total:
        return None
    return retained / total


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish one immutable JSON object without overwriting an earlier gate."""

    if path.exists() or path.is_symlink():
        raise PCF1ComparisonError(f"refusing existing PCF1 comparison: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # link(2), unlike replace(2), cannot overwrite a gate won by a race.
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise PCF1ComparisonError(
            f"refusing existing PCF1 comparison: {path}"
        ) from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def compare(args: argparse.Namespace) -> dict[str, Any]:
    arm_paths = {
        "learned_commit": _path_arg(args, "learned_commit_report", "commit_report"),
        "trained_revision": _path_arg(
            args, "trained_revision_report", "revision_report"
        ),
        "unchanged": _path_arg(args, "unchanged_report"),
        "self_refinement": _path_arg(args, "self_refinement_report"),
    }
    custody_paths = {
        "data_custody": _path_arg(args, "data_custody", "data_custody_report"),
        "model_custody": _path_arg(args, "model_custody", "model_custody_report"),
        "runtime_custody": _path_arg(args, "runtime_custody", "runtime_custody_report"),
        "compute_custody": _path_arg(args, "compute_custody", "compute_custody_report"),
    }
    output = _path_arg(args, "output")

    arms: dict[str, dict[str, Any]] = {}
    input_receipts: dict[str, dict[str, Any]] = {}
    for arm, path in arm_paths.items():
        report, receipt = _load_complete(path, schema=ARM_SCHEMA, label=arm)
        if report.get("arm") != arm:
            raise PCF1ComparisonError(f"PCF1 arm label differs: {arm}")
        arms[arm] = report
        input_receipts[arm] = receipt

    custody: dict[str, dict[str, Any]] = {}
    for role, path in custody_paths.items():
        report, receipt = _load_complete(path, schema=CUSTODY_SCHEMAS[role], label=role)
        custody[role] = report
        input_receipts[role] = receipt

    arm_metrics = {arm: _arm_metrics(report) for arm, report in arms.items()}
    scores = {arm: arm_metrics[arm]["overall"]["correct"] for arm in ARMS}
    domains = {
        arm: {domain: arm_metrics[arm][domain]["correct"] for domain in DOMAINS}
        for arm in ARMS
    }
    domain_totals = {
        arm: {domain: arm_metrics[arm][domain]["total"] for domain in DOMAINS}
        for arm in ARMS
    }

    unchanged = scores["unchanged"]
    self_refinement = scores["self_refinement"]
    revision = scores["trained_revision"]
    commit = scores["learned_commit"]
    revision_vs_unchanged = {
        domain: domains["trained_revision"][domain] - domains["unchanged"][domain]
        for domain in DOMAINS
    }
    revision_vs_self = {
        domain: domains["trained_revision"][domain] - domains["self_refinement"][domain]
        for domain in DOMAINS
    }
    commit_vs_revision = {
        domain: domains["learned_commit"][domain] - domains["trained_revision"][domain]
        for domain in DOMAINS
    }

    revision_retained, revision_retention_total = _retention(
        arms["learned_commit"], "revision_correct"
    )
    unchanged_retained, unchanged_retention_total = _retention(
        arms["learned_commit"], "unchanged_correct"
    )
    order = arms["learned_commit"].get("order_consistency", {})
    order_consistent = (
        _nonnegative_integer(order.get("consistent"))
        if isinstance(order, dict)
        else None
    )
    order_total = (
        _nonnegative_integer(order.get("total")) if isinstance(order, dict) else None
    )

    data = custody["data_custody"]
    model = custody["model_custody"]
    runtime = custody["runtime_custody"]
    compute = custody["compute_custody"]
    precompute_receipt_hashes = {
        role: input_receipts[role]["sha256"]
        for role in ("data_custody", "model_custody", "runtime_custody")
    }
    expected_arm_hashes = {arm: input_receipts[arm]["sha256"] for arm in ARMS}
    expected_arm_custody_bindings = {
        "data_custody_sha256": precompute_receipt_hashes["data_custody"],
        "model_custody_sha256": precompute_receipt_hashes["model_custody"],
        "runtime_custody_sha256": precompute_receipt_hashes["runtime_custody"],
    }

    run_ids = [report.get("run_id") for report in (*arms.values(), *custody.values())]
    identity_hashes = [report.get("identity_order_sha256") for report in arms.values()]
    evaluation_custody_bindings = all(
        isinstance(report.get("custody"), dict)
        and all(
            report["custody"].get(key) == expected
            for key, expected in expected_arm_custody_bindings.items()
        )
        for report in arms.values()
    )
    compute_custody_bindings = all(
        compute.get(key) == expected
        for key, expected in expected_arm_custody_bindings.items()
    )
    compute_arm_hashes = compute.get("arm_report_sha256s")
    runtime_environment_sandbox_hashes = {
        name: runtime.get(name)
        for name in (
            "environment_receipt_sha256",
            "environment_tree_sha256",
            "code_sandbox_config_sha256",
            "code_sandbox_binary_sha256",
            "code_sandbox_probe_sha256",
        )
    }
    compute_environment_sandbox_hashes = {
        name: compute.get(name) for name in runtime_environment_sandbox_hashes
    }

    checks: dict[str, bool] = {
        "unchanged_at_least_387": unchanged >= 387,
        "unchanged_all_domains_nonzero": all(
            domains["unchanged"][domain] > 0 for domain in DOMAINS
        ),
        "revision_at_least_65_over_unchanged": revision >= unchanged + 65,
        "revision_at_least_39_over_self_refinement": revision >= self_refinement + 39,
        "revision_domain_deltas_vs_unchanged_nonnegative": all(
            delta >= 0 for delta in revision_vs_unchanged.values()
        ),
        "revision_domain_deltas_vs_self_refinement_nonnegative": all(
            delta >= 0 for delta in revision_vs_self.values()
        ),
        "commit_at_least_13_over_revision": commit >= revision + 13,
        "commit_revision_retention_denominator_matches": revision_retention_total
        == revision,
        "commit_revision_correct_retention_at_least_95_percent": _at_least_95_percent(
            revision_retained, revision_retention_total
        )
        and revision_retention_total == revision,
        "commit_unchanged_retention_denominator_matches": unchanged_retention_total
        == unchanged,
        "commit_unchanged_correct_retention_at_least_95_percent": _at_least_95_percent(
            unchanged_retained, unchanged_retention_total
        )
        and unchanged_retention_total == unchanged,
        "commit_domain_deltas_vs_revision_nonnegative": all(
            delta >= 0 for delta in commit_vs_revision.values()
        ),
    }

    for arm in ARMS:
        report = arms[arm]
        checks[f"{arm}_full_1289_coverage"] = (
            report.get("split") == "development"
            and _nonnegative_integer(report.get("full_row_count")) == TOTAL_ROWS
            and arm_metrics[arm]["overall"]["total"] == TOTAL_ROWS
            and sum(domain_totals[arm].values()) == TOTAL_ROWS
        )
        checks[f"{arm}_zero_truncation"] = (
            _nonnegative_integer(report.get("truncation_count")) == 0
        )
        checks[f"{arm}_zero_malformed"] = (
            _nonnegative_integer(report.get("malformed_count")) == 0
        )
        checks[f"{arm}_metric_arithmetic_consistent"] = (
            sum(domains[arm].values()) == scores[arm]
        )

    checks.update(
        {
            "all_arms_same_identity_order": (
                all(_hex_digest(value) for value in identity_hashes)
                and len(set(identity_hashes)) == 1
                and data.get("identity_order_sha256") == identity_hashes[0]
            ),
            "all_arms_same_domain_totals": len(
                {
                    tuple(domain_totals[arm][domain] for domain in DOMAINS)
                    for arm in ARMS
                }
            )
            == 1,
            "commit_exact_ab_order_consistency": order_consistent == TOTAL_ROWS
            and order_total == TOTAL_ROWS,
            "all_reports_same_run_id": (
                all(isinstance(value, str) and bool(value) for value in run_ids)
                and len(set(run_ids)) == 1
            ),
            "data_sha256_well_formed": _hex_digest(data.get("data_sha256")),
            "model_revision_is_pinned_commit": model.get("model_revision")
            == PINNED_MODEL_REVISION,
            "runtime_sha256_well_formed": _hex_digest(runtime.get("runtime_sha256")),
            "data_bindings_match": (
                data.get("confirmation_rows") == TOTAL_ROWS
                and data.get("data_sha256")
                == arms["trained_revision"].get("data_sha256")
                and all(
                    report.get("data_sha256") == data.get("data_sha256")
                    for report in arms.values()
                )
            ),
            "model_bindings_match": bool(model.get("model_revision"))
            and all(
                report.get("model_revision") == model.get("model_revision")
                for report in arms.values()
            ),
            "runtime_bindings_match": bool(runtime.get("runtime_sha256"))
            and all(
                report.get("runtime_sha256") == runtime.get("runtime_sha256")
                for report in arms.values()
            ),
            "evaluation_custody_hashes_match": evaluation_custody_bindings,
            "compute_custody_hashes_match": compute_custody_bindings,
            "compute_arm_report_hashes_match": isinstance(compute_arm_hashes, dict)
            and compute_arm_hashes == expected_arm_hashes,
            "data_custody_verified": data.get("custody_verified") is True,
            "model_custody_verified": model.get("custody_verified") is True,
            "runtime_custody_verified": runtime.get("custody_verified") is True,
            "compute_custody_verified": compute.get("custody_verified") is True,
            "runtime_environment_verified": runtime.get("environment_verified") is True,
            "runtime_environment_hashes_well_formed": all(
                _hex_digest(runtime_environment_sandbox_hashes[name])
                for name in ("environment_receipt_sha256", "environment_tree_sha256")
            ),
            "runtime_code_sandbox_verified": runtime.get("code_sandbox_verified")
            is True,
            "runtime_code_sandbox_hashes_well_formed": all(
                _hex_digest(runtime_environment_sandbox_hashes[name])
                for name in (
                    "code_sandbox_config_sha256",
                    "code_sandbox_binary_sha256",
                    "code_sandbox_probe_sha256",
                )
            ),
            "compute_environment_verified": compute.get("environment_verified") is True,
            "compute_code_sandbox_verified": compute.get("code_sandbox_verified")
            is True,
            "runtime_calibration_setup_qualifications_verified": (
                runtime.get("mbpp_calibration_setup_qualifications_verified") is True
                and isinstance(
                    runtime.get("mbpp_calibration_allocation_setup_receipts"), dict
                )
                and set(runtime["mbpp_calibration_allocation_setup_receipts"])
                == {"revision", "unchanged"}
                and all(
                    isinstance(receipt, dict)
                    and isinstance(receipt.get("setup_receipt_count"), int)
                    and not isinstance(receipt.get("setup_receipt_count"), bool)
                    and receipt["setup_receipt_count"] >= 0
                    and _hex_digest(receipt.get("setup_receipt_shards_sha256"))
                    and receipt.get("sandbox_probe_sha256s")
                    == [runtime.get("code_sandbox_probe_sha256")] * 4
                    for receipt in runtime[
                        "mbpp_calibration_allocation_setup_receipts"
                    ].values()
                )
            ),
            "compute_calibration_setup_qualifications_match": (
                compute.get("mbpp_calibration_setup_qualifications_verified") is True
                and compute.get("mbpp_calibration_allocation_setup_receipts")
                == runtime.get("mbpp_calibration_allocation_setup_receipts")
            ),
            "final_score_setup_qualifications_verified": (
                compute.get("mbpp_final_score_setup_qualifications_verified") is True
                and _hex_digest(
                    compute.get("mbpp_final_score_allocation_setup_receipts_sha256")
                )
                and _nonnegative_integer(
                    compute.get("mbpp_final_score_allocation_setup_receipt_count")
                )
                is not None
                and compute.get("mbpp_final_score_allocation_setup_receipt_count", 0)
                > 0
                and all(
                    report.get("mbpp_final_score_setup_qualifications_verified") is True
                    and report.get("mbpp_final_score_allocation_setup_receipts_sha256")
                    == compute.get("mbpp_final_score_allocation_setup_receipts_sha256")
                    and report.get("mbpp_final_score_allocation_setup_receipt_count")
                    == compute.get("mbpp_final_score_allocation_setup_receipt_count")
                    for report in arms.values()
                )
            ),
            "compute_environment_sandbox_hashes_match": (
                compute_environment_sandbox_hashes == runtime_environment_sandbox_hashes
                and all(
                    _hex_digest(value)
                    for value in compute_environment_sandbox_hashes.values()
                )
            ),
            "one_open_verified": compute.get("one_open_verified") is True,
            "score_consumption_verified": (
                compute.get("score_consumption_state") == "consumed"
                and _hex_digest(compute.get("score_consumption_sha256"))
            ),
            "retry_count_zero": _nonnegative_integer(compute.get("retry_count")) == 0,
            "successor_not_authorized": compute.get("successor_authorized") is False,
            "successor_not_submitted": compute.get("successor_submitted") is False,
            "exact_accounting_verified": compute.get("accounting_verified") is True,
            "charged_gpu_seconds_verified": _finite_nonnegative_number(
                compute.get("charged_gpu_seconds")
            )
            is not None,
            "source_disjoint_confirmed": data.get("source_disjoint") is True,
            "holdout_sealed": data.get("holdout_sealed") is True,
            "product_sealed": data.get("product_sealed") is True,
            "public_sealed": data.get("public_sealed") is True,
            "holdout_access_count_zero": _nonnegative_integer(
                data.get("holdout_access_count")
            )
            == 0,
            "product_access_count_zero": _nonnegative_integer(
                data.get("product_access_count")
            )
            == 0,
            "public_access_count_zero": _nonnegative_integer(
                data.get("public_access_count")
            )
            == 0,
        }
    )

    coverage_checks = [
        checks[f"{arm}_full_1289_coverage"]
        and checks[f"{arm}_zero_truncation"]
        and checks[f"{arm}_zero_malformed"]
        and checks[f"{arm}_metric_arithmetic_consistent"]
        for arm in ARMS
    ]
    custody_check_names = (
        "all_arms_same_identity_order",
        "all_arms_same_domain_totals",
        "commit_exact_ab_order_consistency",
        "all_reports_same_run_id",
        "data_sha256_well_formed",
        "model_revision_is_pinned_commit",
        "runtime_sha256_well_formed",
        "data_bindings_match",
        "model_bindings_match",
        "runtime_bindings_match",
        "evaluation_custody_hashes_match",
        "compute_custody_hashes_match",
        "compute_arm_report_hashes_match",
        "data_custody_verified",
        "model_custody_verified",
        "runtime_custody_verified",
        "compute_custody_verified",
        "runtime_environment_verified",
        "runtime_environment_hashes_well_formed",
        "runtime_code_sandbox_verified",
        "runtime_code_sandbox_hashes_well_formed",
        "compute_environment_verified",
        "compute_code_sandbox_verified",
        "runtime_calibration_setup_qualifications_verified",
        "compute_calibration_setup_qualifications_match",
        "final_score_setup_qualifications_verified",
        "compute_environment_sandbox_hashes_match",
        "one_open_verified",
        "score_consumption_verified",
        "retry_count_zero",
        "successor_not_authorized",
        "successor_not_submitted",
        "exact_accounting_verified",
        "charged_gpu_seconds_verified",
        "source_disjoint_confirmed",
        "holdout_sealed",
        "product_sealed",
        "public_sealed",
        "holdout_access_count_zero",
        "product_access_count_zero",
        "public_access_count_zero",
    )
    gates = {
        "capable_host": checks["unchanged_at_least_387"]
        and checks["unchanged_all_domains_nonzero"],
        "causal_revision_margin": checks["revision_at_least_65_over_unchanged"]
        and checks["revision_at_least_39_over_self_refinement"],
        "revision_retention": checks["revision_domain_deltas_vs_unchanged_nonnegative"]
        and checks["revision_domain_deltas_vs_self_refinement_nonnegative"],
        "useful_learned_commitment": checks["commit_at_least_13_over_revision"],
        "conservative_commitment": checks[
            "commit_revision_correct_retention_at_least_95_percent"
        ]
        and checks["commit_unchanged_correct_retention_at_least_95_percent"]
        and checks["commit_domain_deltas_vs_revision_nonnegative"],
        "complete_custody": all(coverage_checks)
        and all(checks[name] for name in custody_check_names),
    }
    passed = all(gates.values())
    result = {
        "schema": OUTPUT_SCHEMA,
        "status": "complete",
        "run_id": arms["learned_commit"].get("run_id"),
        "split": "development",
        "rows": TOTAL_ROWS,
        "inputs": input_receipts,
        "scores": {
            arm: {"overall": scores[arm], "domains": domains[arm]} for arm in ARMS
        },
        "margins": {
            "revision_minus_unchanged": revision - unchanged,
            "revision_minus_self_refinement": revision - self_refinement,
            "commit_minus_revision": commit - revision,
            "revision_domain_deltas_vs_unchanged": revision_vs_unchanged,
            "revision_domain_deltas_vs_self_refinement": revision_vs_self,
            "commit_domain_deltas_vs_revision": commit_vs_revision,
        },
        "retention": {
            "revision_correct": {
                "retained": revision_retained,
                "total": revision_retention_total,
                "rate": _rate(revision_retained, revision_retention_total),
            },
            "unchanged_correct": {
                "retained": unchanged_retained,
                "total": unchanged_retention_total,
                "rate": _rate(unchanged_retained, unchanged_retention_total),
            },
        },
        "checks": checks,
        "gates": gates,
        "gate_pass": passed,
        "final_result": "PASS" if passed else "FAIL",
        "stop_after_gate": True,
        "automatic_successor_authorized": False,
        "automatic_successor_submitted": False,
        "holdout_access_authorized": False,
        "product_access_authorized": False,
        "manual_authorization_required_for_any_later_access": True,
        "next_action": "stop_and_preserve_evidence",
    }
    _atomic_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--learned-commit-report", "--commit-report", type=Path, required=True
    )
    parser.add_argument(
        "--trained-revision-report", "--revision-report", type=Path, required=True
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
    parser.add_argument(
        "--compute-custody", "--compute-custody-report", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    result = compare(parser.parse_args())
    print(
        json.dumps(
            {
                "gate_pass": result["gate_pass"],
                "final_result": result["final_result"],
                "gates": result["gates"],
                "next_action": result["next_action"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["gate_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
