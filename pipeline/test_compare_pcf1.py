"""Focused tests for the frozen PCF1 final reducer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import pytest

from pipeline.compare_pcf1 import PCF1ComparisonError, compare

RUN_ID = "pcf1-test-run"
IDENTITY_SHA256 = "1" * 64
DATA_SHA256 = "2" * 64
MODEL_REVISION = "81eaece1948f3875421d9a45bc55487d10e2d894"
RUNTIME_SHA256 = "4" * 64
ENVIRONMENT_RECEIPT_SHA256 = "5" * 64
ENVIRONMENT_TREE_SHA256 = "6" * 64
SANDBOX_CONFIG_SHA256 = "7" * 64
SANDBOX_BINARY_SHA256 = "8" * 64
SANDBOX_PROBE_SHA256 = "9" * 64
SCORE_CONSUMPTION_SHA256 = "a" * 64
FINAL_SETUP_RECEIPTS_SHA256 = "b" * 64
CALIBRATION_SETUP_CUSTODY = {
    arm: {
        "sandbox_probe_sha256s": [SANDBOX_PROBE_SHA256] * 4,
        "setup_receipt_count": 4,
        "setup_receipt_shards_sha256": digest,
    }
    for arm, digest in (("revision", "c" * 64), ("unchanged", "d" * 64))
}
DOMAIN_TOTALS = {"math500": 623, "bbh_logic": 637, "mbpp": 29}
DOMAIN_CORRECT = {
    "unchanged": {"math500": 100, "bbh_logic": 280, "mbpp": 7},
    "self_refinement": {"math500": 120, "bbh_logic": 284, "mbpp": 9},
    "trained_revision": {"math500": 130, "bbh_logic": 312, "mbpp": 10},
    "learned_commit": {"math500": 140, "bbh_logic": 315, "mbpp": 10},
}


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(correct: int, total: int, *, selected: bool) -> dict[str, int]:
    field = "selected_correct" if selected else "generated_correct"
    return {field: correct, "total": total}


def _arm(arm: str) -> dict:
    selected = arm == "learned_commit"
    domains = DOMAIN_CORRECT[arm]
    report = {
        "schema": "shohin-pcf1-arm-report-v1",
        "status": "complete",
        "arm": arm,
        "run_id": RUN_ID,
        "split": "development",
        "full_row_count": 1289,
        "identity_order_sha256": IDENTITY_SHA256,
        "data_sha256": DATA_SHA256,
        "model_revision": MODEL_REVISION,
        "runtime_sha256": RUNTIME_SHA256,
        "truncation_count": 0,
        "malformed_count": 0,
        "metrics": {
            "overall": _metric(sum(domains.values()), 1289, selected=selected),
            **{
                domain: _metric(correct, DOMAIN_TOTALS[domain], selected=selected)
                for domain, correct in domains.items()
            },
        },
        "mbpp_final_score_setup_qualifications_verified": True,
        "mbpp_final_score_allocation_setup_receipt_count": 1,
        "mbpp_final_score_allocation_setup_receipts_sha256": (
            FINAL_SETUP_RECEIPTS_SHA256
        ),
    }
    if selected:
        report["retention"] = {
            # These are the smallest integers that clear 95% at the frozen
            # boundary scores: ceil(.95*452) and ceil(.95*387).
            "revision_correct": {"retained": 430, "total": 452},
            "unchanged_correct": {"retained": 368, "total": 387},
        }
        report["order_consistency"] = {"consistent": 1289, "total": 1289}
    return report


Mutation = Callable[[dict[str, dict], dict[str, dict]], None]


def _fixture(
    root: Path, mutation: Mutation | None = None
) -> tuple[argparse.Namespace, dict[str, Path]]:
    custody = {
        "data_custody": {
            "schema": "shohin-pcf1-data-custody-v1",
            "status": "complete",
            "run_id": RUN_ID,
            "custody_verified": True,
            "source_disjoint": True,
            "confirmation_rows": 1289,
            "identity_order_sha256": IDENTITY_SHA256,
            "data_sha256": DATA_SHA256,
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
            "model_revision": MODEL_REVISION,
        },
        "runtime_custody": {
            "schema": "shohin-pcf1-runtime-custody-v1",
            "status": "complete",
            "run_id": RUN_ID,
            "custody_verified": True,
            "runtime_sha256": RUNTIME_SHA256,
            "environment_verified": True,
            "environment_receipt_sha256": ENVIRONMENT_RECEIPT_SHA256,
            "environment_tree_sha256": ENVIRONMENT_TREE_SHA256,
            "code_sandbox_verified": True,
            "code_sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
            "code_sandbox_binary_sha256": SANDBOX_BINARY_SHA256,
            "code_sandbox_probe_sha256": SANDBOX_PROBE_SHA256,
            "mbpp_calibration_setup_qualifications_verified": True,
            "mbpp_calibration_allocation_setup_receipts": CALIBRATION_SETUP_CUSTODY,
        },
    }
    arms = {arm: _arm(arm) for arm in DOMAIN_CORRECT}
    if mutation is not None:
        mutation(arms, custody)
    late_compute_mutation = custody.pop("compute_custody_mutation", None)

    paths: dict[str, Path] = {}
    for role, report in custody.items():
        paths[role] = _write(root / f"{role}.json", report)
    custody_hashes = {
        f"{role}_sha256": _sha256(paths[role])
        for role in ("data_custody", "model_custody", "runtime_custody")
    }
    for arm, report in arms.items():
        report["custody"] = dict(custody_hashes)
        paths[arm] = _write(root / f"{arm}.json", report)

    compute = {
        "schema": "shohin-pcf1-compute-custody-v1",
        "status": "complete",
        "run_id": custody["data_custody"]["run_id"],
        "custody_verified": True,
        **custody_hashes,
        "arm_report_sha256s": {arm: _sha256(paths[arm]) for arm in DOMAIN_CORRECT},
        "environment_verified": True,
        "environment_receipt_sha256": ENVIRONMENT_RECEIPT_SHA256,
        "environment_tree_sha256": ENVIRONMENT_TREE_SHA256,
        "code_sandbox_verified": True,
        "code_sandbox_config_sha256": SANDBOX_CONFIG_SHA256,
        "code_sandbox_binary_sha256": SANDBOX_BINARY_SHA256,
        "code_sandbox_probe_sha256": SANDBOX_PROBE_SHA256,
        "mbpp_calibration_setup_qualifications_verified": True,
        "mbpp_calibration_allocation_setup_receipts": CALIBRATION_SETUP_CUSTODY,
        "mbpp_final_score_setup_qualifications_verified": True,
        "mbpp_final_score_allocation_setup_receipt_count": 1,
        "mbpp_final_score_allocation_setup_receipts_sha256": (
            FINAL_SETUP_RECEIPTS_SHA256
        ),
        "one_open_verified": True,
        "accounting_verified": True,
        "score_consumption_state": "consumed",
        "score_consumption_sha256": SCORE_CONSUMPTION_SHA256,
        "retry_count": 0,
        "successor_authorized": False,
        "successor_submitted": False,
        "charged_gpu_seconds": 123.0,
    }
    # Some failure cases intentionally mutate the compute receipt after all
    # exact input hashes are known.
    if callable(late_compute_mutation):
        late_compute_mutation(compute)
    paths["compute_custody"] = _write(root / "compute_custody.json", compute)
    paths["output"] = root / "final.json"
    return (
        argparse.Namespace(
            learned_commit_report=paths["learned_commit"],
            trained_revision_report=paths["trained_revision"],
            unchanged_report=paths["unchanged"],
            self_refinement_report=paths["self_refinement"],
            data_custody=paths["data_custody"],
            model_custody=paths["model_custody"],
            runtime_custody=paths["runtime_custody"],
            compute_custody=paths["compute_custody"],
            output=paths["output"],
        ),
        paths,
    )


def _set_overall(report: dict, correct: int) -> None:
    metrics = report["metrics"]["overall"]
    field = "selected_correct" if "selected_correct" in metrics else "generated_correct"
    metrics[field] = correct


def test_exact_boundary_pass_is_atomic_hash_complete_and_terminal(
    tmp_path: Path,
) -> None:
    args, paths = _fixture(tmp_path)
    result = compare(args)

    assert result["gate_pass"] is True
    assert result["final_result"] == "PASS"
    assert result["gates"] == {
        "capable_host": True,
        "causal_revision_margin": True,
        "revision_retention": True,
        "useful_learned_commitment": True,
        "conservative_commitment": True,
        "complete_custody": True,
    }
    assert result["margins"]["revision_minus_unchanged"] == 65
    assert result["margins"]["revision_minus_self_refinement"] == 39
    assert result["margins"]["commit_minus_revision"] == 13
    assert result["retention"]["revision_correct"]["rate"] == 430 / 452
    assert result["stop_after_gate"] is True
    assert result["automatic_successor_authorized"] is False
    assert result["automatic_successor_submitted"] is False
    assert result["holdout_access_authorized"] is False
    assert result["product_access_authorized"] is False
    assert result["next_action"] == "stop_and_preserve_evidence"
    assert json.loads(paths["output"].read_text(encoding="utf-8")) == result
    for role, path in paths.items():
        if role != "output":
            assert result["inputs"][role]["sha256"] == _sha256(path)


@pytest.mark.parametrize(
    ("name", "mutation", "failed_check", "failed_gate"),
    [
        (
            "unchanged_floor",
            lambda arms, _custody: (
                _set_overall(arms["unchanged"], 386),
                arms["unchanged"]["metrics"]["math500"].__setitem__(
                    "generated_correct", 99
                ),
            ),
            "unchanged_at_least_387",
            "capable_host",
        ),
        (
            "revision_vs_unchanged",
            lambda arms, _custody: (
                _set_overall(arms["trained_revision"], 451),
                arms["trained_revision"]["metrics"]["math500"].__setitem__(
                    "generated_correct", 129
                ),
            ),
            "revision_at_least_65_over_unchanged",
            "causal_revision_margin",
        ),
        (
            "revision_vs_self",
            lambda arms, _custody: (
                _set_overall(arms["self_refinement"], 414),
                arms["self_refinement"]["metrics"]["math500"].__setitem__(
                    "generated_correct", 121
                ),
            ),
            "revision_at_least_39_over_self_refinement",
            "causal_revision_margin",
        ),
        (
            "revision_domain",
            lambda arms, _custody: (
                arms["trained_revision"]["metrics"]["math500"].__setitem__(
                    "generated_correct", 119
                ),
                arms["trained_revision"]["metrics"]["bbh_logic"].__setitem__(
                    "generated_correct", 323
                ),
            ),
            "revision_domain_deltas_vs_self_refinement_nonnegative",
            "revision_retention",
        ),
        (
            "commit_margin",
            lambda arms, _custody: (
                _set_overall(arms["learned_commit"], 464),
                arms["learned_commit"]["metrics"]["math500"].__setitem__(
                    "selected_correct", 139
                ),
            ),
            "commit_at_least_13_over_revision",
            "useful_learned_commitment",
        ),
        (
            "revision_retention",
            lambda arms, _custody: arms["learned_commit"]["retention"][
                "revision_correct"
            ].__setitem__("retained", 429),
            "commit_revision_correct_retention_at_least_95_percent",
            "conservative_commitment",
        ),
        (
            "unchanged_retention",
            lambda arms, _custody: arms["learned_commit"]["retention"][
                "unchanged_correct"
            ].__setitem__("retained", 367),
            "commit_unchanged_correct_retention_at_least_95_percent",
            "conservative_commitment",
        ),
        (
            "commit_domain",
            lambda arms, _custody: (
                arms["learned_commit"]["metrics"]["math500"].__setitem__(
                    "selected_correct", 129
                ),
                arms["learned_commit"]["metrics"]["bbh_logic"].__setitem__(
                    "selected_correct", 326
                ),
            ),
            "commit_domain_deltas_vs_revision_nonnegative",
            "conservative_commitment",
        ),
        (
            "coverage",
            lambda arms, _custody: arms["self_refinement"].__setitem__(
                "full_row_count", 1288
            ),
            "self_refinement_full_1289_coverage",
            "complete_custody",
        ),
        (
            "truncation",
            lambda arms, _custody: arms["trained_revision"].__setitem__(
                "truncation_count", 1
            ),
            "trained_revision_zero_truncation",
            "complete_custody",
        ),
        (
            "malformed",
            lambda arms, _custody: arms["learned_commit"].__setitem__(
                "malformed_count", 1
            ),
            "learned_commit_zero_malformed",
            "complete_custody",
        ),
        (
            "order",
            lambda arms, _custody: arms["learned_commit"][
                "order_consistency"
            ].__setitem__("consistent", 1288),
            "commit_exact_ab_order_consistency",
            "complete_custody",
        ),
        (
            "holdout_seal",
            lambda _arms, custody: custody["data_custody"].__setitem__(
                "holdout_sealed", False
            ),
            "holdout_sealed",
            "complete_custody",
        ),
        (
            "product_seal",
            lambda _arms, custody: custody["data_custody"].__setitem__(
                "product_sealed", False
            ),
            "product_sealed",
            "complete_custody",
        ),
        (
            "public_access",
            lambda _arms, custody: custody["data_custody"].__setitem__(
                "public_access_count", 1
            ),
            "public_access_count_zero",
            "complete_custody",
        ),
        (
            "alternate_model_revision",
            lambda _arms, custody: custody["model_custody"].__setitem__(
                "model_revision", "3" * 40
            ),
            "model_revision_is_pinned_commit",
            "complete_custody",
        ),
        (
            "compute_custody",
            lambda _arms, custody: custody.__setitem__(
                "compute_custody_mutation",
                lambda compute: compute.__setitem__("custody_verified", False),
            ),
            "compute_custody_verified",
            "complete_custody",
        ),
        (
            "compute_hash",
            lambda _arms, custody: custody.__setitem__(
                "compute_custody_mutation",
                lambda compute: compute["arm_report_sha256s"].__setitem__(
                    "learned_commit", "0" * 64
                ),
            ),
            "compute_arm_report_hashes_match",
            "complete_custody",
        ),
        (
            "environment_verification",
            lambda _arms, custody: custody["runtime_custody"].__setitem__(
                "environment_verified", False
            ),
            "runtime_environment_verified",
            "complete_custody",
        ),
        (
            "environment_hash",
            lambda _arms, custody: custody["runtime_custody"].__setitem__(
                "environment_tree_sha256", "bad"
            ),
            "runtime_environment_hashes_well_formed",
            "complete_custody",
        ),
        (
            "sandbox_verification",
            lambda _arms, custody: custody["runtime_custody"].__setitem__(
                "code_sandbox_verified", False
            ),
            "runtime_code_sandbox_verified",
            "complete_custody",
        ),
        (
            "sandbox_probe_hash",
            lambda _arms, custody: custody["runtime_custody"].__setitem__(
                "code_sandbox_probe_sha256", "bad"
            ),
            "runtime_code_sandbox_hashes_well_formed",
            "complete_custody",
        ),
        (
            "calibration_setup_receipt",
            lambda _arms, custody: custody["runtime_custody"][
                "mbpp_calibration_allocation_setup_receipts"
            ]["revision"].__setitem__("setup_receipt_shards_sha256", "bad"),
            "runtime_calibration_setup_qualifications_verified",
            "complete_custody",
        ),
        (
            "final_setup_receipt",
            lambda arms, _custody: arms["learned_commit"].__setitem__(
                "mbpp_final_score_allocation_setup_receipts_sha256", "0" * 64
            ),
            "final_score_setup_qualifications_verified",
            "complete_custody",
        ),
        (
            "compute_environment_verification",
            lambda _arms, custody: custody.__setitem__(
                "compute_custody_mutation",
                lambda compute: compute.__setitem__("environment_verified", False),
            ),
            "compute_environment_verified",
            "complete_custody",
        ),
        (
            "compute_sandbox_verification",
            lambda _arms, custody: custody.__setitem__(
                "compute_custody_mutation",
                lambda compute: compute.__setitem__("code_sandbox_verified", False),
            ),
            "compute_code_sandbox_verified",
            "complete_custody",
        ),
        (
            "compute_sandbox_hash_binding",
            lambda _arms, custody: custody.__setitem__(
                "compute_custody_mutation",
                lambda compute: compute.__setitem__(
                    "code_sandbox_config_sha256", "b" * 64
                ),
            ),
            "compute_environment_sandbox_hashes_match",
            "complete_custody",
        ),
        (
            "one_open",
            lambda _arms, custody: custody.__setitem__(
                "compute_custody_mutation",
                lambda compute: compute.__setitem__("one_open_verified", False),
            ),
            "one_open_verified",
            "complete_custody",
        ),
        (
            "score_consumption",
            lambda _arms, custody: custody.__setitem__(
                "compute_custody_mutation",
                lambda compute: compute.__setitem__(
                    "score_consumption_state", "available"
                ),
            ),
            "score_consumption_verified",
            "complete_custody",
        ),
        (
            "retry_count",
            lambda _arms, custody: custody.__setitem__(
                "compute_custody_mutation",
                lambda compute: compute.__setitem__("retry_count", 1),
            ),
            "retry_count_zero",
            "complete_custody",
        ),
        (
            "successor_authorized",
            lambda _arms, custody: custody.__setitem__(
                "compute_custody_mutation",
                lambda compute: compute.__setitem__("successor_authorized", True),
            ),
            "successor_not_authorized",
            "complete_custody",
        ),
        (
            "successor_submitted",
            lambda _arms, custody: custody.__setitem__(
                "compute_custody_mutation",
                lambda compute: compute.__setitem__("successor_submitted", True),
            ),
            "successor_not_submitted",
            "complete_custody",
        ),
        (
            "accounting",
            lambda _arms, custody: custody.__setitem__(
                "compute_custody_mutation",
                lambda compute: compute.__setitem__("accounting_verified", False),
            ),
            "exact_accounting_verified",
            "complete_custody",
        ),
        (
            "charged_resource",
            lambda _arms, custody: custody.__setitem__(
                "compute_custody_mutation",
                lambda compute: compute.__setitem__(
                    "charged_gpu_seconds", float("nan")
                ),
            ),
            "charged_gpu_seconds_verified",
            "complete_custody",
        ),
    ],
)
def test_each_frozen_conjunct_fails_closed(
    tmp_path: Path,
    name: str,
    mutation: Mutation,
    failed_check: str,
    failed_gate: str,
) -> None:
    del name
    args, paths = _fixture(tmp_path, mutation)
    result = compare(args)

    assert result["checks"][failed_check] is False
    assert result["gates"][failed_gate] is False
    assert result["gate_pass"] is False
    assert result["final_result"] == "FAIL"
    assert result["stop_after_gate"] is True
    assert result["automatic_successor_authorized"] is False
    assert paths["output"].exists()


def test_wrong_schema_or_arm_is_not_a_scientific_result(tmp_path: Path) -> None:
    def mutation(arms: dict[str, dict], _custody: dict[str, dict]) -> None:
        arms["unchanged"]["arm"] = "trained_revision"

    args, paths = _fixture(tmp_path, mutation)
    with pytest.raises(PCF1ComparisonError, match="arm label differs"):
        compare(args)
    assert not paths["output"].exists()


def test_write_once_output_preserves_first_gate(tmp_path: Path) -> None:
    args, paths = _fixture(tmp_path)
    first = compare(args)
    first_bytes = paths["output"].read_bytes()

    with pytest.raises(PCF1ComparisonError, match="refusing existing"):
        compare(args)
    assert paths["output"].read_bytes() == first_bytes
    assert json.loads(first_bytes) == first
