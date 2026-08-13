from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from compare_q36_mtr import Q36MTRComparisonError, compare
from q36_mtr_contract import ARMS, MODEL_REVISION, TOTAL_ROWS, graph_payload

RUN_ID = "q36_mtr_test_r1"
DATA_SHA256 = "1" * 64
RUNTIME_SHA256 = "2" * 64
IDENTITY_SHA256 = "3" * 64
PRECOMPUTE_SHA256 = "4" * 64
CONSUMPTION_SHA256 = "6" * 64
SOURCE_COMMIT = "7" * 40
MODEL_MANIFEST_SHA256 = "8" * 64
RUNTIME_MANIFEST_SHA256 = "9" * 64
ENVIRONMENT_RECEIPT_SHA256 = "a" * 64
SANDBOX_RECEIPT_SHA256 = "b" * 64
ACCOUNTING_SHA256 = "c" * 64
MIRROR_MANIFEST_SHA256 = "d" * 64
DOMAIN_TOTALS = {"math500": 600, "bbh_logic": 489, "mbpp": 200}
DOMAIN_CORRECT = {
    "unchanged": {"math500": 200, "bbh_logic": 150, "mbpp": 37},
    "self_refinement": {"math500": 215, "bbh_logic": 160, "mbpp": 38},
    "draft_hidden": {"math500": 210, "bbh_logic": 160, "mbpp": 40},
    "trained_revision": {"math500": 240, "bbh_logic": 170, "mbpp": 42},
    "learned_commit": {"math500": 248, "bbh_logic": 175, "mbpp": 42},
}


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arm(name: str) -> dict:
    domain = DOMAIN_CORRECT[name]
    report = {
        "schema": "shohin-q36-mtr-arm-report-v1",
        "status": "complete",
        "run_id": RUN_ID,
        "arm": name,
        "split": "development",
        "model_revision": MODEL_REVISION,
        "data_sha256": DATA_SHA256,
        "runtime_sha256": RUNTIME_SHA256,
        "identity_order_sha256": IDENTITY_SHA256,
        "precompute_custody_sha256": PRECOMPUTE_SHA256,
        "full_row_count": TOTAL_ROWS,
        "candidate_count": TOTAL_ROWS,
        "truncation_count": 0,
        "malformed_count": 0,
        "metrics": {
            "overall": {"correct": sum(domain.values()), "total": TOTAL_ROWS},
            **{
                key: {"correct": value, "total": DOMAIN_TOTALS[key]}
                for key, value in domain.items()
            },
        },
    }
    if name == "learned_commit":
        report["retention"] = {
            "revision_correct": {"retained": 430, "total": 452},
            "unchanged_correct": {"retained": 368, "total": 387},
        }
        report["order_consistency"] = {
            "consistent": TOTAL_ROWS,
            "total": TOTAL_ROWS,
        }
    return report


def _fixture(
    tmp_path: Path, mutation=None
) -> tuple[argparse.Namespace, dict[str, Path]]:
    root = tmp_path / "fixture"
    root.mkdir()
    arms = {arm: _arm(arm) for arm in ARMS}
    custody_overrides: dict = {}
    if mutation is not None:
        mutation(arms, custody_overrides)
    paths = {arm: _write(root / f"{arm}.json", report) for arm, report in arms.items()}
    paths["graph_contract"] = _write(root / "graph.json", graph_payload(SOURCE_COMMIT))
    custody = {
        "schema": "shohin-q36-mtr-final-custody-v1",
        "status": "complete",
        "run_id": RUN_ID,
        "model_revision": MODEL_REVISION,
        "data_sha256": DATA_SHA256,
        "runtime_sha256": RUNTIME_SHA256,
        "identity_order_sha256": IDENTITY_SHA256,
        "precompute_custody_sha256": PRECOMPUTE_SHA256,
        "arm_report_sha256s": {arm: _sha256(paths[arm]) for arm in ARMS},
        "custody_verified": True,
        "source_disjoint": True,
        "model_manifest_verified": True,
        "model_manifest_sha256": MODEL_MANIFEST_SHA256,
        "runtime_manifest_verified": True,
        "runtime_manifest_sha256": RUNTIME_MANIFEST_SHA256,
        "runtime_source_commit": SOURCE_COMMIT,
        "checkpoint_hashes_verified": True,
        "checkpoint_sha256s": {
            "owner": "e" * 64,
            "trained_revision": "f" * 64,
            "draft_hidden": "0" * 64,
            "learned_commit": "1" * 64,
        },
        "environment_verified": True,
        "environment_receipt_sha256": ENVIRONMENT_RECEIPT_SHA256,
        "sandbox_verified": True,
        "sandbox_receipt_sha256": SANDBOX_RECEIPT_SHA256,
        "scheduler_accounting_verified": True,
        "scheduler_accounting_sha256": ACCOUNTING_SHA256,
        "one_assessor_open_verified": True,
        "assessor_semantic_reads": 1,
        "public_access_count": 0,
        "holdout_access_count": 0,
        "product_access_count": 0,
        "retry_count": 0,
        "requeue_count": 0,
        "duplicate_shard_count": 0,
        "orphaned_job_count": 0,
        "successor_authorized": False,
        "successor_submitted": False,
        "score_consumption_state": "consumed",
        "score_consumption_sha256": CONSUMPTION_SHA256,
        "graph_contract_sha256": _sha256(paths["graph_contract"]),
        "source_commit": SOURCE_COMMIT,
        "h100_request_count": 61,
        "completed_h100_allocation_count": 61,
        "charged_gpu_seconds": 212040.0,
        "evidence_mirror_verified": True,
        "evidence_mirror_manifest_sha256": MIRROR_MANIFEST_SHA256,
        **custody_overrides,
    }
    paths["final_custody"] = _write(root / "custody.json", custody)
    paths["output"] = root / "final.json"
    return (
        argparse.Namespace(
            **{f"{arm}_report": paths[arm] for arm in ARMS},
            final_custody=paths["final_custody"],
            graph_contract=paths["graph_contract"],
            output=paths["output"],
        ),
        paths,
    )


def _set_score(report: dict, domain: str, value: int) -> None:
    old = report["metrics"][domain]["correct"]
    report["metrics"][domain]["correct"] = value
    report["metrics"]["overall"]["correct"] += value - old


def test_exact_boundary_pass_is_terminal_and_atomic(tmp_path: Path) -> None:
    args, paths = _fixture(tmp_path)
    result = compare(args)
    assert result["formal_result"] == "PASS"
    assert result["gate_pass"] is True
    assert result["margins"] == {
        "revision_minus_unchanged": 65,
        "revision_minus_self_refinement": 39,
        "revision_minus_draft_hidden": 42,
        "commit_minus_revision": 13,
    }
    assert all(result["gates"].values())
    assert result["stop_after_gate"] is True
    assert result["automatic_retry_authorized"] is False
    assert result["automatic_confirmation_authorized"] is False
    assert result["automatic_successor_authorized"] is False
    assert result["next_action"] == "stop_and_preserve_evidence"
    assert json.loads(paths["output"].read_text()) == result
    with pytest.raises(Q36MTRComparisonError):
        compare(args)


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (
            lambda arms, _: _set_score(arms["unchanged"], "math500", 199),
            "unchanged_at_least_387",
        ),
        (
            lambda arms, _: _set_score(arms["trained_revision"], "math500", 239),
            "revision_at_least_65_over_unchanged",
        ),
        (
            lambda arms, _: _set_score(arms["self_refinement"], "math500", 216),
            "revision_at_least_39_over_self_refinement",
        ),
        (
            lambda arms, _: _set_score(arms["draft_hidden"], "math500", 214),
            "revision_at_least_39_over_draft_hidden",
        ),
        (
            lambda arms, _: _set_score(arms["learned_commit"], "math500", 247),
            "commit_at_least_13_over_revision",
        ),
        (
            lambda arms, _: arms["learned_commit"]["retention"][
                "revision_correct"
            ].__setitem__("retained", 429),
            "commit_revision_retention_at_least_95_percent",
        ),
        (
            lambda arms, _: arms["learned_commit"].__setitem__("truncation_count", 1),
            "learned_commit_zero_truncation",
        ),
        (
            lambda arms, _: arms["draft_hidden"].__setitem__("malformed_count", 1),
            "draft_hidden_zero_malformed",
        ),
        (
            lambda arms, _: arms["learned_commit"]["order_consistency"].__setitem__(
                "consistent", 1288
            ),
            "commit_exact_ab_order_consistency",
        ),
        (
            lambda _arms, custody: custody.__setitem__("retry_count", 1),
            "custody_retry_requeue_duplicate_orphan_zero",
        ),
        (
            lambda _arms, custody: custody.__setitem__(
                "evidence_mirror_verified", False
            ),
            "custody_evidence_mirror_verified",
        ),
        (
            lambda _arms, custody: custody.__setitem__(
                "runtime_source_commit", "8" * 40
            ),
            "custody_runtime_manifest_verified",
        ),
        (
            lambda _arms, custody: custody.__setitem__(
                "completed_h100_allocation_count", 60
            ),
            "custody_exact_h100_request_count",
        ),
    ],
)
def test_each_scientific_or_custody_miss_writes_terminal_fail(
    tmp_path: Path, mutation, failed_check: str
) -> None:
    args, _ = _fixture(tmp_path, mutation)
    result = compare(args)
    assert result["formal_result"] == "FAIL"
    assert result["gate_pass"] is False
    assert result["checks"][failed_check] is False
    assert result["stop_after_gate"] is True
    assert result["automatic_confirmation_authorized"] is False


def test_negative_domain_delta_fails_causal_gate(tmp_path: Path) -> None:
    def mutation(arms, _custody) -> None:
        _set_score(arms["trained_revision"], "mbpp", 36)
        _set_score(arms["trained_revision"], "math500", 246)

    args, _ = _fixture(tmp_path, mutation)
    result = compare(args)
    assert (
        result["checks"]["revision_domain_deltas_vs_all_controls_nonnegative"] is False
    )
    assert result["gates"]["causal_revision"] is False
    assert result["formal_result"] == "FAIL"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda arms, _: arms["unchanged"].__setitem__(
            "identity_order_sha256", "8" * 64
        ),
        lambda arms, _: arms["unchanged"].__setitem__("model_revision", "9" * 40),
        lambda arms, _: arms["unchanged"]["metrics"]["overall"].__setitem__(
            "total", 1288
        ),
        lambda arms, _: arms["unchanged"]["metrics"]["math500"].__setitem__(
            "total", 599
        ),
    ],
)
def test_structural_mismatch_is_infrastructure_error_without_result(
    tmp_path: Path, mutation
) -> None:
    args, paths = _fixture(tmp_path, mutation)
    with pytest.raises(Q36MTRComparisonError):
        compare(args)
    assert not paths["output"].exists()


def test_arm_tamper_after_custody_is_rejected(tmp_path: Path) -> None:
    args, paths = _fixture(tmp_path)
    report = json.loads(paths["unchanged"].read_text())
    report["candidate_count"] = 1288
    _write(paths["unchanged"], report)
    with pytest.raises(Q36MTRComparisonError, match="arm hashes"):
        compare(args)
    assert not paths["output"].exists()


def test_graph_tamper_after_custody_is_rejected(tmp_path: Path) -> None:
    args, paths = _fixture(tmp_path)
    graph = json.loads(paths["graph_contract"].read_text())
    graph["source_commit"] = "8" * 40
    _write(paths["graph_contract"], graph)
    with pytest.raises(Q36MTRComparisonError, match="graph binding"):
        compare(args)
    assert not paths["output"].exists()


def test_precompute_custody_binding_mismatch_is_rejected(tmp_path: Path) -> None:
    def mutation(arms, _custody) -> None:
        arms["draft_hidden"]["precompute_custody_sha256"] = "9" * 64

    args, paths = _fixture(tmp_path, mutation)
    with pytest.raises(Q36MTRComparisonError, match="precompute custody"):
        compare(args)
    assert not paths["output"].exists()
