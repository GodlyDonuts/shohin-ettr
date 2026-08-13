from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from build_q36_mtr_custody import (
    ACCOUNTING_SCHEMA,
    EVIDENCE_PRECOMPUTE_ARTIFACTS,
    EVIDENCE_SCHEMA,
    PRECOMPUTE_SCHEMA,
    Q36MTRCustodyError,
    _manifest_tree,
    build_authorization,
    build_final,
    evaluation_checkpoint_sha256,
    sha256_file,
)
from q36_mtr_contract import graph_payload
from q36_mtr_contract import STAGES
from score_q36_mtr import (
    AUTHORIZATION_SCHEMA,
    CONSUMPTION_SCHEMA,
    SCORE_SCHEMA,
)


def test_q36_manifest_tree_accepts_only_exact_hash_bound_members(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "member").write_text("value")
    digest = hashlib.sha256(b"value").hexdigest()
    manifest = root / "SHA256SUMS"
    manifest.write_text(f"{digest}  ./member\n")
    receipt = _manifest_tree(root, manifest)
    assert receipt["exact_membership"] is True
    (root / "extra").write_text("extra")
    with pytest.raises(Q36MTRCustodyError):
        _manifest_tree(root, manifest)


def test_matched_arm_checkpoint_lineage_is_role_isolated() -> None:
    hashes = {
        "owner_checkpoint": "1" * 64,
        "aligned_checkpoint": "2" * 64,
        "draft_hidden_checkpoint": "3" * 64,
    }
    assert evaluation_checkpoint_sha256("revision", hashes) == "2" * 64
    assert evaluation_checkpoint_sha256("unchanged", hashes) == "1" * 64
    assert evaluation_checkpoint_sha256("self_refinement", hashes) == "1" * 64
    assert evaluation_checkpoint_sha256("draft_hidden", hashes) == "3" * 64
    with pytest.raises(Q36MTRCustodyError):
        evaluation_checkpoint_sha256("forged", hashes)


def test_q36_authorization_binds_exact_score_inputs_without_board_open(
    tmp_path: Path,
) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(graph_payload("a" * 40)) + "\n")
    names = {
        "application_report",
        "assessor_receipt",
        "commit_training_report",
        "data_report",
        "development_data",
        "draft_hidden_candidates",
        "draft_hidden_evaluation_report",
        "environment_receipt",
        "prescore_accounting",
        "revision_candidates",
        "revision_report",
        "selections",
        "self_refinement_candidates",
        "self_refinement_report",
        "unchanged_candidates",
        "unchanged_report",
    }
    artifacts = {}
    for name in names:
        path = tmp_path / f"{name}.json"
        path.write_text(name + "\n")
        artifacts[name] = path
    prescore_required = [stage.name for stage in STAGES]
    prescore_required = prescore_required[
        : prescore_required.index("precompute_custody") + 1
    ]
    artifacts["prescore_accounting"].write_text(
        json.dumps(
            {
                "schema": ACCOUNTING_SCHEMA,
                "status": "complete",
                "phase": "prescore",
                "run_id": "run",
                "source_commit": "a" * 40,
                "graph_contract_sha256": sha256_file(graph_path),
                "required_stages": prescore_required,
                "h100_request_count": 61,
                "completed_h100_allocation_count": 61,
                "retry_count": 0,
                "requeue_count": 0,
                "duplicate_shard_count": 0,
                "orphaned_job_count": 0,
                "successor_authorized": False,
                "successor_submitted": False,
                "sealed_access": {"holdout": 0, "product": 0, "public": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    precompute = {
        "schema": PRECOMPUTE_SCHEMA,
        "status": "complete",
        "run_id": "run",
        "source_commit": "a" * 40,
        "graph_contract_sha256": sha256_file(graph_path),
        "identity_order_sha256": hashlib.sha256(b"identities").hexdigest(),
        "assessor_board_sha256": hashlib.sha256(b"board").hexdigest(),
        "artifact_sha256s": {
            name: sha256_file(path) for name, path in artifacts.items()
        },
    }
    precompute_path = tmp_path / "precompute.json"
    precompute_path.write_text(json.dumps(precompute) + "\n")
    output = tmp_path / "authorization.json"
    result = build_authorization(
        argparse.Namespace(
            precompute_custody=precompute_path,
            graph_contract=graph_path,
            artifact=[f"{name}={path}" for name, path in sorted(artifacts.items())],
            score_output_root=tmp_path / "score",
            output=output,
        )
    )
    assert result["schema"] == AUTHORIZATION_SCHEMA
    assert result["one_shot"] is True
    assert result["assessor_board_access_count_before"] == 0
    assert "assessor_board" not in result["input_hashes"]


def _write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _final_fixture(tmp_path: Path) -> argparse.Namespace:
    graph_path = _write_json(tmp_path / "graph.json", graph_payload("a" * 40))
    graph_sha256 = sha256_file(graph_path)
    precompute_artifacts = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in EVIDENCE_PRECOMPUTE_ARTIFACTS
    }
    precompute_artifacts.update(
        {
            "owner_checkpoint": "1" * 64,
            "aligned_checkpoint": "2" * 64,
            "draft_hidden_checkpoint": "3" * 64,
            "commit_checkpoint": "4" * 64,
        }
    )
    precompute = {
        "schema": PRECOMPUTE_SCHEMA,
        "status": "complete",
        "run_id": "run",
        "source_commit": "a" * 40,
        "graph_contract_sha256": graph_sha256,
        "model_revision": graph_payload("a" * 40)["model"]["revision"],
        "model_manifest_sha256": "5" * 64,
        "runtime_manifest_sha256": "6" * 64,
        "runtime_sha256": "6" * 64,
        "environment_receipt_sha256": "7" * 64,
        "data_sha256": "8" * 64,
        "identity_order_sha256": "9" * 64,
        "artifact_sha256s": precompute_artifacts,
    }
    precompute_path = _write_json(tmp_path / "precompute.json", precompute)
    precompute_sha256 = sha256_file(precompute_path)
    consumption_path = _write_json(
        tmp_path / "score.score-authorization-consumed.json",
        {
            "schema": CONSUMPTION_SCHEMA,
            "status": "consumed",
            "run_id": "run",
            "authorization_sha256": "a" * 64,
            "score_output_root": str((tmp_path / "score").resolve()),
        },
    )
    score_path = tmp_path / "score" / "report.json"
    score = {
        "schema": SCORE_SCHEMA,
        "status": "complete",
        "run_id": "run",
        "model_revision": precompute["model_revision"],
        "rows": 1_289,
        "outcome_rows": 1_289,
        "identity_order_sha256": precompute["identity_order_sha256"],
        "score_consumption_sha256": sha256_file(consumption_path),
        "score_consumption_state": "consumed",
        "score_authorization_sha256": "a" * 64,
        "assessor_semantic_reads": 1,
        "assessor_rows_read": 1_289,
        "sandbox_receipt_sha256": "b" * 64,
        "sandbox_probe_sha256": "c" * 64,
        "outcomes_sha256": "d" * 64,
        "input_hashes": {"prescore_accounting_sha256": "e" * 64},
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    _write_json(score_path, score)
    score_sha256 = sha256_file(score_path)
    arms = {}
    for arm in (
        "learned_commit",
        "trained_revision",
        "unchanged",
        "self_refinement",
        "draft_hidden",
    ):
        arms[arm] = _write_json(
            tmp_path / f"{arm}.json",
            {
                "schema": "shohin-q36-mtr-arm-report-v1",
                "status": "complete",
                "arm": arm,
                "split": "development",
                "run_id": "run",
                "model_revision": precompute["model_revision"],
                "full_row_count": 1_289,
                "candidate_count": 1_289,
                "identity_order_sha256": precompute["identity_order_sha256"],
                "data_sha256": precompute["data_sha256"],
                "runtime_sha256": precompute["runtime_sha256"],
                "precompute_custody_sha256": precompute_sha256,
                "score_report_sha256": score_sha256,
                "sealed_access": {"holdout": 0, "product": 0, "public": 0},
            },
        )
    required = [stage.name for stage in STAGES]
    required = required[: required.index("normalize") + 1]
    accounting = {
        "schema": ACCOUNTING_SCHEMA,
        "status": "complete",
        "phase": "final",
        "run_id": "run",
        "source_commit": "a" * 40,
        "graph_contract_sha256": graph_sha256,
        "plan_sha256": "f" * 64,
        "dispatch_receipt_sha256": "0" * 64,
        "required_stages": required,
        "h100_request_count": 61,
        "completed_h100_allocation_count": 61,
        "charged_gpu_seconds": 3600,
        "retry_count": 0,
        "requeue_count": 0,
        "duplicate_shard_count": 0,
        "orphaned_job_count": 0,
        "successor_authorized": False,
        "successor_submitted": False,
    }
    accounting_path = _write_json(tmp_path / "accounting.json", accounting)
    evidence_hashes = {
        "graph_contract": graph_sha256,
        "precompute_custody": precompute_sha256,
        "score_report": score_sha256,
        "score_consumption": sha256_file(consumption_path),
        "scheduler_accounting": sha256_file(accounting_path),
        "prescore_accounting": "e" * 64,
        "score_authorization": "a" * 64,
        "score_outcomes": "d" * 64,
        "score_sandbox_receipt": "b" * 64,
        "plan": "f" * 64,
        "dispatch_receipt": "0" * 64,
        "model_manifest": "5" * 64,
        "runtime_manifest": "6" * 64,
        **{f"arm_{arm}": sha256_file(path) for arm, path in arms.items()},
        **{
            f"precompute_{name}": precompute_artifacts[name]
            for name in EVIDENCE_PRECOMPUTE_ARTIFACTS
        },
    }
    evidence_path = _write_json(
        tmp_path / "evidence.json",
        {
            "schema": EVIDENCE_SCHEMA,
            "status": "complete",
            "verified": True,
            "run_id": "run",
            "source_commit": "a" * 40,
            "graph_contract_sha256": graph_sha256,
            "artifact_sha256s": evidence_hashes,
        },
    )
    return argparse.Namespace(
        precompute_custody=precompute_path,
        score_report=score_path,
        score_consumption=consumption_path,
        scheduler_accounting=accounting_path,
        evidence_mirror=evidence_path,
        graph_contract=graph_path,
        arm_report=[f"{arm}={path}" for arm, path in arms.items()],
        output=tmp_path / "final.json",
    )


def test_q36_final_custody_replays_accounting_score_and_mirror(
    tmp_path: Path,
) -> None:
    args = _final_fixture(tmp_path)
    report = build_final(args)
    assert report["custody_verified"] is True
    assert report["checkpoint_hashes_verified"] is True
    assert report["evidence_mirror_verified"] is True


def test_q36_final_custody_rejects_mirror_hash_drift(tmp_path: Path) -> None:
    args = _final_fixture(tmp_path)
    evidence = json.loads(args.evidence_mirror.read_text(encoding="utf-8"))
    evidence["artifact_sha256s"]["score_report"] = "0" * 64
    args.evidence_mirror.write_text(json.dumps(evidence) + "\n", encoding="utf-8")
    with pytest.raises(Q36MTRCustodyError):
        build_final(args)
