from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import pytest

import build_q36_mtr_custody as custody_module
from build_q36_mtr_custody import (
    ACCOUNTING_SCHEMA,
    EVIDENCE_PRECOMPUTE_ARTIFACTS,
    EVIDENCE_SCHEMA,
    PRECOMPUTE_SCHEMA,
    Q36MTRCustodyError,
    _validate_role_report,
    _manifest_tree,
    build_authorization,
    build_final,
    evaluation_checkpoint_sha256,
    sha256_file,
    validate_causal_intervention_receipt,
    validate_draft_byte_custody,
)
from build_pcf1_data import revision_prompt
from q36_mtr_contract import graph_payload
from q36_mtr_contract import STAGES
from q36_mtr_roles import MODEL_REVISION, TRAINABLE_PARAMETERS, role_contract
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


def test_role_custody_requires_fresh_optimizer_and_restored_state(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "owner.pt"
    checkpoint.write_bytes(b"owner")
    digest = sha256_file(checkpoint)
    report = {
        **role_contract("owner"),
        "schema": custody_module.ROLE_REPORT_SCHEMA,
        "status": "complete",
        "update": 256,
        "updates": 256,
        "selected_rows": 100_000,
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "trainable_master_dtype": "float32",
        "trainable_compute_dtype": "bfloat16",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": digest,
        "warm_start_checkpoint_sha256": None,
        "source_only_model_visible": True,
        "internal_draft_visible": False,
        "draft_token_bytes_present": False,
        "draft_information_available": False,
        "draft_attention_applied": False,
        "optimizer_restored": False,
        "optimizer_initial_state_empty": True,
        "optimizer_state_entries_before_training": 0,
        "optimizer_state_serialized": False,
        "checkpoint_trainable_only": True,
        "router_expert_checkpoint_tensors": 0,
        "serialization_restore_exact": True,
        "initial_trainable_state_sha256": "a" * 64,
        "final_trainable_state_sha256": "b" * 64,
        "sequence_custody": {},
        "training_consumption": {
            "dataset_presentations": 100_000,
            "optimizer_updates": 256,
            "gradient_accumulation": 16,
            "batch_size": 1,
            "microsteps": 4_096,
            "consumed_presentations": 4_096,
            "unique_consumed_presentations": 4_096,
            "complete_dataset_cycles": 0,
            "partial_cycle_presentations": 4_096,
            "presentation_index_sha256": "c" * 64,
            "consumed_token_geometry_sha256": "d" * 64,
            "consumed_draft_attention_sha256": "e" * 64,
        },
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    _validate_role_report(report, "owner", checkpoint, digest)
    for field, forged in (
        ("optimizer_initial_state_empty", False),
        ("optimizer_state_entries_before_training", 1),
        ("optimizer_state_serialized", True),
        ("checkpoint_trainable_only", False),
        ("router_expert_checkpoint_tensors", 1),
        ("serialization_restore_exact", False),
    ):
        changed = copy.deepcopy(report)
        changed[field] = forged
        with pytest.raises(Q36MTRCustodyError):
            _validate_role_report(changed, "owner", checkpoint, digest)


def _causal_mechanics_fixture() -> dict:
    def route_rows(changes: int, delta: float) -> list[dict]:
        return [
            {
                "layer": index,
                "target_positions": 3,
                "experts": 256,
                "top1_changes": 1 if changes and index == 0 else 0,
                "topk_assignment_changes": changes if index == 0 else 0,
                "router_max_abs_delta": delta if index == 0 else 0.0,
            }
            for index in range(40)
        ]

    aligned_route = {
        "control": "aligned",
        "layers": 40,
        "top_k": 8,
        "target_positions_per_layer": 3,
        "top1_changes": 1,
        "topk_assignment_changes": 3,
        "sensitive_layers": 1,
        "router_max_abs_delta": 0.25,
        "route_path_sha256": "a" * 64,
        "layer_receipts": route_rows(3, 0.25),
    }
    hidden_route = {
        "control": "draft_hidden",
        "layers": 40,
        "top_k": 8,
        "target_positions_per_layer": 3,
        "top1_changes": 0,
        "topk_assignment_changes": 0,
        "sensitive_layers": 0,
        "router_max_abs_delta": 0.0,
        "route_path_sha256": "b" * 64,
        "layer_receipts": route_rows(0, 0.0),
    }
    return {
        "causal_draft_intervention": {
            "token_count_exact": True,
            "position_geometry_exact": True,
            "aligned_response_max_abs_delta": 0.5,
            "draft_hidden_response_max_abs_delta": 0.0,
            "draft_hidden_invariant_tolerance": 2e-3,
            "aligned_sensitivity_floor": 1e-2,
            "draft_hidden_counterfactual_invariant": True,
            "aligned_counterfactual_sensitive": True,
            "native_router": {
                "aligned": aligned_route,
                "draft_hidden": hidden_route,
                "invariant_tolerance": 2e-3,
                "sensitivity_floor": 1e-2,
                "draft_hidden_route_invariant": True,
                "aligned_route_sensitive": True,
                "aligned_expert_selection_changed": True,
            },
        }
    }


def test_causal_router_receipt_binds_sensitivity_and_hidden_invariance() -> None:
    validate_causal_intervention_receipt(_causal_mechanics_fixture())
    mutations = (
        (
            "hidden_route_change",
            lambda value: value["causal_draft_intervention"]["native_router"][
                "draft_hidden"
            ].__setitem__("topk_assignment_changes", 1),
        ),
        (
            "false_aligned_sensitivity",
            lambda value: value["causal_draft_intervention"][
                "native_router"
            ].__setitem__("aligned_route_sensitive", False),
        ),
        (
            "selection_claim_mismatch",
            lambda value: value["causal_draft_intervention"][
                "native_router"
            ].__setitem__("aligned_expert_selection_changed", False),
        ),
        (
            "nonfinite_router_delta",
            lambda value: value["causal_draft_intervention"]["native_router"][
                "aligned"
            ].__setitem__("router_max_abs_delta", float("nan")),
        ),
        (
            "wrong_layer_count",
            lambda value: value["causal_draft_intervention"]["native_router"][
                "aligned"
            ].__setitem__("layers", 64),
        ),
        (
            "wrong_router_top_k",
            lambda value: value["causal_draft_intervention"]["native_router"][
                "aligned"
            ].__setitem__("top_k", 4),
        ),
        (
            "wrong_expert_count",
            lambda value: value["causal_draft_intervention"]["native_router"][
                "aligned"
            ]["layer_receipts"][0].__setitem__("experts", 255),
        ),
        (
            "wrong_layer_order",
            lambda value: value["causal_draft_intervention"]["native_router"][
                "aligned"
            ]["layer_receipts"][0].__setitem__("layer", 1),
        ),
    )
    for _name, mutate in mutations:
        forged = copy.deepcopy(_causal_mechanics_fixture())
        mutate(forged)
        with pytest.raises(Q36MTRCustodyError):
            validate_causal_intervention_receipt(forged)


def test_draft_byte_custody_replays_raw_to_canonical_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    train_identity = "1" * 64
    development_identity = "2" * 64
    train_sources = {
        train_identity: {
            "source_prompt": "train problem",
            "split": "train",
            "task": "math500",
        }
    }
    development_sources = {
        development_identity: {
            "source_prompt": "development problem",
            "split": "development",
            "task": "math500",
        }
    }
    raw = {
        train_identity: " \ntrain draft\t",
        development_identity: "\ndevelopment draft \n",
    }
    drafts = tmp_path / "drafts.jsonl"
    drafts.write_text(
        "".join(
            json.dumps(
                {
                    "schema": custody_module.DRAFT_SCHEMA,
                    "identity_sha256": identity,
                    "split": ("train" if identity == train_identity else "development"),
                    "task": "math500",
                    "prompt_sha256": hashlib.sha256(
                        (
                            "train problem"
                            if identity == train_identity
                            else "development problem"
                        ).encode()
                    ).hexdigest(),
                    "owner_checkpoint_sha256": "c" * 64,
                    "model_revision": MODEL_REVISION,
                    "completion": raw[identity],
                }
            )
            + "\n"
            for identity in (development_identity, train_identity)
        ),
        encoding="utf-8",
    )

    def eval_row(identity: str, source: str) -> dict:
        canonical = raw[identity].strip()
        return {
            "identity_sha256": identity,
            "question": revision_prompt(source, canonical),
            "internal_draft": {"completion": canonical},
            "model_owned_draft_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "raw_model_owned_draft_sha256": hashlib.sha256(
                raw[identity].encode()
            ).hexdigest(),
            "draft_canonicalization": "unicode_outer_whitespace_strip_v1",
        }

    calibration = eval_row(train_identity, "train problem")
    development = eval_row(development_identity, "development problem")
    monkeypatch.setattr(custody_module, "REVISION_PRESENTATIONS", 1)
    monkeypatch.setattr(
        custody_module,
        "load_rows",
        lambda _path, split: [calibration if split == "calibration" else development],
    )
    revision = tmp_path / "revision.jsonl"
    revision.write_text(
        json.dumps(
            {
                **{
                    key: value
                    for key, value in calibration.items()
                    if key != "internal_draft"
                },
                "source_identity_sha256": train_identity,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    receipts = [
        {
            "identity_sha256": identity,
            "raw_sha256": hashlib.sha256(raw[identity].encode()).hexdigest(),
            "canonical_sha256": hashlib.sha256(
                raw[identity].strip().encode()
            ).hexdigest(),
        }
        for identity in sorted(raw)
    ]
    digest = hashlib.sha256(
        b"".join(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            for row in receipts
        )
    ).hexdigest()
    report = {
        "owner_checkpoint_sha256": "c" * 64,
        "draft_byte_custody": {
            "raw_decode_preserved_in_merged_drafts": True,
            "canonicalization": "unicode_outer_whitespace_strip_v1",
            "canonicalized_drafts": 2,
            "identity_raw_canonical_sha256": digest,
        },
    }
    artifacts = {
        "drafts": drafts,
        "calibration_data": tmp_path / "calibration.jsonl",
        "development_data": tmp_path / "development.jsonl",
        "revision_training_data": revision,
    }
    validate_draft_byte_custody(artifacts, report, train_sources, development_sources)
    forged = copy.deepcopy(report)
    forged["draft_byte_custody"]["identity_raw_canonical_sha256"] = "0" * 64
    with pytest.raises(Q36MTRCustodyError):
        validate_draft_byte_custody(
            artifacts, forged, train_sources, development_sources
        )


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
