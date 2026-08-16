from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from build_q36_mtr_custody import (
    ACCOUNTING_SCHEMA,
    EVIDENCE_PRECOMPUTE_ARTIFACTS,
    PRECOMPUTE_SCHEMA,
)
from compare_q36_mtr import ARM_SCHEMA
from compile_q36_mtr_plan import compile_plan
from mirror_q36_mtr_evidence import Q36MTREvidenceError, mirror, sha256_file
from q36_mtr_contract import MODEL_REVISION, graph_payload
from q36_mtr_evidence import verify_evidence_snapshot
from score_q36_mtr import AUTHORIZATION_SCHEMA, CONSUMPTION_SCHEMA, SCORE_SCHEMA

COMMIT = "1" * 40
RUN_ID = "q36-evidence-test"


def _write(path: Path, value: object) -> Path:
    if isinstance(value, dict):
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    else:
        path.write_text(str(value), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> argparse.Namespace:
    graph = _write(tmp_path / "graph.json", graph_payload(COMMIT))
    graph_sha = sha256_file(graph)
    plan = _write(
        tmp_path / "plan.json", compile_plan(graph_payload(COMMIT), graph_sha)
    )
    dispatch = _write(tmp_path / "dispatch.json", {"dispatch": True})
    model_manifest = _write(tmp_path / "model.SHA256SUMS", "model\n")
    runtime_manifest = _write(tmp_path / "runtime.SHA256SUMS", "runtime\n")
    artifacts = {}
    for name in EVIDENCE_PRECOMPUTE_ARTIFACTS:
        artifacts[name] = _write(tmp_path / f"precompute_{name}.bin", f"{name}\n")
    precompute = _write(
        tmp_path / "precompute.json",
        {
            "schema": PRECOMPUTE_SCHEMA,
            "status": "complete",
            "run_id": RUN_ID,
            "source_commit": COMMIT,
            "graph_contract_sha256": graph_sha,
            "model_revision": MODEL_REVISION,
            "model_manifest_sha256": sha256_file(model_manifest),
            "runtime_manifest_sha256": sha256_file(runtime_manifest),
            "artifact_sha256s": {
                name: sha256_file(path) for name, path in artifacts.items()
            },
        },
    )
    prescore = _write(
        tmp_path / "prescore.json",
        {
            "schema": ACCOUNTING_SCHEMA,
            "status": "complete",
            "phase": "prescore",
            "run_id": RUN_ID,
        },
    )
    accounting = _write(
        tmp_path / "accounting.json",
        {
            "schema": ACCOUNTING_SCHEMA,
            "status": "complete",
            "phase": "final",
            "run_id": RUN_ID,
            "graph_contract_sha256": graph_sha,
            "plan_sha256": sha256_file(plan),
            "dispatch_receipt_sha256": sha256_file(dispatch),
        },
    )
    authorization = _write(
        tmp_path / "authorization.json",
        {"schema": AUTHORIZATION_SCHEMA, "status": "complete", "run_id": RUN_ID},
    )
    consumption = _write(
        tmp_path / "consumption.json",
        {
            "schema": CONSUMPTION_SCHEMA,
            "status": "consumed",
            "run_id": RUN_ID,
            "authorization_sha256": sha256_file(authorization),
        },
    )
    outcomes = _write(tmp_path / "outcomes.jsonl", "outcomes\n")
    sandbox = _write(tmp_path / "sandbox.json", "sandbox\n")
    score = _write(
        tmp_path / "score.json",
        {
            "schema": SCORE_SCHEMA,
            "status": "complete",
            "run_id": RUN_ID,
            "score_authorization_sha256": sha256_file(authorization),
            "score_consumption_sha256": sha256_file(consumption),
            "outcomes_sha256": sha256_file(outcomes),
            "sandbox_receipt_sha256": sha256_file(sandbox),
            "input_hashes": {"prescore_accounting_sha256": sha256_file(prescore)},
        },
    )
    arms = {}
    for name in (
        "learned_commit",
        "trained_revision",
        "unchanged",
        "self_refinement",
        "draft_hidden",
    ):
        arms[name] = _write(
            tmp_path / f"{name}.json",
            {
                "schema": ARM_SCHEMA,
                "status": "complete",
                "arm": name,
                "run_id": RUN_ID,
                "score_report_sha256": sha256_file(score),
                "precompute_custody_sha256": sha256_file(precompute),
            },
        )
    authorized_root = tmp_path / "authorized"
    authorized_root.mkdir()
    return argparse.Namespace(
        run_id=RUN_ID,
        source_commit=COMMIT,
        graph_contract=graph,
        precompute_custody=precompute,
        prescore_accounting=prescore,
        score_authorization=authorization,
        score_consumption=consumption,
        score_report=score,
        score_outcomes=outcomes,
        score_sandbox_receipt=sandbox,
        scheduler_accounting=accounting,
        plan=plan,
        dispatch_receipt=dispatch,
        model_manifest=model_manifest,
        runtime_manifest=runtime_manifest,
        arm_report=[f"{name}={path}" for name, path in sorted(arms.items())],
        precompute_artifact=[
            f"{name}={path}" for name, path in sorted(artifacts.items())
        ],
        authorized_root=authorized_root,
        output_root=authorized_root / "snapshot",
    )


def test_q36_evidence_mirror_copies_and_rehashes_exact_snapshot(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    result = mirror(args)
    manifest = json.loads((args.output_root / "manifest.json").read_text())
    assert result == manifest
    assert result["verified"] is True
    assert result["assessor_board_copied_or_opened"] is False
    assert result["artifact_count"] == len(result["artifact_sha256s"])
    assert (
        verify_evidence_snapshot(args.output_root / "manifest.json", result)[
            "artifact_tree_sha256"
        ]
        == result["artifact_tree_sha256"]
    )
    for record in result["records"]:
        assert sha256_file(Path(record["mirror"])) == record["sha256"]


def test_q36_evidence_mirror_rejects_tampered_precompute_artifact(
    tmp_path: Path,
) -> None:
    args = _fixture(tmp_path)
    _, rendered = args.precompute_artifact[0].split("=", 1)
    Path(rendered).write_text("tampered\n", encoding="utf-8")
    with pytest.raises(Q36MTREvidenceError):
        mirror(args)
