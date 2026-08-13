from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pytest

import validate_q36_mtr_live_preflight as module
from authorize_q36_mtr_phase import SCHEMA as AUTHORIZATION_SCHEMA
from capture_q36_mtr_cluster_preflight import SCHEMA as CLUSTER_SCHEMA
from compile_q36_mtr_plan import compile_plan
from q36_mtr_contract import MODEL_REVISION, graph_payload
from validate_q36_mtr_live_preflight import Q36MTRLivePreflightError, sha256_file

COMMIT = "1" * 40


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch) -> argparse.Namespace:
    graph = tmp_path / "graph.json"
    _write(graph, graph_payload(COMMIT))
    plan = tmp_path / "plan.json"
    _write(plan, compile_plan(graph_payload(COMMIT), sha256_file(graph)))
    environment = tmp_path / "environment.json"
    _write(
        environment,
        {
            "schema": "shohin-q36-mtr-environment-v1",
            "status": "pass",
            "scientific_rows_read": 0,
        },
    )
    sandbox = tmp_path / "sandbox.json"
    _write(sandbox, {"qualified": True})
    monkeypatch.setattr(module, "validate_sandbox_receipt_payload", lambda value: value)
    cluster = tmp_path / "cluster.json"
    _write(
        cluster,
        {
            "schema": CLUSTER_SCHEMA,
            "status": "pass",
            "user": "sa305415",
            "filesystem": "/lustre/fs1",
            "accounting_start": "2026-08-01",
            "queue_empty": True,
            "scientific_jobs_submitted": 0,
        },
    )
    authorization = tmp_path / "authorization.json"
    _write(
        authorization,
        {
            "schema": AUTHORIZATION_SCHEMA,
            "status": "authorized",
            "run_id": "q36-run",
            "source_commit": COMMIT,
            "graph_contract_sha256": sha256_file(graph),
            "plan_sha256": sha256_file(plan),
            "environment_receipt_sha256": sha256_file(environment),
            "sandbox_receipt_sha256": sha256_file(sandbox),
            "cluster_preflight_sha256": sha256_file(cluster),
            "model_revision": MODEL_REVISION,
            "scientific_submit_authorized": True,
            "gate": "one_source_disjoint_development_gate",
            "automatic_retry": False,
            "automatic_successor": False,
            "automatic_confirmation": False,
            "stop_after_gate": True,
        },
    )
    return argparse.Namespace(
        run_id="q36-run",
        phase_authorization=authorization,
        graph_contract=graph,
        plan=plan,
        environment_receipt=environment,
        sandbox_receipt=sandbox,
        cluster_preflight=cluster,
        output=tmp_path / "live.json",
    )


def _runner(overrides: dict[str, str] | None = None):
    values = {
        "lfs": (
            "/lustre/fs1 800000000 1059061760 1059061760 - "
            "700000 1010000 1010000 -\n"
        ),
        "sinfo": "evc20|idle|gpu:nvidia_h100_pcie:2\n",
        "sacct": "700001|360000|gres/gpu:nvidia_h100_pcie=1|0|\n",
    }
    values.update(overrides or {})

    def run(command, **_kwargs):
        return subprocess.CompletedProcess(
            command, 0, stdout=values[command[0]], stderr=""
        )

    return run


def test_live_preflight_revalidates_cluster_without_science(
    tmp_path: Path, monkeypatch
) -> None:
    report = module.validate(_fixture(tmp_path, monkeypatch), _runner())
    assert report["status"] == "pass"
    assert report["scientific_rows_read"] == 0
    assert report["capability_scored"] is False
    assert report["live_eligible_h100_node_count"] == 1
    assert report["live_h100_hours_remaining_before_plan"] == 1_900.0


def test_live_preflight_rejects_mutated_authorization(
    tmp_path: Path, monkeypatch
) -> None:
    args = _fixture(tmp_path, monkeypatch)
    authorization = json.loads(args.phase_authorization.read_text(encoding="utf-8"))
    authorization["automatic_successor"] = True
    args.phase_authorization.write_text(json.dumps(authorization) + "\n")
    with pytest.raises(Q36MTRLivePreflightError):
        module.validate(args, _runner())


def test_live_preflight_rejects_lost_storage_headroom(
    tmp_path: Path, monkeypatch
) -> None:
    with pytest.raises(Q36MTRLivePreflightError):
        module.validate(
            _fixture(tmp_path, monkeypatch),
            _runner(
                {
                    "lfs": (
                        "/lustre/fs1 1050000000 1059061760 1059061760 - "
                        "900000 1010000 1010000 -\n"
                    )
                }
            ),
        )
