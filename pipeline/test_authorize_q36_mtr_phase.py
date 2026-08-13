from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pytest

import authorize_q36_mtr_phase as module
from authorize_q36_mtr_phase import (
    Q36MTRPhaseAuthorizationError,
    authorize,
    sha256_file,
)
from capture_q36_mtr_cluster_preflight import SCHEMA as CLUSTER_SCHEMA
from capture_q36_mtr_environment import (
    BNB_MANIFEST_SHA256,
    FAST_KERNEL_MANIFEST_SHA256,
)
from compile_q36_mtr_plan import compile_plan
from q36_mtr_contract import (
    MIN_FREE_BYTES,
    MIN_FREE_INODES,
    MODEL_REVISION,
    graph_payload,
)

COMMIT = "1" * 40


def _manifest_root(root: Path, members: dict[str, str]) -> Path:
    root.mkdir()
    rows = []
    for name, content in members.items():
        path = root / name
        path.write_text(content, encoding="utf-8")
        rows.append(f"{sha256_file(path)}  {name}")
    manifest = root / "SHA256SUMS"
    manifest.write_text("\n".join(sorted(rows)) + "\n", encoding="utf-8")
    return manifest


def _fixture(tmp_path: Path, monkeypatch) -> argparse.Namespace:
    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps(graph_payload(COMMIT)) + "\n", encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(compile_plan(graph_payload(COMMIT), sha256_file(graph))) + "\n",
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime"
    runtime_manifest = _manifest_root(
        runtime,
        {
            "runtime.json": json.dumps(
                {
                    "schema": "shohin-q36-mtr-runtime-v1",
                    "status": "complete",
                    "source_commit": COMMIT,
                    "scientific_submit_capability": True,
                    "submission_count": 1,
                }
            )
            + "\n"
        },
    )
    model = tmp_path / "model"
    model_manifest = _manifest_root(
        model,
        {
            "config.json": "{}\n",
            "SOURCE_REVISION": MODEL_REVISION + "\n",
        },
    )
    monkeypatch.setattr(
        module, "MODEL_CONFIG_SHA256", sha256_file(model / "config.json")
    )
    monkeypatch.setattr(module, "MODEL_MANIFEST_SHA256", sha256_file(model_manifest))
    environment = tmp_path / "environment.json"
    environment.write_text(
        json.dumps(
            {
                "schema": "shohin-q36-mtr-environment-v1",
                "status": "pass",
                "model_revision": MODEL_REVISION,
                "model_config_sha256": sha256_file(model / "config.json"),
                "runtime_manifest_sha256": sha256_file(runtime_manifest),
                "environment_tree_sha256": "e" * 64,
                "bitsandbytes_overlay": {
                    "manifest_sha256": BNB_MANIFEST_SHA256,
                },
                "fast_kernel_overlay": {
                    "manifest_sha256": FAST_KERNEL_MANIFEST_SHA256,
                },
                "scientific_rows_read": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sandbox = tmp_path / "sandbox.json"
    sandbox.write_text('{"sandbox":"qualified"}\n', encoding="utf-8")
    monkeypatch.setattr(module, "validate_sandbox_receipt_payload", lambda value: value)
    cluster = tmp_path / "cluster.json"
    cluster.write_text(
        json.dumps(
            {
                "schema": CLUSTER_SCHEMA,
                "status": "pass",
                "source_commit": COMMIT,
                "graph_contract_sha256": sha256_file(graph),
                "plan_sha256": sha256_file(plan),
                "queue_empty": True,
                "scientific_rows_read": 0,
                "scientific_jobs_submitted": 0,
                "eligible_h100_node_count": 1,
                "quota": {
                    "free_bytes": MIN_FREE_BYTES,
                    "free_inodes": MIN_FREE_INODES,
                },
                "h100_hours_remaining_before_plan": 1_000.0,
                "sealed_access": {"holdout": 0, "product": 0, "public": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sources = {}
    for name in ("pairs", "math", "logic_science", "code", "b1"):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(f"{name}\n", encoding="utf-8")
        sources[name] = path
    monkeypatch.setattr(
        module,
        "SOURCE_SHA256",
        {name: sha256_file(path) for name, path in sources.items()},
    )
    assessor_board = tmp_path / "confirmation_assessors.jsonl"
    assessor_board.write_text("sealed\n", encoding="utf-8")
    train_sources = tmp_path / "train_sources.jsonl"
    train_sources.write_text("train\n", encoding="utf-8")
    development_sources = tmp_path / "development_sources.jsonl"
    development_sources.write_text("development\n", encoding="utf-8")
    freeze_report = tmp_path / "freeze_report.json"
    freeze_report.write_text(
        json.dumps(
            {
                "schema": "shohin-pcf1-data-freeze-report-v1",
                "status": "complete",
                "source_disjoint": True,
                "sealed_content_materialized": False,
                "split_seed": 2026080811,
                "counts": {"train": 5824, "development": 1289, "holdout": 1279},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assessor_receipt = tmp_path / "assessor_receipt.json"
    assessor_receipt.write_text(
        json.dumps(
            {
                "schema": "shohin-pcf1-confirmation-assessor-receipt-v1",
                "status": "complete",
                "rows": 1289,
                "semantic_access": "final_score_only",
                "board_sha256": sha256_file(assessor_board),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "SOURCE_FREEZE_SHA256",
        {
            "train_sources": sha256_file(train_sources),
            "development_sources": sha256_file(development_sources),
            "freeze_report": sha256_file(freeze_report),
            "assessor_receipt": sha256_file(assessor_receipt),
            "assessor_board": sha256_file(assessor_board),
        },
    )
    repository = tmp_path / "repository"
    repository.mkdir()
    return argparse.Namespace(
        run_id="q36-run",
        repository=repository,
        graph_contract=graph,
        plan=plan,
        runtime_root=runtime,
        runtime_manifest=runtime_manifest,
        model_root=model,
        model_manifest=model_manifest,
        environment_receipt=environment,
        sandbox_receipt=sandbox,
        cluster_preflight=cluster,
        pairs=sources["pairs"],
        math=sources["math"],
        logic_science=sources["logic_science"],
        code=sources["code"],
        b1=sources["b1"],
        train_sources=train_sources,
        development_sources=development_sources,
        freeze_report=freeze_report,
        assessor_receipt=assessor_receipt,
        assessor_board=assessor_board,
        run_root=tmp_path / "run",
        output=tmp_path / "authorization.json",
    )


def _runner(dirty: bool = False):
    def run(command, **_kwargs):
        tail = command[3:]
        if tail[:2] == ["ls-remote", "--heads"]:
            stdout = (
                f"{COMMIT}\trefs/heads/{module.BRANCH}\n" if tail[2] == "origin" else ""
            )
        elif tail == ["rev-parse", "HEAD"]:
            stdout = COMMIT + "\n"
        elif tail == ["branch", "--show-current"]:
            stdout = module.BRANCH + "\n"
        elif tail == ["status", "--porcelain=v1"]:
            stdout = " M changed\n" if dirty else ""
        elif tail == ["config", "--get", "remote.origin.url"]:
            stdout = module.PRIVATE_REMOTE + "\n"
        elif tail == ["config", "--get", "remote.public.url"]:
            stdout = module.PUBLIC_REMOTE + "\n"
        elif tail == ["config", "--get", "remote.public.pushurl"]:
            stdout = module.PUBLIC_PUSH_DISABLED + "\n"
        else:
            raise AssertionError(command)
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    return run


def test_q36_phase_authorization_binds_clean_private_push_and_admission(
    tmp_path: Path, monkeypatch
) -> None:
    report = authorize(_fixture(tmp_path, monkeypatch), _runner())
    assert report["status"] == "authorized"
    assert report["scientific_submit_authorized"] is True
    assert report["automatic_retry"] is False
    assert report["automatic_successor"] is False
    assert report["public_remote_branch_present"] is False


def test_q36_phase_authorization_rejects_dirty_repository(
    tmp_path: Path, monkeypatch
) -> None:
    with pytest.raises(Q36MTRPhaseAuthorizationError):
        authorize(_fixture(tmp_path, monkeypatch), _runner(dirty=True))
