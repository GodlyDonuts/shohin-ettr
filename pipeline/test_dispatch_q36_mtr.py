from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import dispatch_q36_mtr as module
from dispatch_q36_mtr import ACK, SCRIPTS, _stage_exports, submit
from q36_mtr_contract import STAGES, graph_payload

COMMIT = "1" * 40


def _args(tmp_path: Path) -> argparse.Namespace:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    environment = tmp_path / "environment.json"
    environment.write_text(
        json.dumps({"environment_tree_sha256": "e" * 64}) + "\n",
        encoding="utf-8",
    )
    values = {
        "runtime": runtime,
        "runtime_manifest_sha256": "a" * 64,
        "source_commit": COMMIT,
        "python": Path("/exact/python"),
        "run_id": "q36-test",
        "run_root": tmp_path / "run",
        "evidence_root": evidence,
        "phase_authorization": tmp_path / "authorization.json",
        "phase_authorization_sha256": "b" * 64,
        "model_root": tmp_path / "model",
        "model_manifest": tmp_path / "model" / "SHA256SUMS",
        "model_manifest_sha256": "c" * 64,
        "model_revision": "d" * 40,
        "model_config_sha256": "f" * 64,
        "environment_receipt": environment,
        "sandbox_receipt": tmp_path / "sandbox.json",
        "cluster_preflight": tmp_path / "cluster.json",
        "graph_contract": tmp_path / "graph.json",
        "plan": tmp_path / "plan.json",
        "train_sources": tmp_path / "sources/train_sources.jsonl",
        "development_sources": tmp_path / "sources/development_sources.jsonl",
        "freeze_report": tmp_path / "sources/report.json",
        "assessor_receipt": tmp_path / "custodian/receipt.json",
        "assessor_board": tmp_path / "custodian/board.jsonl",
        "b1": tmp_path / "b1.jsonl",
        "b1_sha256": "1" * 64,
        "user": "user",
        "submit_ack": ACK,
    }
    for name in ("model_root",):
        values[name].mkdir()
    for name in (
        "train_sources",
        "development_sources",
        "freeze_report",
        "assessor_receipt",
        "assessor_board",
        "b1",
    ):
        values[name].parent.mkdir(parents=True, exist_ok=True)
        values[name].write_text(name + "\n", encoding="utf-8")
    return argparse.Namespace(**values)


def test_every_frozen_stage_has_one_packaged_script_and_bound_exports(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    environment = {"environment_tree_sha256": "e" * 64}
    assert set(SCRIPTS) == {stage.name for stage in STAGES}
    for stage in STAGES:
        exports = _stage_exports(stage.name, args, environment)
        assert exports["RUN_ID"] == args.run_id
        assert exports["PHASE_AUTHORIZATION"] == str(args.phase_authorization)
        assert "ASSESSOR_BOARD" not in exports or stage.name == "score_once"
    assert _stage_exports("score_once", args, environment)["ASSESSOR_BOARD"] == str(
        args.assessor_board
    )


def test_common_cleanup_removes_frozen_job_local_tree() -> None:
    if sys.platform != "linux":
        pytest.skip("Q36 Slurm cleanup uses GNU rm semantics")
    target = Path(f"/tmp/q36-mtr-{os.getpid()}_987654")
    assert not target.exists() and not target.is_symlink()
    nested = target / "frozen" / "child"
    nested.mkdir(parents=True, mode=0o700)
    (nested / "member").write_text("staged\n", encoding="utf-8")
    for directory in (nested, nested.parent):
        directory.chmod(0o555)
    common = Path("train/jobs/q36_mtr_common.sh").resolve()
    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; SLURM_TMPDIR="$2"; q36_cleanup_local_tmp',
            "bash",
            str(common),
            str(target),
        ],
        check=True,
    )
    assert not target.exists() and not target.is_symlink()


def test_submit_prestages_exact_graph_writes_receipt_then_releases_root(
    tmp_path: Path, monkeypatch
) -> None:
    args = _args(tmp_path)
    graph = graph_payload(COMMIT)
    plan = {"h100_requests": 61}
    args.graph_contract.write_text(json.dumps(graph) + "\n", encoding="utf-8")
    args.plan.write_text(json.dumps(plan) + "\n", encoding="utf-8")
    environment = {"environment_tree_sha256": "e" * 64}
    monkeypatch.setattr(module, "preflight", lambda _args: (graph, plan, environment))
    calls = []
    next_id = iter(range(810000, 810000 + len(STAGES)))

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "sbatch":
            return subprocess.CompletedProcess(
                command, 0, stdout=f"{next(next_id)}\n", stderr=""
            )
        if command[:2] == ["scontrol", "release"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(module.subprocess, "run", run)
    receipt = submit(args)
    sbatch_calls = [command for command, _ in calls if command[0] == "sbatch"]
    assert len(sbatch_calls) == len(STAGES) == 33
    assert "--hold" in sbatch_calls[0]
    assert all("--no-requeue" in command for command in sbatch_calls)
    assert sum("--array=" in " ".join(command) for command in sbatch_calls) == 7
    assert calls[-1][0] == ["scontrol", "release", receipt["job_ids"]["preflight_cpu"]]
    assert receipt["h100_requests"] == 61
    assert receipt["expected_h100_hours"] == 58.9
    assert receipt["maximum_concurrent_single_h100_requests"] == 32
    stored = json.loads((args.run_root / "dispatch/dispatch.json").read_text())
    assert stored == receipt
    assert not (args.run_root / "final_comparison.json").exists()
