from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess

import pytest

from capture_q36_mtr_accounting import (
    DISPATCH_SCHEMA,
    Q36MTRAccountingError,
    capture,
)
from compile_q36_mtr_plan import compile_plan
from q36_mtr_contract import EXCLUDED_NODES, STAGES, graph_payload

COMMIT = "1" * 40
RUN_ID = "q36-accounting-test"


def _write(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _fixture(tmp_path: Path, phase: str = "prescore") -> tuple[argparse.Namespace, str]:
    graph_path = _write(tmp_path / "graph.json", graph_payload(COMMIT))
    import hashlib

    graph_sha256 = hashlib.sha256(graph_path.read_bytes()).hexdigest()
    plan_path = _write(
        tmp_path / "plan.json", compile_plan(graph_payload(COMMIT), graph_sha256)
    )
    plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    job_ids = {stage.name: str(10_000 + index) for index, stage in enumerate(STAGES)}
    resources = {
        stage.name: {
            "tasks": stage.tasks,
            "h100s_per_task": stage.h100_per_task,
            "is_array": stage.tasks > 1,
        }
        for stage in STAGES
    }
    dispatch = {
        "schema": DISPATCH_SCHEMA,
        "status": "submitted",
        "run_id": RUN_ID,
        "source_commit": COMMIT,
        "graph_contract_sha256": graph_sha256,
        "plan_sha256": plan_sha256,
        "partition": "normal",
        "excluded_nodes": list(EXCLUDED_NODES),
        "requeue": False,
        "retry_authorized": False,
        "successor_authorized": False,
        "preflight_queue_empty": True,
        "job_ids": job_ids,
        "stage_resources": resources,
    }
    dispatch_path = _write(tmp_path / "dispatch.json", dispatch)
    terminal = "precompute_custody" if phase == "prescore" else "normalize"
    required = [stage.name for stage in STAGES]
    required = required[: required.index(terminal) + 1]
    stage_by_name = {stage.name: stage for stage in STAGES}
    rows = []
    for name in required:
        stage = stage_by_name[name]
        for index in range(stage.tasks):
            job_id = job_ids[name] if stage.tasks == 1 else f"{job_ids[name]}_{index}"
            tres = (
                "billing=1,cpu=2,gres/gpu=1,gres/gpu:nvidia_h100_pcie=1,mem=16G"
                if stage.h100_per_task
                else "billing=1,cpu=1,mem=4G"
            )
            rows.append(
                "|".join(
                    (job_id, "COMPLETED", "normal", "60", tres, "evc20", "0:0", "0")
                )
                + "|"
            )
    args = argparse.Namespace(
        phase=phase,
        run_id=RUN_ID,
        graph_contract=graph_path,
        plan=plan_path,
        dispatch_receipt=dispatch_path,
        output=tmp_path / "accounting.json",
    )
    return args, "\n".join(rows) + "\n"


def _runner(stdout: str):
    def run(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")

    return run


def test_q36_accounting_requires_all_61_single_h100_allocations(
    tmp_path: Path,
) -> None:
    args, stdout = _fixture(tmp_path)
    report = capture(args, _runner(stdout))
    assert report["phase"] == "prescore"
    assert report["h100_request_count"] == 61
    assert report["completed_h100_allocation_count"] == 61
    assert report["charged_gpu_seconds"] == 61 * 60
    assert report["retry_count"] == report["duplicate_shard_count"] == 0
    assert len({row["request_key"] for row in report["records"]}) == len(
        report["records"]
    )


@pytest.mark.parametrize("mutation", ("drop", "restart", "wrong_gpu", "bad_node"))
def test_q36_accounting_fails_closed_on_scheduler_drift(
    tmp_path: Path, mutation: str
) -> None:
    args, stdout = _fixture(tmp_path)
    rows = stdout.splitlines()
    if mutation == "drop":
        rows.pop(2)
    else:
        parts = rows[2].split("|")
        if mutation == "restart":
            parts[7] = "1"
        elif mutation == "wrong_gpu":
            parts[4] = "billing=1,cpu=2,mem=16G"
        else:
            parts[5] = "evc26"
        rows[2] = "|".join(parts)
    with pytest.raises(Q36MTRAccountingError):
        capture(args, _runner("\n".join(rows) + "\n"))


def test_q36_accounting_rejects_dispatch_job_id_reuse(tmp_path: Path) -> None:
    args, stdout = _fixture(tmp_path)
    dispatch = json.loads(args.dispatch_receipt.read_text(encoding="utf-8"))
    changed = copy.deepcopy(dispatch)
    changed["job_ids"]["mechanics"] = changed["job_ids"]["owner_fit"]
    args.dispatch_receipt.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    with pytest.raises(Q36MTRAccountingError):
        capture(args, _runner(stdout))
