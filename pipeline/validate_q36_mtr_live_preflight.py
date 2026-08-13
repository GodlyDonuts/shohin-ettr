#!/usr/bin/env python3
"""Revalidate phase admission inside the first no-science graph allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Callable

from authorize_q36_mtr_phase import SCHEMA as AUTHORIZATION_SCHEMA
from capture_q36_mtr_cluster_preflight import (
    H100_HOUR_CAP,
    Q36MTRClusterPreflightError,
    SCHEMA as CLUSTER_SCHEMA,
    _charged_hours,
    _nodes,
    _quota,
)
from pcf1_code_sandbox import validate_sandbox_receipt_payload
from q36_mtr_contract import MODEL_REVISION, validate_graph
from compile_q36_mtr_plan import validate_plan

SCHEMA = "shohin-q36-mtr-live-preflight-v1"


class Q36MTRLivePreflightError(RuntimeError):
    """The first graph allocation differs from its read-only admission."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path, schema: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise Q36MTRLivePreflightError("Q36 live preflight input differs")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise Q36MTRLivePreflightError("Q36 live preflight schema differs")
    return value


def _atomic_json(path: Path, payload: dict) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRLivePreflightError("Q36 live preflight output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _run(
    runner: Callable[..., subprocess.CompletedProcess[str]], command: list[str]
) -> str:
    return runner(command, check=True, text=True, capture_output=True).stdout


def validate(
    args: argparse.Namespace,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict:
    inputs = (
        args.phase_authorization,
        args.graph_contract,
        args.plan,
        args.environment_receipt,
        args.sandbox_receipt,
        args.cluster_preflight,
    )
    if (
        any(
            not path.is_absolute() or path.is_symlink() or not path.is_file()
            for path in inputs
        )
        or not args.output.is_absolute()
        or args.output.parent.is_symlink()
        or not args.output.parent.is_dir()
    ):
        raise Q36MTRLivePreflightError("Q36 live preflight path differs")
    authorization = _load(args.phase_authorization, AUTHORIZATION_SCHEMA)
    graph = _load(args.graph_contract, "shohin-q36-mtr-graph-v1")
    plan = _load(args.plan, "shohin-q36-mtr-execution-plan-v1")
    environment = _load(args.environment_receipt, "shohin-q36-mtr-environment-v1")
    cluster = _load(args.cluster_preflight, CLUSTER_SCHEMA)
    sandbox = json.loads(args.sandbox_receipt.read_text(encoding="utf-8"))
    validate_graph(graph)
    validate_plan(plan)
    try:
        validate_sandbox_receipt_payload(sandbox)
    except Exception as error:
        raise Q36MTRLivePreflightError("Q36 live sandbox differs") from error
    try:
        live_quota = _quota(
            _run(
                runner,
                [
                    "lfs",
                    "quota",
                    "-u",
                    str(cluster.get("user", "")),
                    str(cluster.get("filesystem", "")),
                ],
            ),
            str(cluster.get("filesystem", "")),
        )
        live_nodes = _nodes(
            _run(
                runner,
                ["sinfo", "-N", "-h", "-p", "normal", "-o", "%N|%t|%G"],
            )
        )
        charged_hours, accounting_records = _charged_hours(
            _run(
                runner,
                [
                    "sacct",
                    "-X",
                    "-n",
                    "-P",
                    "-u",
                    str(cluster.get("user", "")),
                    "-S",
                    str(cluster.get("accounting_start", "")),
                    "--format=JobIDRaw,ElapsedRaw,AllocTRES,Restarts",
                ],
            )
        )
    except Q36MTRClusterPreflightError as error:
        raise Q36MTRLivePreflightError("Q36 live cluster differs") from error
    remaining_hours = H100_HOUR_CAP - charged_hours
    if (
        authorization.get("status") != "authorized"
        or authorization.get("run_id") != args.run_id
        or authorization.get("source_commit") != graph.get("source_commit")
        or authorization.get("graph_contract_sha256")
        != sha256_file(args.graph_contract)
        or authorization.get("plan_sha256") != sha256_file(args.plan)
        or authorization.get("environment_receipt_sha256")
        != sha256_file(args.environment_receipt)
        or authorization.get("sandbox_receipt_sha256")
        != sha256_file(args.sandbox_receipt)
        or authorization.get("cluster_preflight_sha256")
        != sha256_file(args.cluster_preflight)
        or authorization.get("model_revision") != MODEL_REVISION
        or authorization.get("scientific_submit_authorized") is not True
        or authorization.get("gate") != "one_source_disjoint_development_gate"
        or authorization.get("automatic_retry") is not False
        or authorization.get("automatic_successor") is not False
        or authorization.get("automatic_confirmation") is not False
        or authorization.get("stop_after_gate") is not True
        or environment.get("status") != "pass"
        or environment.get("scientific_rows_read") != 0
        or cluster.get("status") != "pass"
        or cluster.get("queue_empty") is not True
        or cluster.get("scientific_jobs_submitted") != 0
        or remaining_hours < graph.get("expected_h100_hours", float("inf"))
    ):
        raise Q36MTRLivePreflightError("Q36 live phase authorization differs")
    payload = {
        "schema": SCHEMA,
        "status": "pass",
        "run_id": args.run_id,
        "source_commit": graph["source_commit"],
        "model_revision": MODEL_REVISION,
        "graph_contract_sha256": sha256_file(args.graph_contract),
        "plan_sha256": sha256_file(args.plan),
        "phase_authorization_sha256": sha256_file(args.phase_authorization),
        "environment_receipt_sha256": sha256_file(args.environment_receipt),
        "sandbox_receipt_sha256": sha256_file(args.sandbox_receipt),
        "cluster_preflight_sha256": sha256_file(args.cluster_preflight),
        "live_quota": live_quota,
        "live_eligible_h100_nodes": live_nodes,
        "live_eligible_h100_node_count": len(live_nodes),
        "live_h100_accounting_records": accounting_records,
        "live_h100_hours_charged": charged_hours,
        "live_h100_hours_remaining_before_plan": remaining_hours,
        "scientific_rows_read": 0,
        "capability_scored": False,
        "scientific_jobs_submitted_by_preflight": 0,
        "automatic_retry": False,
        "automatic_successor": False,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    _atomic_json(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase-authorization", type=Path, required=True)
    parser.add_argument("--graph-contract", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--sandbox-receipt", type=Path, required=True)
    parser.add_argument("--cluster-preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = validate(parser.parse_args())
    print(json.dumps({"status": result["status"], "run_id": result["run_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
