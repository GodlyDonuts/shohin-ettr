#!/usr/bin/env python3
"""Capture read-only scheduler, quota, and H100-budget admission for Q36."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Callable

from capture_q36_mtr_accounting import _allocated_gpus
from compile_q36_mtr_plan import validate_plan
from q36_mtr_contract import (
    EXCLUDED_NODES,
    MIN_FREE_BYTES,
    MIN_FREE_INODES,
    validate_graph,
)

SCHEMA = "shohin-q36-mtr-cluster-preflight-v1"
H100_HOUR_CAP = 2_000.0
PLANNED_H100_HOURS = 58.90


class Q36MTRClusterPreflightError(RuntimeError):
    """The live cluster cannot currently admit the frozen Q36 graph."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRClusterPreflightError("Q36 cluster preflight output exists")
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
    completed = runner(command, check=True, text=True, capture_output=True)
    return completed.stdout


def _quota(text: str, filesystem: str) -> dict[str, int]:
    candidates = []
    for line in text.splitlines():
        values = line.split()
        if values and values[0] == filesystem:
            candidates.append(values)
    if len(candidates) != 1:
        raise Q36MTRClusterPreflightError("Q36 quota row differs")
    values = candidates[0]
    if len(values) < 8:
        raise Q36MTRClusterPreflightError("Q36 quota columns differ")
    try:
        used_kib = int(values[1])
        hard_kib = int(values[3])
        used_inodes = int(values[5])
        hard_inodes = int(values[7])
    except ValueError as error:
        raise Q36MTRClusterPreflightError("Q36 quota values differ") from error
    free_bytes = (hard_kib - used_kib) * 1024
    free_inodes = hard_inodes - used_inodes
    if (
        min(used_kib, hard_kib, used_inodes, hard_inodes) < 0
        or free_bytes < MIN_FREE_BYTES
        or free_inodes < MIN_FREE_INODES
    ):
        raise Q36MTRClusterPreflightError("Q36 durable quota headroom is insufficient")
    return {
        "used_kib": used_kib,
        "hard_kib": hard_kib,
        "free_bytes": free_bytes,
        "used_inodes": used_inodes,
        "hard_inodes": hard_inodes,
        "free_inodes": free_inodes,
    }


def _nodes(text: str) -> list[dict[str, str]]:
    result = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = line.split("|")
        if values[-1:] == [""]:
            values.pop()
        if len(values) != 3:
            raise Q36MTRClusterPreflightError("Q36 sinfo row differs")
        node, state, gres = values
        normalized = state.split("+")[0].split("*")[0]
        if (
            node not in EXCLUDED_NODES
            and normalized in {"idle", "mix", "alloc"}
            and "nvidia_h100_pcie" in gres
        ):
            result.append({"node": node, "state": normalized, "gres": gres})
    if not result:
        raise Q36MTRClusterPreflightError("Q36 has no eligible H100 node")
    return sorted(result, key=lambda item: item["node"])


def _charged_hours(text: str) -> tuple[float, int]:
    seconds = 0
    records = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        values = line.split("|")
        if values[-1:] == [""]:
            values.pop()
        if len(values) != 4:
            raise Q36MTRClusterPreflightError("Q36 historical sacct row differs")
        job_id, elapsed_rendered, alloc_tres, restarts = values
        if "." in job_id:
            continue
        try:
            elapsed = int(elapsed_rendered)
            restart_count = int(restarts)
        except ValueError as error:
            raise Q36MTRClusterPreflightError(
                "Q36 historical accounting differs"
            ) from error
        if elapsed < 0 or restart_count < 0:
            raise Q36MTRClusterPreflightError("Q36 historical accounting is negative")
        gpus, types = _allocated_gpus(alloc_tres)
        h100s = types.get("nvidia_h100_pcie", gpus if not types else 0)
        if h100s:
            seconds += elapsed * h100s
            records += 1
    return seconds / 3600.0, records


def capture(
    args: argparse.Namespace,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    if (
        not args.graph_contract.is_absolute()
        or args.graph_contract.is_symlink()
        or not args.graph_contract.is_file()
        or not args.plan.is_absolute()
        or args.plan.is_symlink()
        or not args.plan.is_file()
        or not args.output.is_absolute()
        or args.output.parent.is_symlink()
        or not args.output.parent.is_dir()
    ):
        raise Q36MTRClusterPreflightError("Q36 preflight path differs")
    graph = json.loads(args.graph_contract.read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_graph(graph)
    validate_plan(plan)
    graph_sha256 = sha256_file(args.graph_contract)
    if (
        plan.get("graph_sha256") != graph_sha256
        or plan.get("source_commit") != graph.get("source_commit")
        or not isinstance(plan.get("expected_h100_hours"), (int, float))
        or not math.isclose(
            float(plan["expected_h100_hours"]), PLANNED_H100_HOURS, abs_tol=1e-12
        )
    ):
        raise Q36MTRClusterPreflightError("Q36 graph/plan preflight binding differs")
    queue = _run(runner, ["squeue", "-h", "-u", args.user, "-o", "%i|%T|%P|%R"])
    if queue.strip():
        raise Q36MTRClusterPreflightError("Q36 preflight queue is not empty")
    quota = _quota(
        _run(runner, ["lfs", "quota", "-u", args.user, args.filesystem]),
        args.filesystem,
    )
    nodes = _nodes(
        _run(runner, ["sinfo", "-N", "-h", "-p", "normal", "-o", "%N|%t|%G"])
    )
    charged_hours, h100_records = _charged_hours(
        _run(
            runner,
            [
                "sacct",
                "-X",
                "-n",
                "-P",
                "-u",
                args.user,
                "-S",
                args.accounting_start,
                "--format=JobIDRaw,ElapsedRaw,AllocTRES,Restarts",
            ],
        )
    )
    remaining = H100_HOUR_CAP - charged_hours
    if charged_hours < 0 or remaining < PLANNED_H100_HOURS:
        raise Q36MTRClusterPreflightError("Q36 H100-hour budget is insufficient")
    payload = {
        "schema": SCHEMA,
        "status": "pass",
        "source_commit": graph["source_commit"],
        "graph_contract_sha256": graph_sha256,
        "plan_sha256": sha256_file(args.plan),
        "user": args.user,
        "partition": "normal",
        "excluded_nodes": list(EXCLUDED_NODES),
        "queue_empty": True,
        "queue_rows": 0,
        "filesystem": args.filesystem,
        "quota": quota,
        "minimum_free_bytes": MIN_FREE_BYTES,
        "minimum_free_inodes": MIN_FREE_INODES,
        "eligible_h100_nodes": nodes,
        "eligible_h100_node_count": len(nodes),
        "accounting_start": args.accounting_start,
        "h100_hour_cap": H100_HOUR_CAP,
        "h100_hours_charged": charged_hours,
        "h100_accounting_records": h100_records,
        "planned_h100_hours": PLANNED_H100_HOURS,
        "h100_hours_remaining_before_plan": remaining,
        "h100_hours_remaining_after_plan": remaining - PLANNED_H100_HOURS,
        "scientific_rows_read": 0,
        "scientific_jobs_submitted": 0,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    _atomic_json(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True)
    parser.add_argument("--filesystem", required=True)
    parser.add_argument("--accounting-start", default="2026-08-01")
    parser.add_argument("--graph-contract", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = capture(parser.parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "free_bytes": result["quota"]["free_bytes"],
                "free_inodes": result["quota"]["free_inodes"],
                "remaining_h100_hours": result["h100_hours_remaining_after_plan"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
