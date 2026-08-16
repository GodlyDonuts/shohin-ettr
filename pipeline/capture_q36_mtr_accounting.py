#!/usr/bin/env python3
"""Capture exact Slurm accounting for the frozen Q36 graph predecessors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable

from compile_q36_mtr_plan import validate_plan
from q36_mtr_contract import EXCLUDED_NODES, STAGES, validate_graph

SCHEMA = "shohin-q36-mtr-slurm-accounting-v1"
DISPATCH_SCHEMA = "shohin-q36-mtr-dispatch-v1"
FIELDS = (
    "JobIDRaw",
    "State",
    "Partition",
    "ElapsedRaw",
    "AllocTRES",
    "NodeList",
    "ExitCode",
    "Restarts",
)
PHASE_TERMINAL_STAGE = {"prescore": "precompute_custody", "final": "normalize"}


class Q36MTRAccountingError(RuntimeError):
    """Slurm cannot prove the exact non-retried Q36 predecessor graph."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRAccountingError("Q36 accounting output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parse_sacct(text: str) -> list[dict[str, str]]:
    records = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        values = line.split("|")
        if values[-1:] == [""]:
            values.pop()
        if len(values) != len(FIELDS):
            raise Q36MTRAccountingError(
                f"Q36 sacct field count differs on line {line_number}"
            )
        records.append(dict(zip(FIELDS, values, strict=True)))
    if not records:
        raise Q36MTRAccountingError("Q36 sacct returned no allocation records")
    return records


def _allocated_gpus(value: str) -> tuple[int, dict[str, int]]:
    generic = []
    typed: dict[str, int] = {}
    for item in value.split(","):
        key, separator, rendered = item.partition("=")
        if not separator:
            continue
        if key in {"gpu", "gres/gpu"} or key.startswith("gres/gpu:"):
            try:
                count = int(rendered)
            except ValueError as error:
                raise Q36MTRAccountingError("Q36 GPU TRES is malformed") from error
            if count < 0:
                raise Q36MTRAccountingError("Q36 GPU TRES is negative")
            if key in {"gpu", "gres/gpu"}:
                generic.append(count)
            else:
                kind = key.removeprefix("gres/gpu:")
                if not kind or kind in typed:
                    raise Q36MTRAccountingError("Q36 typed GPU TRES duplicates")
                typed[kind] = count
    if generic and len(set(generic)) != 1:
        raise Q36MTRAccountingError("Q36 generic GPU TRES conflicts")
    total = generic[0] if generic else sum(typed.values())
    if generic and typed and total != sum(typed.values()):
        raise Q36MTRAccountingError("Q36 generic and typed GPU TRES differ")
    return total, dict(sorted(typed.items()))


def _allocation_rows(
    records: list[dict[str, str]], root: str, tasks: int
) -> list[dict[str, str]]:
    matching = [
        row
        for row in records
        if "." not in row["JobIDRaw"]
        and (row["JobIDRaw"] == root or row["JobIDRaw"].startswith(f"{root}_"))
    ]
    expected = [root] if tasks == 1 else [f"{root}_{index}" for index in range(tasks)]
    by_id = {row["JobIDRaw"]: row for row in matching}
    if len(matching) != len(expected) or set(by_id) != set(expected):
        raise Q36MTRAccountingError(f"Q36 allocation geometry differs: {root}")
    return [by_id[value] for value in expected]


def _validate_row(
    row: dict[str, str], expected_gpus: int, request_key: str
) -> tuple[dict[str, Any], int]:
    state = row["State"].split()[0].split("+")[0]
    if state != "COMPLETED" or row["ExitCode"] != "0:0" or row["Restarts"] != "0":
        raise Q36MTRAccountingError(
            f"Q36 allocation did not complete once: {request_key}"
        )
    if row["Partition"] != "normal" or any(
        re.search(rf"(?<![A-Za-z0-9]){node}(?![A-Za-z0-9])", row["NodeList"])
        for node in EXCLUDED_NODES
    ):
        raise Q36MTRAccountingError(f"Q36 allocation host differs: {request_key}")
    try:
        elapsed = int(row["ElapsedRaw"])
    except ValueError as error:
        raise Q36MTRAccountingError("Q36 elapsed time is malformed") from error
    if elapsed < 0:
        raise Q36MTRAccountingError("Q36 elapsed time is negative")
    gpus, typed = _allocated_gpus(row["AllocTRES"])
    if gpus != expected_gpus or (
        expected_gpus == 1 and typed not in ({}, {"nvidia_h100_pcie": 1})
    ):
        raise Q36MTRAccountingError(f"Q36 GPU allocation differs: {request_key}")
    normalized = {
        "request_key": request_key,
        "job_id_raw": row["JobIDRaw"],
        "state": state,
        "partition": row["Partition"],
        "elapsed_raw": elapsed,
        "alloc_tres": row["AllocTRES"],
        "allocated_gpus": gpus,
        "allocated_gpu_types": typed,
        "node_list": row["NodeList"],
        "exit_code": row["ExitCode"],
        "restarts": 0,
        "charged_gpu_seconds": elapsed * gpus,
    }
    return normalized, elapsed * gpus


def _required_stages(phase: str) -> list[str]:
    terminal = PHASE_TERMINAL_STAGE[phase]
    names = [stage.name for stage in STAGES]
    return names[: names.index(terminal) + 1]


def capture(
    args: argparse.Namespace,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    graph = json.loads(args.graph_contract.read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    dispatch = json.loads(args.dispatch_receipt.read_text(encoding="utf-8"))
    validate_graph(graph)
    validate_plan(plan)
    graph_sha256 = sha256_file(args.graph_contract)
    plan_sha256 = sha256_file(args.plan)
    required = _required_stages(args.phase)
    job_ids = dispatch.get("job_ids")
    resources = dispatch.get("stage_resources")
    all_stages = [stage.name for stage in STAGES]
    if (
        plan.get("source_commit") != graph.get("source_commit")
        or plan.get("graph_sha256") != graph_sha256
        or dispatch.get("schema") != DISPATCH_SCHEMA
        or dispatch.get("status") != "submitted"
        or dispatch.get("run_id") != args.run_id
        or dispatch.get("source_commit") != graph.get("source_commit")
        or dispatch.get("graph_contract_sha256") != graph_sha256
        or dispatch.get("plan_sha256") != plan_sha256
        or dispatch.get("partition") != "normal"
        or dispatch.get("excluded_nodes") != list(EXCLUDED_NODES)
        or dispatch.get("requeue") is not False
        or dispatch.get("retry_authorized") is not False
        or dispatch.get("successor_authorized") is not False
        or dispatch.get("preflight_queue_empty") is not True
        or not isinstance(job_ids, dict)
        or set(job_ids) != set(all_stages)
        or not isinstance(resources, dict)
        or set(resources) != set(all_stages)
        or len(set(job_ids.values())) != len(all_stages)
    ):
        raise Q36MTRAccountingError("Q36 dispatch receipt differs")
    stage_by_name = {stage.name: stage for stage in STAGES}
    for name in all_stages:
        stage = stage_by_name[name]
        resource = resources.get(name)
        if (
            not isinstance(job_ids[name], str)
            or not job_ids[name].isdigit()
            or resource
            != {
                "tasks": stage.tasks,
                "h100s_per_task": stage.h100_per_task,
                "is_array": stage.tasks > 1,
            }
        ):
            raise Q36MTRAccountingError(f"Q36 dispatch stage differs: {name}")
    roots = [job_ids[name] for name in required]
    completed = runner(
        [
            "sacct",
            "-n",
            "-P",
            "-j",
            ",".join(roots),
            "--format=" + ",".join(FIELDS),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    records = _parse_sacct(completed.stdout)
    normalized = []
    charged_gpu_seconds = 0
    seen_job_ids = set()
    for name in required:
        stage = stage_by_name[name]
        rows = _allocation_rows(records, job_ids[name], stage.tasks)
        for index, row in enumerate(rows):
            request_key = name if stage.h100_per_task == 0 else f"{name}/{index:02d}"
            value, charged = _validate_row(row, stage.h100_per_task, request_key)
            if value["job_id_raw"] in seen_job_ids:
                raise Q36MTRAccountingError("Q36 allocation is duplicated")
            seen_job_ids.add(value["job_id_raw"])
            normalized.append(value)
            charged_gpu_seconds += charged
    h100_records = [row for row in normalized if row["allocated_gpus"] == 1]
    expected_h100_keys = {
        task["request_key"] for task in plan["gpu_tasks"] if task["stage"] in required
    }
    if (
        {row["request_key"] for row in h100_records} != expected_h100_keys
        or len(h100_records) != 61
        or len(normalized) != sum(stage_by_name[name].tasks for name in required)
    ):
        raise Q36MTRAccountingError("Q36 accounted request set differs")
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "phase": args.phase,
        "run_id": args.run_id,
        "source_commit": graph["source_commit"],
        "graph_contract_sha256": graph_sha256,
        "plan_sha256": plan_sha256,
        "dispatch_receipt_sha256": sha256_file(args.dispatch_receipt),
        "required_stages": required,
        "records": normalized,
        "record_count": len(normalized),
        "h100_request_count": 61,
        "completed_h100_allocation_count": len(h100_records),
        "charged_gpu_seconds": charged_gpu_seconds,
        "retry_count": 0,
        "requeue_count": 0,
        "duplicate_shard_count": 0,
        "orphaned_job_count": 0,
        "successor_authorized": False,
        "successor_submitted": False,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    _atomic_json(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(PHASE_TERMINAL_STAGE), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--graph-contract", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dispatch-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = capture(parser.parse_args())
    print(json.dumps({"status": result["status"], "phase": result["phase"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
