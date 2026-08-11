#!/usr/bin/env python3
"""Capture one fail-closed sacct receipt for every pre-score PCF1 job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Callable

SCHEMA = "shohin-pcf1-slurm-accounting-v1"
DISPATCH_SCHEMA = "shohin-pcf1-dispatch-v1"
PARTITION = "normal"
EXCLUDED = ("evc26", "evc29", "evc31", "evc32", "evc33", "evc38", "evc46")
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


class PCF1AccountingError(RuntimeError):
    """Slurm accounting cannot prove the one PCF1 pre-score graph complete."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_protected_path(path: Path) -> None:
    if any(word in str(path).casefold() for word in ("holdout", "product", "public")):
        raise PCF1AccountingError(f"protected PCF1 accounting path: {path}")


def parse_sacct(text: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        values = line.split("|")
        if values[-1] == "":
            values.pop()
        if len(values) != len(FIELDS):
            raise PCF1AccountingError(
                f"sacct field count differs at line {line_number}"
            )
        records.append(dict(zip(FIELDS, values, strict=True)))
    if not records:
        raise PCF1AccountingError("sacct returned no PCF1 records")
    return records


def allocation_records(
    records: list[dict[str, str]],
    root: str,
    *,
    is_array: bool,
    array_tasks: int,
) -> list[dict[str, str]]:
    matching = [
        row
        for row in records
        if "." not in row["JobIDRaw"]
        and (row["JobIDRaw"] == root or row["JobIDRaw"].startswith(f"{root}_"))
    ]
    if is_array:
        array_rows = [row for row in matching if row["JobIDRaw"].startswith(f"{root}_")]
        by_id = {row["JobIDRaw"]: row for row in array_rows}
        expected = [f"{root}_{index}" for index in range(array_tasks)]
        if (
            len(array_rows) != array_tasks
            or len(by_id) != array_tasks
            or set(by_id) != set(expected)
        ):
            raise PCF1AccountingError(
                f"PCF1 array allocation geometry differs for job {root}"
            )
        return [by_id[job_id] for job_id in expected]
    scalar = [row for row in matching if row["JobIDRaw"] == root]
    array_rows = [row for row in matching if row["JobIDRaw"].startswith(f"{root}_")]
    if len(scalar) != 1 or array_rows or array_tasks != 1:
        raise PCF1AccountingError(
            f"PCF1 scalar allocation geometry differs for job {root}"
        )
    if not scalar:
        raise PCF1AccountingError(f"missing sacct allocation for job {root}")
    return scalar


def allocated_gpus(alloc_tres: str) -> tuple[int, dict[str, int]]:
    generic: list[int] = []
    typed: dict[str, int] = {}
    for item in alloc_tres.split(","):
        key, separator, value = item.partition("=")
        if separator and (key in {"gres/gpu", "gpu"} or key.startswith("gres/gpu:")):
            try:
                count = int(value)
            except ValueError as error:
                raise PCF1AccountingError("invalid GPU AllocTRES") from error
            if count < 0:
                raise PCF1AccountingError("negative GPU AllocTRES")
            if key in {"gres/gpu", "gpu"}:
                generic.append(count)
            else:
                gpu_type = key.removeprefix("gres/gpu:")
                if not gpu_type or gpu_type in typed:
                    raise PCF1AccountingError("duplicate typed GPU AllocTRES")
                typed[gpu_type] = count
    if generic and len(set(generic)) != 1:
        raise PCF1AccountingError("conflicting generic GPU AllocTRES")
    typed_total = sum(typed.values())
    total = generic[0] if generic else typed_total
    if generic and typed and total != typed_total:
        raise PCF1AccountingError("generic/typed GPU AllocTRES differs")
    return total, dict(sorted(typed.items()))


def validate_record(row: dict[str, str]) -> tuple[dict[str, Any], int]:
    state = row["State"].split()[0].split("+")[0]
    if state != "COMPLETED" or row["ExitCode"] != "0:0":
        raise PCF1AccountingError(
            f"PCF1 job is not complete: {row['JobIDRaw']} {row['State']} {row['ExitCode']}"
        )
    if row["Restarts"] != "0":
        raise PCF1AccountingError(
            f"PCF1 job restarted: {row['JobIDRaw']} {row['Restarts']}"
        )
    if row["Partition"] != PARTITION:
        raise PCF1AccountingError(f"PCF1 job used wrong partition: {row['JobIDRaw']}")
    if any(
        re.search(rf"(?<![A-Za-z0-9]){node}(?![A-Za-z0-9])", row["NodeList"])
        for node in EXCLUDED
    ):
        raise PCF1AccountingError(
            f"PCF1 job used excluded node: {row['JobIDRaw']} {row['NodeList']}"
        )
    try:
        elapsed = int(row["ElapsedRaw"])
    except ValueError as error:
        raise PCF1AccountingError("invalid PCF1 elapsed time") from error
    if elapsed < 0:
        raise PCF1AccountingError("negative PCF1 elapsed time")
    gpus, gpu_types = allocated_gpus(row["AllocTRES"])
    normalized = {
        "job_id_raw": row["JobIDRaw"],
        "state": state,
        "partition": row["Partition"],
        "elapsed_raw": elapsed,
        "alloc_tres": row["AllocTRES"],
        "node_list": row["NodeList"],
        "exit_code": row["ExitCode"],
        "restarts": 0,
        "allocated_gpus": gpus,
        "allocated_gpu_types": gpu_types,
        "charged_gpu_seconds": elapsed * gpus,
    }
    return normalized, elapsed * gpus


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise PCF1AccountingError(f"refusing existing accounting receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except FileExistsError as error:
        raise PCF1AccountingError("PCF1 accounting publication race") from error
    finally:
        temporary.unlink(missing_ok=True)


def capture(
    args: argparse.Namespace,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    reject_protected_path(args.dispatch_receipt)
    reject_protected_path(args.output)
    dispatch = json.loads(args.dispatch_receipt.read_text(encoding="utf-8"))
    predecessors = dispatch.get("accounting_predecessors")
    job_ids = dispatch.get("job_ids")
    stage_resources = dispatch.get("stage_resources")
    if (
        dispatch.get("schema") != DISPATCH_SCHEMA
        or dispatch.get("status") != "submitted"
        or dispatch.get("run_id") != args.run_id
        or dispatch.get("partition") != PARTITION
        or dispatch.get("excluded_nodes") != list(EXCLUDED)
        or dispatch.get("retry_authorized") is not False
        or dispatch.get("successor_authorized") is not False
        or dispatch.get("stop_after_gate") is not True
        or not isinstance(predecessors, list)
        or not predecessors
        or len(predecessors) != len(set(predecessors))
        or not isinstance(job_ids, dict)
        or not isinstance(stage_resources, dict)
        or any(
            not isinstance(stage, str)
            or not isinstance(job_ids.get(stage), str)
            or not job_ids[stage].isdigit()
            for stage in predecessors
        )
        or any(
            not isinstance(stage_resources.get(stage), dict)
            or stage_resources[stage].get("gpus") not in (0, 1)
            or not isinstance(stage_resources[stage].get("is_array"), bool)
            or isinstance(stage_resources[stage].get("array_tasks"), bool)
            or not isinstance(stage_resources[stage].get("array_tasks"), int)
            or stage_resources[stage]["array_tasks"] <= 0
            or (
                not stage_resources[stage]["is_array"]
                and stage_resources[stage]["array_tasks"] != 1
            )
            for stage in predecessors
        )
    ):
        raise PCF1AccountingError("PCF1 dispatch accounting boundary differs")
    roots = [job_ids[stage] for stage in predecessors]
    if len(set(roots)) != len(roots):
        raise PCF1AccountingError("PCF1 dispatch reuses a Slurm job ID")
    command = [
        "sacct",
        "-n",
        "-P",
        "-j",
        ",".join(roots),
        "--format=" + ",".join(FIELDS),
    ]
    completed = runner(command, check=True, text=True, capture_output=True)
    raw_records = parse_sacct(completed.stdout)
    jobs: dict[str, Any] = {}
    total_gpu_seconds = 0
    for stage in predecessors:
        root = job_ids[stage]
        normalized: list[dict[str, Any]] = []
        stage_gpu_seconds = 0
        resource = stage_resources[stage]
        for record in allocation_records(
            raw_records,
            root,
            is_array=resource["is_array"],
            array_tasks=resource["array_tasks"],
        ):
            item, gpu_seconds = validate_record(record)
            expected_gpus = stage_resources[stage]["gpus"]
            expected_types = {"nvidia_h100_pcie": 1} if expected_gpus else {}
            if (
                item["allocated_gpus"] != expected_gpus
                or item["allocated_gpu_types"] != expected_types
            ):
                raise PCF1AccountingError(
                    f"PCF1 GPU allocation differs: {stage} "
                    f"expected={expected_gpus}/{expected_types} "
                    f"actual={item['allocated_gpus']}/{item['allocated_gpu_types']}"
                )
            normalized.append(item)
            stage_gpu_seconds += gpu_seconds
        jobs[stage] = {
            "submitted_job_id": root,
            "records": normalized,
            "charged_gpu_seconds": stage_gpu_seconds,
        }
        total_gpu_seconds += stage_gpu_seconds
    receipt = {
        "schema": SCHEMA,
        "status": "complete",
        "run_id": args.run_id,
        "dispatch_receipt": str(args.dispatch_receipt.resolve()),
        "dispatch_receipt_sha256": sha256_file(args.dispatch_receipt),
        "partition": PARTITION,
        "excluded_nodes": list(EXCLUDED),
        "required_stages": predecessors,
        "jobs": jobs,
        "charged_gpu_seconds": total_gpu_seconds,
        "all_required_complete": True,
        "retry_count": 0,
        "successor_authorized": False,
        "successor_submitted": False,
    }
    atomic_json(args.output, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dispatch-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    receipt = capture(parser.parse_args())
    print(
        json.dumps(
            {"charged_gpu_seconds": receipt["charged_gpu_seconds"]}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
