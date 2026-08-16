#!/usr/bin/env python3
"""Capture exact Slurm allocation accounting for one upward-MoE result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from capture_q36_mtr_accounting import _allocated_gpus

SCHEMA = "shohin-upward-moe-slurm-accounting-v1"
FIELDS = (
    "JobIDRaw",
    "State",
    "ExitCode",
    "ElapsedRaw",
    "AllocTRES",
    "Partition",
    "NodeList",
    "Restarts",
    "Start",
    "End",
)
JOB_ID = re.compile(r"^[0-9]+(?:_[0-9]+)?$")
STAGE = re.compile(r"^[a-z][a-z0-9_]*$")


class UpwardMoEAccountingError(RuntimeError):
    """The completed Slurm graph cannot support exact compute accounting."""


def _parse_allocation(value: str) -> tuple[str, str, int]:
    parts = value.split(",")
    if (
        len(parts) != 3
        or not STAGE.fullmatch(parts[0])
        or not JOB_ID.fullmatch(parts[1])
    ):
        raise UpwardMoEAccountingError("allocation specification differs")
    try:
        expected_gpus = int(parts[2])
    except ValueError as error:
        raise UpwardMoEAccountingError("allocation GPU count differs") from error
    if expected_gpus < 0:
        raise UpwardMoEAccountingError("allocation GPU count differs")
    return parts[0], parts[1], expected_gpus


def _parse_sacct(value: str) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for number, line in enumerate(value.splitlines(), start=1):
        if not line:
            continue
        fields = line.split("|")
        if len(fields) != len(FIELDS):
            raise UpwardMoEAccountingError(
                f"sacct field count differs on line {number}"
            )
        record = dict(zip(FIELDS, fields, strict=True))
        job_id = record["JobIDRaw"]
        if job_id in records:
            raise UpwardMoEAccountingError("duplicate sacct allocation")
        records[job_id] = record
    if not records:
        raise UpwardMoEAccountingError("sacct returned no allocations")
    return records


def _nonnegative_integer(value: str, label: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise UpwardMoEAccountingError(f"{label} differs") from error
    if result < 0:
        raise UpwardMoEAccountingError(f"{label} differs")
    return result


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise UpwardMoEAccountingError("accounting output exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o400)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def capture(
    *,
    host: str,
    source_commit: str,
    allocations: list[str],
    output: Path,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    if not host or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise UpwardMoEAccountingError("accounting identity differs")
    parsed = [_parse_allocation(value) for value in allocations]
    if not parsed or len({job_id for _, job_id, _ in parsed}) != len(parsed):
        raise UpwardMoEAccountingError("allocation job identities differ")
    requested = [job_id for _, job_id, _ in parsed]
    completed = runner(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(requested),
            f"--format={','.join(FIELDS)}",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    raw = _parse_sacct(completed.stdout)
    if set(raw) != set(requested):
        raise UpwardMoEAccountingError("sacct allocation coverage differs")

    stages: dict[str, list[dict[str, Any]]] = {}
    charged_gpu_seconds = 0
    for stage, job_id, expected_gpus in parsed:
        row = raw[job_id]
        state = row["State"].split()[0].split("+")[0]
        if (
            state != "COMPLETED"
            or row["ExitCode"] != "0:0"
            or _nonnegative_integer(row["Restarts"], "restarts") != 0
        ):
            raise UpwardMoEAccountingError("allocation did not complete exactly once")
        if row["Partition"] != "normal" or not row["NodeList"]:
            raise UpwardMoEAccountingError("allocation placement differs")
        elapsed = _nonnegative_integer(row["ElapsedRaw"], "elapsed")
        gpus, gpu_types = _allocated_gpus(row["AllocTRES"])
        if gpus != expected_gpus:
            raise UpwardMoEAccountingError("allocated GPU count differs")
        if expected_gpus > 0 and gpu_types != {"nvidia_h100_pcie": expected_gpus}:
            raise UpwardMoEAccountingError("allocated GPU type differs")
        charged = elapsed * gpus
        charged_gpu_seconds += charged
        stages.setdefault(stage, []).append(
            {
                "job_id": job_id,
                "state": state,
                "exit_code": row["ExitCode"],
                "elapsed_seconds": elapsed,
                "allocated_gpus": gpus,
                "gpu_types": gpu_types,
                "charged_gpu_seconds": charged,
                "node": row["NodeList"],
                "partition": row["Partition"],
                "restarts": 0,
                "start": row["Start"],
                "end": row["End"],
                "alloc_tres": row["AllocTRES"],
            }
        )
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "host": host,
        "source_commit": source_commit,
        "allocation_count": len(parsed),
        "stages": stages,
        "charged_gpu_seconds": charged_gpu_seconds,
        "charged_h100_hours": charged_gpu_seconds / 3600.0,
        "all_required_complete": True,
        "retry_count": 0,
    }
    if not math.isfinite(payload["charged_h100_hours"]):
        raise UpwardMoEAccountingError("charged H100 hours differ")
    payload["allocation_identity_sha256"] = hashlib.sha256(
        json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic_json(output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--allocation", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    capture(
        host=arguments.host,
        source_commit=arguments.source_commit,
        allocations=arguments.allocation,
        output=arguments.output,
    )
