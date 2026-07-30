#!/usr/bin/env python3
"""Assemble a fresh ETTR materialization tree from verified worker outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
from typing import Mapping, Sequence

import materialize_ettr_il_v3_corpus as original

from accelerate_ettr_il_v3_materialization import (
    ADAPTER_SCHEMA,
    CONFIRMATION_ADAPTER_SCHEMA,
)


ASSEMBLY_SCHEMA = "r12-ettr-il-v3-materialization-recovery-assembly-v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class RecoveryAssemblyError(ValueError):
    """Worker outputs cannot be assembled without weakening custody."""


def _stable_sha256(path: Path, label: str) -> str:
    before = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise RecoveryAssemblyError(
            f"{label} is not a single-link regular file"
        )
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.lstat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_nlink,
    ):
        raise RecoveryAssemblyError(f"{label} changed during measurement")
    return digest.hexdigest()


def _copy_no_replace(
    source_path: Path,
    destination_path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    label: str,
) -> None:
    before = source_path.lstat()
    if (
        source_path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_mode & 0o222
    ):
        raise RecoveryAssemblyError(
            f"{label} is not an immutable single-link regular file"
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    source_flags = os.O_RDONLY
    source_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    output_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    output_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    source_descriptor = os.open(source_path, source_flags)
    output_descriptor = -1
    digest = hashlib.sha256()
    size = 0
    try:
        opened = os.fstat(source_descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
        ) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
            opened.st_nlink,
        ):
            raise RecoveryAssemblyError(f"{label} changed before copying")
        output_descriptor = os.open(destination_path, output_flags, 0o400)
        while block := os.read(source_descriptor, 8 * 1024 * 1024):
            view = memoryview(block)
            while view:
                written = os.write(output_descriptor, view)
                view = view[written:]
            digest.update(block)
            size += len(block)
        os.fsync(output_descriptor)
        finished = os.fstat(source_descriptor)
        after = source_path.lstat()
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
        )
        if identity != (
            finished.st_dev,
            finished.st_ino,
            finished.st_size,
            finished.st_mtime_ns,
            finished.st_ctime_ns,
            finished.st_nlink,
        ) or identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        ):
            raise RecoveryAssemblyError(f"{label} changed during copying")
    except BaseException:
        destination_path.unlink(missing_ok=True)
        raise
    finally:
        os.close(source_descriptor)
        if output_descriptor >= 0:
            os.close(output_descriptor)
    if size != expected_bytes or digest.hexdigest() != expected_sha256:
        destination_path.unlink(missing_ok=True)
        raise RecoveryAssemblyError(f"{label} identity differs")
    copied = destination_path.lstat()
    if (
        not stat.S_ISREG(copied.st_mode)
        or copied.st_nlink != 1
        or copied.st_mode & 0o222
        or copied.st_size != expected_bytes
    ):
        destination_path.unlink(missing_ok=True)
        raise RecoveryAssemblyError(f"{label} copied identity differs")


def _validate_worker_report(
    report: Mapping[str, object],
    *,
    task: Mapping[str, object],
    task_manifest_sha256: str,
) -> str:
    report_sha256 = original._verify_self_hash(
        report,
        field="report_sha256",
        label="worker report",
    )
    if (
        report.get("schema") != original.WORKER_SCHEMA
        or report.get("status") != "pass"
        or report.get("protocol") != original.PROTOCOL
        or report.get("task") != task
        or report.get("task_manifest_sha256") != task_manifest_sha256
        or report.get("output_path") != task["output_path"]
        or report.get("row_count") != task["input_rows"]
        or not isinstance(report.get("output_bytes"), int)
        or _HEX64.fullmatch(str(report.get("output_sha256", ""))) is None
    ):
        raise RecoveryAssemblyError("worker report contract differs")
    return report_sha256


def _parse_recovery(value: str) -> tuple[int, Path, Path]:
    fields = value.split("|")
    if len(fields) != 3 or not fields[0].isdigit():
        raise argparse.ArgumentTypeError(
            "recovery must be TASK_INDEX|OUTPUT_ROOT|REPORTS_ROOT"
        )
    output_root = Path(fields[1])
    reports_root = Path(fields[2])
    if not output_root.is_absolute() or not reports_root.is_absolute():
        raise argparse.ArgumentTypeError("recovery roots must be absolute")
    return int(fields[0]), output_root, reports_root


def assemble_recovery(
    task_manifest: Path,
    primary_output_root: Path,
    primary_reports_root: Path,
    output_root: Path,
    reports_root: Path,
    receipt_path: Path,
    recoveries: Mapping[int, tuple[Path, Path]],
    *,
    execution_source_commit: str,
    execution_code_sha256: str,
    expected_adapter_source_commit: str,
    expected_adapter_sha256: str,
    execution_code: Path | None = None,
) -> dict[str, object]:
    """Copy an exact serial/recovery worker inventory into a fresh tree."""

    for value, expression, label in (
        (execution_source_commit, _HEX40, "assembly source commit"),
        (execution_code_sha256, _HEX64, "assembly source SHA-256"),
        (expected_adapter_source_commit, _HEX40, "adapter source commit"),
        (expected_adapter_sha256, _HEX64, "adapter source SHA-256"),
    ):
        if expression.fullmatch(value) is None:
            raise RecoveryAssemblyError(f"{label} differs")
    code_path = (execution_code or Path(__file__)).resolve()
    if _stable_sha256(code_path, "assembly source") != execution_code_sha256:
        raise RecoveryAssemblyError("assembly source SHA-256 differs")
    if (
        output_root.exists()
        or output_root.is_symlink()
        or reports_root.exists()
        or reports_root.is_symlink()
        or receipt_path.exists()
        or receipt_path.is_symlink()
    ):
        raise RecoveryAssemblyError("assembly destination already exists")

    manifest, task_manifest_sha256 = original._load_tasks(task_manifest)
    role = manifest.get("role")
    tasks = manifest.get("tasks")
    if role not in {"main", "sealed_confirmation"} or not isinstance(tasks, list):
        raise RecoveryAssemblyError("assembly task manifest differs")
    if any(index < 0 or index >= len(tasks) for index in recoveries):
        raise RecoveryAssemblyError("recovery task index differs")

    output_root.mkdir(parents=True, mode=0o700)
    reports_root.mkdir(parents=True, mode=0o700)
    descriptors: list[dict[str, object]] = []
    replacements: list[int] = []
    try:
        for index, task_value in enumerate(tasks):
            if not isinstance(task_value, dict) or task_value.get("index") != index:
                raise RecoveryAssemblyError("task descriptor differs")
            task = task_value
            if index in recoveries:
                source_output_root, source_reports_root = recoveries[index]
                replacements.append(index)
            else:
                source_output_root = primary_output_root
                source_reports_root = primary_reports_root
            relative_output = original._relative(
                task["output_path"],
                "task output path",
            )
            relative_report = original._relative(
                task["report_path"],
                "task report path",
            )
            source_report = source_reports_root / relative_report
            report_payload = original._regular_bytes(
                source_report,
                "worker report",
                original.MAX_METADATA_BYTES,
            )
            report = original._strict_load(report_payload, "worker report")
            report_sha256 = _validate_worker_report(
                report,
                task=task,
                task_manifest_sha256=task_manifest_sha256,
            )
            if index in recoveries:
                expected_adapter: dict[str, object] = {
                    "schema": (
                        ADAPTER_SCHEMA
                        if role == "main"
                        else CONFIRMATION_ADAPTER_SCHEMA
                    ),
                    "source_commit": expected_adapter_source_commit,
                    "source_sha256": expected_adapter_sha256,
                    "workers": report.get("execution_adapter", {}).get("workers")
                    if isinstance(report.get("execution_adapter"), dict)
                    else None,
                }
                if role == "sealed_confirmation":
                    expected_adapter["role"] = role
                if report.get("execution_adapter") != expected_adapter:
                    raise RecoveryAssemblyError(
                        "recovery execution adapter differs"
                    )
                adapter = report["execution_adapter"]
                assert isinstance(adapter, dict)
                workers = adapter.get("workers")
                if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 64:
                    raise RecoveryAssemblyError("recovery worker count differs")
            output_sha256 = str(report["output_sha256"])
            output_bytes = int(report["output_bytes"])
            _copy_no_replace(
                source_output_root / relative_output,
                output_root / relative_output,
                expected_sha256=output_sha256,
                expected_bytes=output_bytes,
                label="worker output",
            )
            _copy_no_replace(
                source_report,
                reports_root / relative_report,
                expected_sha256=hashlib.sha256(report_payload).hexdigest(),
                expected_bytes=len(report_payload),
                label="worker report",
            )
            descriptors.append(
                {
                    "index": index,
                    "output_bytes": output_bytes,
                    "output_path": relative_output,
                    "output_sha256": output_sha256,
                    "recovery": index in recoveries,
                    "report_path": relative_report,
                    "report_sha256": report_sha256,
                }
            )
        for root in (output_root, reports_root):
            for directory, _, _ in os.walk(root, topdown=False):
                Path(directory).chmod(0o500)
        receipt: dict[str, object] = {
            "assembly_source": {
                "commit": execution_source_commit,
                "sha256": execution_code_sha256,
            },
            "expected_adapter": {
                "schema": (
                    ADAPTER_SCHEMA
                    if role == "main"
                    else CONFIRMATION_ADAPTER_SCHEMA
                ),
                "source_commit": expected_adapter_source_commit,
                "source_sha256": expected_adapter_sha256,
            },
            "protocol": original.PROTOCOL,
            "replacement_indices": replacements,
            "role": role,
            "schema": ASSEMBLY_SCHEMA,
            "shards": descriptors,
            "status": "pass",
            "task_count": len(tasks),
            "task_manifest_sha256": task_manifest_sha256,
        }
        receipt["assembly_sha256"] = hashlib.sha256(
            original.canonical_json_bytes(receipt)
        ).hexdigest()
        original._write_no_replace(
            receipt_path,
            original.canonical_json_bytes(receipt),
        )
        return receipt
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        shutil.rmtree(reports_root, ignore_errors=True)
        receipt_path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--primary-output-root", type=Path, required=True)
    parser.add_argument("--primary-reports-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--recovery",
        action="append",
        default=[],
        type=_parse_recovery,
    )
    parser.add_argument("--execution-source-commit", required=True)
    parser.add_argument("--execution-code-sha256", required=True)
    parser.add_argument("--expected-adapter-source-commit", required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    recoveries: dict[int, tuple[Path, Path]] = {}
    for index, output_root, reports_root in arguments.recovery:
        if index in recoveries:
            raise RecoveryAssemblyError("recovery task index repeats")
        recoveries[index] = (output_root, reports_root)
    receipt = assemble_recovery(
        arguments.task_manifest,
        arguments.primary_output_root,
        arguments.primary_reports_root,
        arguments.output_root,
        arguments.reports_root,
        arguments.receipt,
        recoveries,
        execution_source_commit=arguments.execution_source_commit,
        execution_code_sha256=arguments.execution_code_sha256,
        expected_adapter_source_commit=arguments.expected_adapter_source_commit,
        expected_adapter_sha256=arguments.expected_adapter_sha256,
    )
    print(
        json.dumps(
            {
                "assembly_sha256": receipt["assembly_sha256"],
                "replacement_indices": receipt["replacement_indices"],
                "status": receipt["status"],
                "task_count": receipt["task_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
