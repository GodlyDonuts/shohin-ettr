#!/usr/bin/env python3
"""Parallel adapter for exact frozen ETTR-IL-v3 materialization cells."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import stat
from typing import Mapping, Sequence

from ettr_il_v2_token_native_surface import TokenNativeSurfaceCodec
from ettr_il_v3_materialize import materialize_candidate
import materialize_ettr_il_v3_corpus as original


ADAPTER_SCHEMA = "r12-ettr-il-v3-parallel-materializer-adapter-v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_WORKER_CODEC: TokenNativeSurfaceCodec | None = None


class ParallelMaterializationError(ValueError):
    """The parallel adapter cannot preserve the frozen materialization contract."""


def _stable_sha256(path: Path, label: str) -> str:
    before = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise ParallelMaterializationError(
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
        raise ParallelMaterializationError(f"{label} changed during measurement")
    return digest.hexdigest()


def _initialize_worker(tokenizer_path: str) -> None:
    global _WORKER_CODEC
    _WORKER_CODEC = TokenNativeSurfaceCodec(Path(tokenizer_path))


def _materialize_one(
    indexed_payload: tuple[int, bytes],
) -> tuple[int, bytes, str, str, str, str]:
    index, payload = indexed_payload
    if _WORKER_CODEC is None:
        raise ParallelMaterializationError("worker codec is unavailable")
    if len(payload) > original.MAX_ROW_BYTES:
        raise ParallelMaterializationError("selected candidate row exceeds size bound")
    candidate = original._strict_load(payload, "selected candidate row")
    record = materialize_candidate(candidate, _WORKER_CODEC, confirmation_key=None)
    return (
        index,
        record.canonical_bytes(),
        record.identity.split,
        record.identity.curriculum_stage,
        record.identity.core_id,
        record.assessor_only.audit.semantic_hash,
    )


def _report(
    *,
    manifest: Mapping[str, object],
    task: Mapping[str, object],
    task_manifest_sha256: str,
    codec: TokenNativeSurfaceCodec,
    rows: int,
    uncompressed_bytes: int,
    output_sha256: str,
    output_bytes: int,
    execution_source_commit: str,
    execution_code_sha256: str,
    workers: int,
) -> dict[str, object]:
    report: dict[str, object] = {
        "charged_positions": (
            rows * original.ROWS_PER_CORE * original.CHARGED_POSITIONS_PER_ROW
        ),
        "codebook_sha256": codec.codebook_sha256,
        "execution_adapter": {
            "schema": ADAPTER_SCHEMA,
            "source_commit": execution_source_commit,
            "source_sha256": execution_code_sha256,
            "workers": workers,
        },
        "expanded_rows": rows * original.ROWS_PER_CORE,
        "materializer_freeze_sha256": manifest["materializer_freeze_sha256"],
        "materializer_source_commit": manifest["materializer_source_commit"],
        "output_bytes": output_bytes,
        "output_path": task["output_path"],
        "output_sha256": output_sha256,
        "protocol": original.PROTOCOL,
        "protocol_freeze_sha256": manifest["protocol_freeze_sha256"],
        "qualification_admitted_rows": manifest["qualification_admitted_rows"],
        "qualification_freeze_sha256": manifest["qualification_freeze_sha256"],
        "qualification_input_rows": manifest["qualification_input_rows"],
        "qualification_rejected_rows": manifest["qualification_rejected_rows"],
        "qualification_source_commit": manifest["qualification_source_commit"],
        "row_count": rows,
        "schema": original.WORKER_SCHEMA,
        "selected_manifest_sha256": manifest["selected_manifest_sha256"],
        "selected_source_commit": manifest["selected_source_commit"],
        "selector_freeze_sha256": manifest["selector_freeze_sha256"],
        "selector_source_commit": manifest["selector_source_commit"],
        "status": "pass",
        "task": dict(task),
        "task_manifest_sha256": task_manifest_sha256,
        "tokenizer_sha256": codec.tokenizer_sha256,
        "uncompressed_bytes": uncompressed_bytes,
    }
    report["report_sha256"] = hashlib.sha256(
        original.canonical_json_bytes(report)
    ).hexdigest()
    return report


def materialize_task_parallel(
    task_manifest: Path,
    selected_root: Path,
    output_root: Path,
    reports_root: Path,
    tokenizer_path: Path,
    materializer_source_root: Path,
    materializer_freeze: Path,
    *,
    task_index: int,
    workers: int,
    execution_source_commit: str,
    execution_code_sha256: str,
    execution_code: Path | None = None,
) -> dict[str, object]:
    """Materialize one main cell in parallel using the frozen implementation."""

    if (
        workers < 1
        or workers > 64
        or _HEX40.fullmatch(execution_source_commit) is None
        or _HEX64.fullmatch(execution_code_sha256) is None
    ):
        raise ParallelMaterializationError("parallel adapter settings differ")
    code_path = (execution_code or Path(__file__)).resolve()
    if _stable_sha256(code_path, "execution adapter") != execution_code_sha256:
        raise ParallelMaterializationError("execution adapter SHA-256 differs")

    manifest, task_manifest_sha = original._load_tasks(task_manifest)
    original._verify_materializer_freeze(
        manifest,
        materializer_source_root,
        materializer_freeze,
    )
    if manifest.get("role") != "main":
        raise ParallelMaterializationError(
            "parallel adapter is restricted to public main cells"
        )
    task = original._task_at(manifest, task_index)
    selected_path = selected_root / original._relative(
        task["input_path"],
        "input path",
    )
    observed_sha256, observed_bytes = original._sha256_file(selected_path)
    if (
        observed_sha256 != task["input_sha256"]
        or observed_bytes != task["input_bytes"]
    ):
        raise ParallelMaterializationError("selected input identity differs")

    codec = TokenNativeSurfaceCodec(tokenizer_path)
    if (
        codec.tokenizer_sha256 != manifest["selected_tokenizer_sha256"]
        or codec.codebook_sha256 != manifest["selected_codebook_sha256"]
    ):
        raise ParallelMaterializationError(
            "selected and materializer tokenizer custody differs"
        )
    with gzip.open(selected_path, "rb") as selected_input:
        payloads = tuple(selected_input)
    if len(payloads) != task["input_rows"]:
        raise ParallelMaterializationError("selected input row count differs")

    output_path = output_root / original._relative(
        task["output_path"],
        "output path",
    )
    report_path = reports_root / original._relative(
        task["report_path"],
        "report path",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(output_path, flags, 0o444)
    rows = 0
    uncompressed_bytes = 0
    core_ids: set[str] = set()
    semantic_hashes: set[str] = set()
    try:
        context = multiprocessing.get_context("spawn")
        with (
            os.fdopen(descriptor, "wb") as raw_output,
            ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_initialize_worker,
                initargs=(str(tokenizer_path),),
            ) as executor,
        ):
            descriptor = -1
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=6,
                fileobj=raw_output,
                mtime=0,
            ) as compressed_output:
                for expected_index, result in enumerate(
                    executor.map(
                        _materialize_one,
                        enumerate(payloads),
                        chunksize=1,
                    )
                ):
                    (
                        index,
                        row,
                        split,
                        stage,
                        core_id,
                        semantic_hash,
                    ) = result
                    if (
                        index != expected_index
                        or split != task["split"]
                        or stage != task["stage"]
                        or core_id in core_ids
                        or semantic_hash in semantic_hashes
                    ):
                        raise ParallelMaterializationError(
                            "parallel materialized row identity differs"
                        )
                    core_ids.add(core_id)
                    semantic_hashes.add(semantic_hash)
                    compressed_output.write(row)
                    rows += 1
                    uncompressed_bytes += len(row)
            raw_output.flush()
            os.fsync(raw_output.fileno())
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        output_path.unlink(missing_ok=True)
        raise
    if rows != task["input_rows"]:
        output_path.unlink(missing_ok=True)
        raise ParallelMaterializationError("materialized task row count differs")

    output_sha256, output_bytes = original._sha256_file(output_path)
    report = _report(
        manifest=manifest,
        task=task,
        task_manifest_sha256=task_manifest_sha,
        codec=codec,
        rows=rows,
        uncompressed_bytes=uncompressed_bytes,
        output_sha256=output_sha256,
        output_bytes=output_bytes,
        execution_source_commit=execution_source_commit,
        execution_code_sha256=execution_code_sha256,
        workers=workers,
    )
    try:
        original._write_no_replace(
            report_path,
            original.canonical_json_bytes(report),
        )
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--selected-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--materializer-source-root", type=Path, required=True)
    parser.add_argument("--materializer-freeze", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--execution-source-commit", required=True)
    parser.add_argument("--execution-code-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = materialize_task_parallel(
        arguments.task_manifest,
        arguments.selected_root,
        arguments.output_root,
        arguments.reports_root,
        arguments.tokenizer,
        arguments.materializer_source_root,
        arguments.materializer_freeze,
        task_index=arguments.task_index,
        workers=arguments.workers,
        execution_source_commit=arguments.execution_source_commit,
        execution_code_sha256=arguments.execution_code_sha256,
    )
    print(
        json.dumps(
            {
                "output_sha256": report["output_sha256"],
                "row_count": report["row_count"],
                "status": report["status"],
                "task_index": arguments.task_index,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
