#!/usr/bin/env python3
"""Parallel CPU materialization and global audit for ETTR-IL-v3.

The selector emits small, independently hash-bound semantic candidate shards.
This module turns those shards into architecture-facing ``SemanticCoreRecord``
shards without loading the complete corpus into memory.  Main and sealed
confirmation roots use distinct task manifests and confirmation requires a
separate 32-byte key file.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Iterator, Mapping, Sequence

from tokenizers import Tokenizer

from ettr_il_v2_token_native_surface import TokenNativeSurfaceCodec
from ettr_il_v3_materialize import materialize_candidate
from ettr_il_v3_protocol import (
    CHARGED_POSITIONS_PER_ROW,
    PROTOCOL,
    QUERY_WIDTH,
    ROWS_PER_CORE,
    VIEWS_PER_CORE,
    WORLD_WIDTH,
    COMMAND_WIDTH,
    canonical_json_bytes,
)
from ettr_il_v3_shards import SemanticCoreRecord
from select_ettr_il_v3 import (
    CONFIRMATION_SPLITS,
    MAIN_SPLITS,
    MANIFEST_SCHEMA as SELECTED_MANIFEST_SCHEMA,
)


TASK_SCHEMA = "r12-ettr-il-v3-materialization-tasks-v1"
WORKER_SCHEMA = "r12-ettr-il-v3-materialization-worker-v1"
AUDIT_SCHEMA = "r12-ettr-il-v3-materialization-audit-v1"
PUBLICATION_SCHEMA = "r12-ettr-il-v3-hf-publication-manifest-v1"
MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_ROW_BYTES = 32 * 1024 * 1024
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class CorpusMaterializationError(ValueError):
    """The corpus cannot be materialized without violating its contract."""


def _strict_load(payload: bytes, label: str) -> dict[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise CorpusMaterializationError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=pairs,
            parse_float=lambda _: (_ for _ in ()).throw(
                CorpusMaterializationError(f"{label} contains a float")
            ),
            parse_constant=lambda _: (_ for _ in ()).throw(
                CorpusMaterializationError(f"{label} contains a non-finite value")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorpusMaterializationError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise CorpusMaterializationError(f"{label} is not canonical JSON")
    return value


def _regular_bytes(path: Path, label: str, maximum: int | None = None) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise CorpusMaterializationError(f"{label} cannot be inspected") from exc
    if path.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise CorpusMaterializationError(f"{label} is not a single-link regular file")
    if maximum is not None and before.st_size > maximum:
        raise CorpusMaterializationError(f"{label} exceeds its size bound")
    payload = path.read_bytes()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(payload) != before.st_size:
        raise CorpusMaterializationError(f"{label} changed during measurement")
    return payload


def _canonical_file(path: Path, label: str) -> dict[str, object]:
    return _strict_load(_regular_bytes(path, label, MAX_METADATA_BYTES), label)


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _hex(value: object, label: str, length: int = 64) -> str:
    expression = _HEX40 if length == 40 else _HEX64
    if not isinstance(value, str) or expression.fullmatch(value) is None:
        raise CorpusMaterializationError(f"{label} differs")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CorpusMaterializationError(f"{label} differs")
    return value


def _relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CorpusMaterializationError(f"{label} differs")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise CorpusMaterializationError(f"{label} is unsafe")
    return value


def _write_no_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _verify_self_hash(
    value: Mapping[str, object],
    *,
    field: str,
    label: str,
) -> str:
    expected = _hex(value.get(field), f"{label} self-hash")
    unhashed = dict(value)
    del unhashed[field]
    if hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest() != expected:
        raise CorpusMaterializationError(f"{label} self-hash differs")
    return expected


def build_task_manifest(
    selected_root: Path,
    output: Path,
    *,
    materializer_source_commit: str,
) -> dict[str, object]:
    """Freeze one selected root into an exact worker inventory."""

    source_commit = _hex(
        materializer_source_commit,
        "materializer source commit",
        length=40,
    )
    selected_path = selected_root / "manifest.json"
    selected_payload = _regular_bytes(
        selected_path, "selected manifest", MAX_METADATA_BYTES
    )
    selected = _strict_load(selected_payload, "selected manifest")
    if selected.get("schema") != SELECTED_MANIFEST_SCHEMA:
        raise CorpusMaterializationError("selected manifest schema differs")
    selected_sha = _verify_self_hash(
        selected, field="manifest_sha256", label="selected manifest"
    )
    role = selected.get("role")
    if role not in {"main", "sealed_confirmation"}:
        raise CorpusMaterializationError("selected role differs")
    allowed = MAIN_SPLITS if role == "main" else CONFIRMATION_SPLITS
    shards = selected.get("shards")
    if not isinstance(shards, list) or not shards:
        raise CorpusMaterializationError("selected shard inventory differs")
    tasks: list[dict[str, object]] = []
    observed_paths: set[str] = set()
    total_rows = 0
    for index, raw in enumerate(shards):
        if not isinstance(raw, dict) or set(raw) != {
            "bytes",
            "family",
            "path",
            "rows",
            "sha256",
            "split",
            "stage",
        }:
            raise CorpusMaterializationError("selected shard descriptor differs")
        split = raw["split"]
        family = raw["family"]
        stage = raw["stage"]
        if (
            split not in allowed
            or not isinstance(family, str)
            or not isinstance(stage, str)
        ):
            raise CorpusMaterializationError("selected task identity differs")
        input_path = _relative(raw["path"], "selected input path")
        if input_path in observed_paths:
            raise CorpusMaterializationError("selected input path repeats")
        observed_paths.add(input_path)
        rows = _integer(raw["rows"], "selected rows", 1)
        task = {
            "family": family,
            "index": index,
            "input_bytes": _integer(raw["bytes"], "selected bytes", 1),
            "input_path": input_path,
            "input_rows": rows,
            "input_sha256": _hex(raw["sha256"], "selected shard SHA-256"),
            "output_path": f"{split}/{family}-{stage}.jsonl.gz",
            "report_path": f"task-{index:05d}.json",
            "split": split,
            "stage": stage,
        }
        tasks.append(task)
        total_rows += rows
    if total_rows != selected.get("total_rows"):
        raise CorpusMaterializationError("selected total rows differ")
    manifest: dict[str, object] = {
        "materializer_source_commit": source_commit,
        "protocol": PROTOCOL,
        "protocol_freeze_sha256": _hex(
            selected.get("protocol_freeze_sha256"), "protocol freeze"
        ),
        "role": role,
        "schema": TASK_SCHEMA,
        "selected_manifest_sha256": selected_sha,
        "selected_source_commit": _hex(
            selected.get("source_commit"), "selected source commit", length=40
        ),
        "task_count": len(tasks),
        "tasks": tasks,
        "total_rows": total_rows,
    }
    manifest["task_manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    _write_no_replace(output, canonical_json_bytes(manifest))
    return manifest


def _load_tasks(path: Path) -> tuple[dict[str, object], str]:
    manifest = _canonical_file(path, "task manifest")
    if manifest.get("schema") != TASK_SCHEMA or manifest.get("protocol") != PROTOCOL:
        raise CorpusMaterializationError("task manifest contract differs")
    digest = _verify_self_hash(
        manifest, field="task_manifest_sha256", label="task manifest"
    )
    tasks = manifest.get("tasks")
    if (
        not isinstance(tasks, list)
        or len(tasks) != manifest.get("task_count")
        or any(not isinstance(task, dict) for task in tasks)
        or sum(
            _integer(task.get("input_rows"), "task rows", 1)
            for task in tasks
            if isinstance(task, dict)
        )
        != manifest.get("total_rows")
    ):
        raise CorpusMaterializationError("task manifest inventory differs")
    return manifest, digest


def _confirmation_key(path: Path | None, role: object) -> bytes | None:
    if role == "main":
        if path is not None:
            raise CorpusMaterializationError("main materializer received a sealed key")
        return None
    if path is None:
        raise CorpusMaterializationError("confirmation key file is required")
    payload = _regular_bytes(path, "confirmation key", 32)
    mode = stat.S_IMODE(path.stat().st_mode)
    if len(payload) != 32 or mode & 0o077:
        raise CorpusMaterializationError("confirmation key custody differs")
    return payload


def _task_at(manifest: Mapping[str, object], index: int) -> dict[str, object]:
    tasks = manifest["tasks"]
    if not isinstance(tasks, list) or not 0 <= index < len(tasks):
        raise CorpusMaterializationError("task index differs")
    task = tasks[index]
    if not isinstance(task, dict) or task.get("index") != index:
        raise CorpusMaterializationError("task identity differs")
    return task


def materialize_task(
    task_manifest: Path,
    selected_root: Path,
    output_root: Path,
    reports_root: Path,
    tokenizer_path: Path,
    *,
    task_index: int,
    confirmation_key_file: Path | None = None,
) -> dict[str, object]:
    """Materialize one exact selected shard using bounded streaming I/O."""

    manifest, task_manifest_sha = _load_tasks(task_manifest)
    task = _task_at(manifest, task_index)
    selected_path = selected_root / _relative(task["input_path"], "input path")
    observed_input_sha, observed_input_bytes = _sha256_file(selected_path)
    if (
        observed_input_sha != task["input_sha256"]
        or observed_input_bytes != task["input_bytes"]
    ):
        raise CorpusMaterializationError("selected input identity differs")
    key = _confirmation_key(confirmation_key_file, manifest["role"])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    codec = TokenNativeSurfaceCodec(tokenizer)
    output_path = output_root / _relative(task["output_path"], "output path")
    report_path = reports_root / _relative(task["report_path"], "report path")
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
        with os.fdopen(descriptor, "wb") as raw_output:
            descriptor = -1
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=6,
                fileobj=raw_output,
                mtime=0,
            ) as compressed_output:
                with gzip.open(selected_path, "rb") as selected_input:
                    for payload in selected_input:
                        if len(payload) > MAX_ROW_BYTES:
                            raise CorpusMaterializationError(
                                "selected candidate row exceeds size bound"
                            )
                        candidate = _strict_load(payload, "selected candidate row")
                        record = materialize_candidate(
                            candidate,
                            tokenizer,
                            confirmation_key=key,
                        )
                        if (
                            record.identity.split != task["split"]
                            or record.identity.curriculum_stage != task["stage"]
                        ):
                            raise CorpusMaterializationError(
                                "materialized record task identity differs"
                            )
                        semantic_hash = record.assessor_only.audit.semantic_hash
                        if (
                            record.identity.core_id in core_ids
                            or semantic_hash in semantic_hashes
                        ):
                            raise CorpusMaterializationError(
                                "materialized task contains duplicate semantics"
                            )
                        core_ids.add(record.identity.core_id)
                        semantic_hashes.add(semantic_hash)
                        row = record.canonical_bytes()
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
        raise CorpusMaterializationError("materialized task row count differs")
    output_sha, output_bytes = _sha256_file(output_path)
    report: dict[str, object] = {
        "charged_positions": rows * ROWS_PER_CORE * CHARGED_POSITIONS_PER_ROW,
        "codebook_sha256": codec.codebook_sha256,
        "expanded_rows": rows * ROWS_PER_CORE,
        "materializer_source_commit": manifest["materializer_source_commit"],
        "output_bytes": output_bytes,
        "output_path": task["output_path"],
        "output_sha256": output_sha,
        "protocol": PROTOCOL,
        "protocol_freeze_sha256": manifest["protocol_freeze_sha256"],
        "row_count": rows,
        "schema": WORKER_SCHEMA,
        "selected_manifest_sha256": manifest["selected_manifest_sha256"],
        "selected_source_commit": manifest["selected_source_commit"],
        "status": "pass",
        "task": task,
        "task_manifest_sha256": task_manifest_sha,
        "tokenizer_sha256": codec.tokenizer_sha256,
        "uncompressed_bytes": uncompressed_bytes,
    }
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    try:
        _write_no_replace(report_path, canonical_json_bytes(report))
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return report


def _iter_records(path: Path) -> Iterator[tuple[bytes, SemanticCoreRecord]]:
    try:
        with gzip.open(path, "rb") as handle:
            for payload in handle:
                if len(payload) > MAX_ROW_BYTES:
                    raise CorpusMaterializationError(
                        "materialized row exceeds size bound"
                    )
                yield payload, SemanticCoreRecord.from_canonical_bytes(payload)
    except (OSError, EOFError) as exc:
        raise CorpusMaterializationError("materialized shard is corrupt") from exc


def _audit_record(record: SemanticCoreRecord, tokenizer: Tokenizer) -> None:
    record.validate()
    views = record.source_visible.views
    if len(views) != VIEWS_PER_CORE or {view.renderer for view in views} != set(
        range(VIEWS_PER_CORE)
    ):
        raise CorpusMaterializationError("renderer inventory differs")
    for view in views:
        if (
            len(view.world_sources) != 4
            or len(view.command_sources) != 4
            or len(view.query_sources) != 4
        ):
            raise CorpusMaterializationError("source rectangle geometry differs")
        for source in view.world_sources:
            if (
                len(tokenizer.encode(source, add_special_tokens=False).ids)
                != WORLD_WIDTH
            ):
                raise CorpusMaterializationError("WORLD token width differs")
        for source in view.command_sources:
            if (
                len(tokenizer.encode(source, add_special_tokens=False).ids)
                != COMMAND_WIDTH
            ):
                raise CorpusMaterializationError("COMMAND token width differs")
        for source in view.query_sources:
            for answer in ("0\n", "1\n"):
                if (
                    len(
                        tokenizer.encode(
                            source + answer,
                            add_special_tokens=False,
                        ).ids
                    )
                    != QUERY_WIDTH
                ):
                    raise CorpusMaterializationError("QUERY token width differs")
    oracle = record.assessor_only.oracle
    if oracle.primary.to_value() != oracle.replay.to_value():
        raise CorpusMaterializationError("primary/replay oracle disagreement")
    targets = record.assessor_only.targets
    for index, observation in enumerate(oracle.primary.terminal_observations):
        if not isinstance(observation, Mapping):
            raise CorpusMaterializationError("terminal observation differs")
        if (
            observation.get("answers") != list(targets.answer_matrix[index])
            or observation.get("terminal_packet") != targets.terminal_packets[index]
        ):
            raise CorpusMaterializationError("oracle/target disagreement")


def audit_materialization(
    task_manifest: Path,
    output_root: Path,
    reports_root: Path,
    tokenizer_path: Path,
    audit_report: Path,
) -> dict[str, object]:
    """Fully reload every worker output and emit one global custody receipt."""

    manifest, task_manifest_sha = _load_tasks(task_manifest)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    codec = TokenNativeSurfaceCodec(tokenizer)
    tasks = manifest["tasks"]
    assert isinstance(tasks, list)
    expected_reports = {
        _relative(task["report_path"], "report path")
        for task in tasks
        if isinstance(task, dict)
    }
    observed_reports = {path.name for path in reports_root.iterdir() if path.is_file()}
    if observed_reports != expected_reports:
        raise CorpusMaterializationError("worker report inventory differs")
    core_ids: set[str] = set()
    semantic_hashes: set[str] = set()
    split_counts: dict[str, int] = {}
    descriptors: list[dict[str, object]] = []
    total_rows = 0
    total_expanded = 0
    total_positions = 0
    for task in tasks:
        if not isinstance(task, dict):
            raise CorpusMaterializationError("task descriptor differs")
        report_path = reports_root / _relative(task["report_path"], "report path")
        report = _canonical_file(report_path, "worker report")
        report_sha = _verify_self_hash(
            report, field="report_sha256", label="worker report"
        )
        if (
            report.get("schema") != WORKER_SCHEMA
            or report.get("status") != "pass"
            or report.get("protocol") != PROTOCOL
            or report.get("task") != task
            or report.get("task_manifest_sha256") != task_manifest_sha
            or report.get("tokenizer_sha256") != codec.tokenizer_sha256
            or report.get("codebook_sha256") != codec.codebook_sha256
        ):
            raise CorpusMaterializationError("worker report contract differs")
        output_path = output_root / _relative(task["output_path"], "output path")
        output_sha, output_bytes = _sha256_file(output_path)
        if output_sha != report.get("output_sha256") or output_bytes != report.get(
            "output_bytes"
        ):
            raise CorpusMaterializationError("worker output identity differs")
        rows = 0
        uncompressed = 0
        for payload, record in _iter_records(output_path):
            _audit_record(record, tokenizer)
            if (
                record.identity.split != task["split"]
                or record.identity.curriculum_stage != task["stage"]
            ):
                raise CorpusMaterializationError("record split/stage differs")
            core_id = record.identity.core_id
            semantic_hash = record.assessor_only.audit.semantic_hash
            if core_id in core_ids or semantic_hash in semantic_hashes:
                raise CorpusMaterializationError(
                    "global core or semantic identity repeats"
                )
            core_ids.add(core_id)
            semantic_hashes.add(semantic_hash)
            rows += 1
            uncompressed += len(payload)
        if (
            rows != report.get("row_count")
            or rows != task["input_rows"]
            or uncompressed != report.get("uncompressed_bytes")
        ):
            raise CorpusMaterializationError("worker output counts differ")
        split = str(task["split"])
        split_counts[split] = split_counts.get(split, 0) + rows
        total_rows += rows
        total_expanded += rows * ROWS_PER_CORE
        total_positions += rows * ROWS_PER_CORE * CHARGED_POSITIONS_PER_ROW
        descriptors.append(
            {
                "bytes": output_bytes,
                "path": task["output_path"],
                "report_sha256": report_sha,
                "rows": rows,
                "sha256": output_sha,
                "split": split,
            }
        )
    if total_rows != manifest["total_rows"]:
        raise CorpusMaterializationError("global materialized cardinality differs")
    audit: dict[str, object] = {
        "charged_positions": total_positions,
        "codebook_sha256": codec.codebook_sha256,
        "core_rows": total_rows,
        "expanded_rows": total_expanded,
        "materializer_source_commit": manifest["materializer_source_commit"],
        "protocol": PROTOCOL,
        "protocol_freeze_sha256": manifest["protocol_freeze_sha256"],
        "role": manifest["role"],
        "schema": AUDIT_SCHEMA,
        "selected_manifest_sha256": manifest["selected_manifest_sha256"],
        "shards": descriptors,
        "split_counts": split_counts,
        "status": "pass",
        "task_manifest_sha256": task_manifest_sha,
        "tokenizer_sha256": codec.tokenizer_sha256,
        "unique_core_ids": len(core_ids),
        "unique_semantic_hashes": len(semantic_hashes),
    }
    audit["audit_sha256"] = hashlib.sha256(canonical_json_bytes(audit)).hexdigest()
    _write_no_replace(audit_report, canonical_json_bytes(audit))
    return audit


def prepare_publication(
    audit_report: Path,
    dataset_root: Path,
    dataset_card: Path,
    publication_manifest: Path,
) -> dict[str, object]:
    """Create the exact private-main Hugging Face publication inventory."""

    audit = _canonical_file(audit_report, "materialization audit")
    if (
        audit.get("schema") != AUDIT_SCHEMA
        or audit.get("status") != "pass"
        or audit.get("role") != "main"
    ):
        raise CorpusMaterializationError("only an admitted main audit may publish")
    _verify_self_hash(audit, field="audit_sha256", label="materialization audit")
    card_payload = _regular_bytes(dataset_card, "dataset card", MAX_METADATA_BYTES)
    card_target = dataset_root / "README.md"
    _write_no_replace(card_target, card_payload)
    shards = audit.get("shards")
    if not isinstance(shards, list):
        raise CorpusMaterializationError("audit shard inventory differs")
    publication_shards: list[dict[str, object]] = []
    for descriptor in shards:
        if not isinstance(descriptor, dict):
            raise CorpusMaterializationError("audit shard descriptor differs")
        split = descriptor.get("split")
        if split not in MAIN_SPLITS:
            raise CorpusMaterializationError("confirmation payload in main audit")
        path = "shards/" + _relative(descriptor.get("path"), "audit shard path")
        local = dataset_root / path
        digest, size = _sha256_file(local)
        if digest != descriptor.get("sha256") or size != descriptor.get("bytes"):
            raise CorpusMaterializationError("publication shard identity differs")
        publication_shards.append(
            {
                "path": path,
                "sha256": digest,
                "size_bytes": size,
                "split": split,
            }
        )
    manifest: dict[str, object] = {
        "card": {
            "path": "README.md",
            "sha256": hashlib.sha256(card_payload).hexdigest(),
            "size_bytes": len(card_payload),
        },
        "dataset_protocol": PROTOCOL,
        "schema": PUBLICATION_SCHEMA,
        "shards": publication_shards,
    }
    _write_no_replace(publication_manifest, canonical_json_bytes(manifest))
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    tasks = subparsers.add_parser("tasks")
    tasks.add_argument("--selected-root", type=Path, required=True)
    tasks.add_argument("--output", type=Path, required=True)
    tasks.add_argument("--materializer-source-commit", required=True)
    worker = subparsers.add_parser("worker")
    worker.add_argument("--task-manifest", type=Path, required=True)
    worker.add_argument("--selected-root", type=Path, required=True)
    worker.add_argument("--output-root", type=Path, required=True)
    worker.add_argument("--reports-root", type=Path, required=True)
    worker.add_argument("--tokenizer", type=Path, required=True)
    worker.add_argument("--task-index", type=int, required=True)
    worker.add_argument("--confirmation-key-file", type=Path)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--task-manifest", type=Path, required=True)
    audit.add_argument("--output-root", type=Path, required=True)
    audit.add_argument("--reports-root", type=Path, required=True)
    audit.add_argument("--tokenizer", type=Path, required=True)
    audit.add_argument("--audit-report", type=Path, required=True)
    prepare = subparsers.add_parser("prepare-publication")
    prepare.add_argument("--audit-report", type=Path, required=True)
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--dataset-card", type=Path, required=True)
    prepare.add_argument("--publication-manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "tasks":
        result = build_task_manifest(
            arguments.selected_root,
            arguments.output,
            materializer_source_commit=arguments.materializer_source_commit,
        )
    elif arguments.command == "worker":
        result = materialize_task(
            arguments.task_manifest,
            arguments.selected_root,
            arguments.output_root,
            arguments.reports_root,
            arguments.tokenizer,
            task_index=arguments.task_index,
            confirmation_key_file=arguments.confirmation_key_file,
        )
    elif arguments.command == "audit":
        result = audit_materialization(
            arguments.task_manifest,
            arguments.output_root,
            arguments.reports_root,
            arguments.tokenizer,
            arguments.audit_report,
        )
    else:
        result = prepare_publication(
            arguments.audit_report,
            arguments.dataset_root,
            arguments.dataset_card,
            arguments.publication_manifest,
        )
    print(
        json.dumps(
            {
                "command": arguments.command,
                "status": result.get("status", "pass"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
