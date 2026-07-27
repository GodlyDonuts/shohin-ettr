#!/usr/bin/env python3
"""Qualify raw ETTR-IL-v3 candidates through the real causal receiver.

Candidate generation intentionally overproduces semantic episodes.  This
stage runs every raw candidate through the exact architecture-facing
materializer and preserves only rows that the receiver admits.  Main and
sealed-confirmation cells are written to physically distinct roots.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Sequence

from ettr_il_v2_horn_adapter import HornAdapterError
from ettr_il_v2_materialize import MaterializationError
from ettr_il_v2_resource_adapter import ResourceAdapterError
from ettr_il_v2_semantics import SemanticAdmissionError
from ettr_il_v2_surface import SurfaceError
from ettr_il_v2_surface_adapter import SurfaceAdapterError
from ettr_il_v2_token_native_surface import TokenNativeSurfaceCodec
from ettr_il_v3_local_adapter import LocalAdapterError
from ettr_il_v3_materialize import (
    V3MaterializationError,
    materialize_candidate,
)
from ettr_il_v3_production import ProductionCell, production_cells
from ettr_il_v3_protocol import PROTOCOL, canonical_json_bytes
from ettr_il_v3_reconstruct import ReconstructionError
from ettr_il_v3_rectangles import RectangleError
from freeze_ettr_il_v3_protocol import load_and_verify_freeze
from select_ettr_il_v3 import (
    CONFIRMATION_SPLITS,
    MAIN_SPLITS,
    Candidate,
    load_production_candidates,
    verify_production_report,
)


SCHEMA = "r12-ettr-il-v3-qualified-candidate-cell-v1"
MAIN_ROLE = "main"
CONFIRMATION_ROLE = "sealed_confirmation"
MAX_KEY_BYTES = 32
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ADMISSION_ERRORS = (
    HornAdapterError,
    LocalAdapterError,
    MaterializationError,
    ReconstructionError,
    RectangleError,
    ResourceAdapterError,
    SemanticAdmissionError,
    SurfaceAdapterError,
    SurfaceError,
    V3MaterializationError,
)


class QualificationError(ValueError):
    """Candidate qualification custody or output contract failed."""


def role_for_cell(cell: ProductionCell) -> str:
    if cell.split in MAIN_SPLITS:
        return MAIN_ROLE
    if cell.split in CONFIRMATION_SPLITS:
        return CONFIRMATION_ROLE
    raise QualificationError("candidate cell split differs")


def cells_for_role(role: str) -> tuple[ProductionCell, ...]:
    if role not in {MAIN_ROLE, CONFIRMATION_ROLE}:
        raise QualificationError("qualification role differs")
    return tuple(cell for cell in production_cells() if role_for_cell(cell) == role)


def _hex(value: object, label: str, *, length: int = 64) -> str:
    expression = _HEX40 if length == 40 else _HEX64
    if not isinstance(value, str) or expression.fullmatch(value) is None:
        raise QualificationError(f"{label} differs")
    return value


def _regular_bytes(path: Path, label: str, maximum: int | None = None) -> bytes:
    before = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise QualificationError(f"{label} is not a single-link regular file")
    if maximum is not None and before.st_size > maximum:
        raise QualificationError(f"{label} exceeds its size bound")
    payload = path.read_bytes()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise QualificationError(f"{label} changed during measurement")
    return payload


def _confirmation_key(path: Path | None, role: str) -> bytes | None:
    if role == MAIN_ROLE:
        if path is not None:
            raise QualificationError("main qualification received a sealed key")
        return None
    if path is None:
        raise QualificationError("confirmation qualification requires a sealed key")
    payload = _regular_bytes(path, "confirmation key", MAX_KEY_BYTES)
    if len(payload) != MAX_KEY_BYTES or stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise QualificationError("confirmation key custody differs")
    return payload


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


def _rejection_reason(error: Exception) -> str:
    message = str(error).lower()
    categories = (
        ("prefix independent", "prefix_independent"),
        ("no generic state effect", "no_generic_state_effect"),
        ("causal-neighbor search exhausted", "causal_neighbor_exhausted"),
        ("token width", "token_width"),
        ("tokenize", "tokenization"),
        ("geometry differs", "geometry"),
        ("execution differs", "execution_mismatch"),
        ("identity differs", "identity_mismatch"),
        ("binding differs", "binding_mismatch"),
        ("oracle", "oracle_mismatch"),
        ("packet", "packet_contract"),
        ("rectangle", "rectangle_contract"),
    )
    category = next(
        (code for fragment, code in categories if fragment in message),
        "other",
    )
    return f"{type(error).__name__}:{category}"


def _qualified_payload(candidates: Sequence[Candidate]) -> tuple[bytes, bytes]:
    rows = tuple(canonical_json_bytes(candidate.row) for candidate in candidates)
    uncompressed = b"".join(rows)
    return uncompressed, gzip.compress(uncompressed, compresslevel=6, mtime=0)


def qualify_cell(
    candidate_root: Path,
    output_root: Path,
    tokenizer_path: Path,
    qualifier_source_root: Path,
    qualifier_freeze: Path,
    *,
    matrix_index: int,
    role: str,
    qualifier_source_commit: str,
    confirmation_key_file: Path | None = None,
) -> dict[str, object]:
    """Receiver-qualify one raw production cell and publish no-replace output."""

    source_commit = _hex(
        qualifier_source_commit,
        "qualifier source commit",
        length=40,
    )
    freeze = load_and_verify_freeze(
        qualifier_source_root,
        qualifier_freeze,
        source_commit=source_commit,
    )
    qualifier_freeze_sha256 = _hex(
        freeze.get("freeze_sha256"),
        "qualifier freeze SHA-256",
    )
    cells = production_cells()
    if not 0 <= matrix_index < len(cells):
        raise QualificationError("qualification matrix index differs")
    cell = cells[matrix_index]
    if cell.index != matrix_index or role_for_cell(cell) != role:
        raise QualificationError("qualification cell role differs")
    key = _confirmation_key(confirmation_key_file, role)

    production_report, candidate_source_commit, protocol_freeze_sha256 = (
        verify_production_report(
            candidate_root,
            cell,
            source_commit=None,
            protocol_freeze=None,
        )
    )
    candidates = load_production_candidates(
        candidate_root,
        cell,
        production_report,
    )
    codec = TokenNativeSurfaceCodec(tokenizer_path)
    admitted: list[Candidate] = []
    rejected: Counter[str] = Counter()
    for candidate in candidates:
        try:
            materialize_candidate(
                candidate.row,
                codec,
                confirmation_key=key,
            )
        except _ADMISSION_ERRORS as error:
            rejected[_rejection_reason(error)] += 1
            continue
        admitted.append(candidate)

    uncompressed, compressed = _qualified_payload(admitted)
    shard_name = f"cell-{cell.index}.jsonl.gz"
    shard_path = output_root / "shards" / shard_name
    report_path = output_root / "reports" / f"cell-{cell.index}.json"
    _write_no_replace(shard_path, compressed)
    report: dict[str, object] = {
        "admitted_ordinals_sha256": hashlib.sha256(
            canonical_json_bytes(
                [int(candidate.row["ordinal"]) for candidate in admitted]
            )
        ).hexdigest(),
        "admitted_row_count": len(admitted),
        "candidate_source_commit": candidate_source_commit,
        "cell": cell.to_value(),
        "codebook_sha256": codec.codebook_sha256,
        "input_report_sha256": production_report["report_sha256"],
        "input_row_count": len(candidates),
        "input_shard_sha256": production_report["shard_sha256"],
        "protocol": PROTOCOL,
        "protocol_freeze_sha256": protocol_freeze_sha256,
        "qualified_compressed_bytes": len(compressed),
        "qualified_shard_name": shard_name,
        "qualified_shard_sha256": hashlib.sha256(compressed).hexdigest(),
        "qualified_uncompressed_bytes": len(uncompressed),
        "qualifier_freeze_sha256": qualifier_freeze_sha256,
        "qualifier_source_commit": source_commit,
        "rejected_row_count": sum(rejected.values()),
        "rejection_histogram": [
            [reason, count] for reason, count in sorted(rejected.items())
        ],
        "role": role,
        "schema": SCHEMA,
        "status": "pass",
        "tokenizer_sha256": codec.tokenizer_sha256,
    }
    report["report_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    try:
        _write_no_replace(report_path, canonical_json_bytes(report))
    except BaseException:
        shard_path.unlink(missing_ok=True)
        raise
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--qualifier-source-root", type=Path, required=True)
    parser.add_argument("--qualifier-source-commit", required=True)
    parser.add_argument("--qualifier-freeze", type=Path, required=True)
    parser.add_argument("--matrix-index", type=int, required=True)
    parser.add_argument(
        "--role",
        choices=(MAIN_ROLE, CONFIRMATION_ROLE),
        required=True,
    )
    parser.add_argument("--confirmation-key-file", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = qualify_cell(
        arguments.candidates,
        arguments.output_root,
        arguments.tokenizer,
        arguments.qualifier_source_root,
        arguments.qualifier_freeze,
        matrix_index=arguments.matrix_index,
        role=arguments.role,
        qualifier_source_commit=arguments.qualifier_source_commit,
        confirmation_key_file=arguments.confirmation_key_file,
    )
    print(
        json.dumps(
            {
                "admitted_row_count": report["admitted_row_count"],
                "matrix_index": arguments.matrix_index,
                "rejected_row_count": report["rejected_row_count"],
                "report_sha256": report["report_sha256"],
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
