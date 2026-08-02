#!/usr/bin/env python3
"""Measure exact public operation-role capacity for role-anchored effects."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

from audit_ettr_public_opcode_identifiability import (
    parse_public_transport,
    public_document_indices,
)
from audit_ettr_public_operation_identifiability import resolved_operations
from ettr_il_v2_token_native_surface import TokenNativeSurfaceCodec
from ettr_il_v3_protocol import canonical_json_bytes
from materialize_ettr_il_v3_corpus import _iter_records, _sha256_file


REPORT_SCHEMA = "r12-ettr-public-operation-role-capacity-audit-v2"
EFFECT_CAPACITY_SCHEMA = "r12-ettr-public-operation-state-delta-audit-v4"
_SPLITS = ("train", "development")
EFFECT_SLOTS = 20
MAXIMUM_ROLES = 4
MOTORS_PER_ROLE = 5


class OperationRoleCapacityAuditError(ValueError):
    """A public operation or source receipt differs."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effect-capacity-report", type=Path, required=True)
    parser.add_argument("--effect-capacity-report-sha256", required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def operation_arity(operation: object) -> int:
    """Return operator-plus-argument child count from a resolved public call."""

    if (
        not isinstance(operation, list)
        or len(operation) != 3
        or operation[:2] != ["call", 4]
        or not isinstance(operation[2], list)
        or not operation[2]
    ):
        raise OperationRoleCapacityAuditError(
            "resolved public operation differs"
        )
    return len(operation[2])


def _record_arities(record: object, codec: TokenNativeSurfaceCodec) -> tuple[int, ...]:
    views = tuple(record.source_visible.views)
    if not views or len(views[0].command_sources) != 4:
        raise OperationRoleCapacityAuditError("public command orbit differs")
    values = []
    for source in views[0].command_sources:
        command = parse_public_transport(
            public_document_indices(codec, source),
            codebook_size=len(codec.codebook.token_ids),
        )
        operations = resolved_operations(command)
        if not operations:
            raise OperationRoleCapacityAuditError("public command is empty")
        values.extend(operation_arity(operation) for operation in operations)
    return tuple(values)


def _audit_shard(
    arguments: tuple[Path, Path, str, Path],
) -> tuple[Counter[int], set[str], dict[str, object]]:
    path, data_root, split, tokenizer = arguments
    codec = TokenNativeSurfaceCodec(tokenizer)
    histogram: Counter[int] = Counter()
    core_ids: set[str] = set()
    rows = 0
    digest, size = _sha256_file(path)
    for payload, record in _iter_records(path):
        if record.canonical_bytes() != payload or record.identity.split != split:
            raise OperationRoleCapacityAuditError("semantic-core record differs")
        if record.identity.core_id in core_ids:
            raise OperationRoleCapacityAuditError(
                "duplicate semantic-core identity"
            )
        core_ids.add(record.identity.core_id)
        histogram.update(_record_arities(record, codec))
        rows += 1
    return histogram, core_ids, {
        "bytes": size,
        "path": path.relative_to(data_root).as_posix(),
        "rows": rows,
        "sha256": digest,
    }


def _shards(data_root: Path, split: str) -> tuple[Path, ...]:
    root = data_root / split
    paths = tuple(sorted(root.glob("*.jsonl.gz"))) if root.is_dir() else ()
    if not paths or any(path.is_symlink() or not path.is_file() for path in paths):
        raise OperationRoleCapacityAuditError(f"split shard set differs: {split}")
    return paths


def _audit_split(
    data_root: Path,
    tokenizer: Path,
    split: str,
    workers: int,
) -> dict[str, object]:
    paths = _shards(data_root, split)
    arguments = tuple((path, data_root, split, tokenizer) for path in paths)
    if workers == 1:
        results = tuple(_audit_shard(argument) for argument in arguments)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(paths))) as pool:
            results = tuple(pool.map(_audit_shard, arguments))
    histogram: Counter[int] = Counter()
    core_ids: set[str] = set()
    receipts = []
    for shard_histogram, shard_ids, receipt in results:
        if core_ids.intersection(shard_ids):
            raise OperationRoleCapacityAuditError(
                "duplicate semantic-core identity across shards"
            )
        core_ids.update(shard_ids)
        histogram.update(shard_histogram)
        receipts.append(receipt)
    return {
        "core_rows": len(core_ids),
        "histogram": {
            str(arity): histogram[arity] for arity in sorted(histogram)
        },
        "maximum_operation_arity": max(histogram),
        "operation_instances": sum(histogram.values()),
        "required_effect_roles": max(histogram) + 1,
        "shards": receipts,
    }


def _load_effect_capacity(
    path: Path,
    expected_sha256: str,
    *,
    data_root: Path,
    tokenizer_sha256: str,
) -> dict[str, dict[str, object]]:
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise OperationRoleCapacityAuditError(
            "effect-capacity report receipt differs"
        )
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise OperationRoleCapacityAuditError(
            "effect-capacity report hash differs"
        )
    try:
        report = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperationRoleCapacityAuditError(
            "effect-capacity report cannot be decoded"
        ) from exc
    report_payload = dict(report) if isinstance(report, dict) else {}
    claimed_payload_sha256 = report_payload.pop("report_payload_sha256", None)
    if (
        not isinstance(report, dict)
        or report.get("schema") != EFFECT_CAPACITY_SCHEMA
        or report.get("status") != "pass"
        or Path(str(report.get("data_root"))).resolve() != data_root
        or not isinstance(report.get("tokenizer"), dict)
        or report["tokenizer"].get("sha256") != tokenizer_sha256
        or not isinstance(report.get("effect_set_capacity"), dict)
        or claimed_payload_sha256
        != hashlib.sha256(canonical_json_bytes(report_payload)).hexdigest()
    ):
        raise OperationRoleCapacityAuditError(
            "effect-capacity report contract differs"
        )
    result: dict[str, dict[str, object]] = {}
    for split in _SPLITS:
        value = report["effect_set_capacity"].get(split)
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("maximum"), int)
            or not isinstance(value.get("instances"), int)
            or not isinstance(value.get("histogram"), dict)
        ):
            raise OperationRoleCapacityAuditError(
                "effect-capacity split differs"
            )
        result[split] = value
    return result


def audit(
    data_root: Path,
    tokenizer: Path,
    *,
    effect_capacity_report: Path,
    effect_capacity_report_sha256: str,
    workers: int = 1,
) -> dict[str, object]:
    if workers < 1:
        raise OperationRoleCapacityAuditError("worker count differs")
    data_root = data_root.resolve()
    tokenizer = tokenizer.resolve()
    tokenizer_sha256, tokenizer_bytes = _sha256_file(tokenizer)
    effect_capacity_report = effect_capacity_report.resolve()
    effect_capacity = _load_effect_capacity(
        effect_capacity_report,
        effect_capacity_report_sha256,
        data_root=data_root,
        tokenizer_sha256=tokenizer_sha256,
    )
    splits = {
        split: _audit_split(data_root, tokenizer, split, workers)
        for split in _SPLITS
    }
    minimum_valid_motors = min(
        2 * MOTORS_PER_ROLE,
        MAXIMUM_ROLES * MOTORS_PER_ROLE,
    )
    capacity_pass = all(
        int(splits[split]["required_effect_roles"]) <= MAXIMUM_ROLES
        and int(effect_capacity[split]["maximum"]) <= minimum_valid_motors
        and int(effect_capacity[split]["instances"])
        == int(splits[split]["operation_instances"])
        for split in _SPLITS
    )
    report = {
        "capacity": {
            "effect_slots": EFFECT_SLOTS,
            "maximum_roles": MAXIMUM_ROLES,
            "minimum_valid_motors": minimum_valid_motors,
            "motors_per_role": MOTORS_PER_ROLE,
            "pass": capacity_pass,
        },
        "data_root": str(data_root),
        "input_contract": {
            "answer_read": False,
            "assessor_read": False,
            "first_public_renderer_sufficient_for_invariant_arity": True,
            "operation_root_is_role_zero": True,
            "remaining_roles_are_direct_semantic_children": True,
            "target_read": False,
        },
        "effect_capacity_receipt": {
            "path": str(effect_capacity_report),
            "sha256": effect_capacity_report_sha256,
            "splits": effect_capacity,
        },
        "schema": REPORT_SCHEMA,
        "splits": splits,
        "status": "pass" if capacity_pass else "fail",
        "tokenizer": {
            "bytes": tokenizer_bytes,
            "path": str(tokenizer),
            "sha256": tokenizer_sha256,
        },
    }
    report["report_payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(report)
    ).hexdigest()
    return report


def _write_no_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(descriptor, "wb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    report = audit(
        args.data_root,
        args.tokenizer,
        effect_capacity_report=args.effect_capacity_report,
        effect_capacity_report_sha256=(
            args.effect_capacity_report_sha256
        ),
        workers=args.workers,
    )
    _write_no_replace(args.output, canonical_json_bytes(report))
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
