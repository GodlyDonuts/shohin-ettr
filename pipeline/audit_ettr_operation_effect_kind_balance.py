#!/usr/bin/env python3
"""Audit exact non-NOOP effect-kind balance for cardinality-gated ETTR."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from audit_ettr_program_templates import _corner_from_targets, _packet_from_value
from audit_ettr_public_operation_state_delta import state_delta_value
from ettr_il_v2_materialize import (
    _encode_mutation,
    _independent_replay,
    _project_initial,
)
from ettr_il_v3_protocol import canonical_json_bytes
from materialize_ettr_il_v3_corpus import _iter_records, _sha256_file


REPORT_SCHEMA = "r12-ettr-operation-effect-kind-balance-audit-v2"
CAPACITY_SCHEMA = "r12-ettr-public-operation-state-delta-audit-v4"
_SPLITS = ("train", "development")
_KIND_FAMILY = {
    "allocate": "entity",
    "write": "entity",
    "clear": "entity",
    "replace": "entity",
    "link": "relation",
    "unlink": "relation",
    "root_clear": "root",
    "root_set": "root",
    "commit": "disposition",
    "halt": "disposition",
    "reject": "disposition",
}


class EffectKindBalanceAuditError(ValueError):
    """A corpus receipt or exact effect label differs."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--capacity-report", type=Path, required=True)
    parser.add_argument("--capacity-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def effect_kinds(delta: Mapping[str, object]) -> tuple[str, ...]:
    """Return the exact non-NOOP kinds used by the v12-v14 target contract."""

    nodes = delta.get("nodes")
    added = delta.get("edges_added")
    removed = delta.get("edges_removed")
    status = delta.get("status")
    if (
        not isinstance(nodes, list)
        or not isinstance(added, list)
        or not isinstance(removed, list)
        or not isinstance(status, list)
        or len(status) != 4
    ):
        raise EffectKindBalanceAuditError("state delta differs")
    kinds: list[str] = []
    root_changed = False
    target_has_root = False
    for item in nodes:
        if not isinstance(item, list) or len(item) != 3:
            raise EffectKindBalanceAuditError("state delta node differs")
        _slot, before_value, after_value = item
        if (
            not isinstance(before_value, list)
            or not isinstance(after_value, list)
            or len(before_value) != 4
            or len(after_value) != 4
        ):
            raise EffectKindBalanceAuditError("state delta node value differs")
        before_active = bool(before_value[0])
        after_active = bool(after_value[0])
        if not before_active and after_active:
            kinds.append("allocate")
        elif before_active and not after_active:
            kinds.append("clear")
        elif before_active and after_active and before_value[1] != after_value[1]:
            kinds.append("replace")
        elif before_active and after_active and before_value[2] != after_value[2]:
            kinds.append("write")
        root_changed = root_changed or before_value[3] != after_value[3]
        target_has_root = target_has_root or bool(after_value[3])
    kinds.extend("link" for _ in added)
    kinds.extend("unlink" for _ in removed)
    if root_changed:
        kinds.append("root_set" if target_has_root else "root_clear")
    before_committed, before_halted, after_committed, after_halted = map(bool, status)
    if (before_committed, before_halted) != (after_committed, after_halted):
        if after_committed and not after_halted:
            kinds.append("commit")
        elif after_halted and not after_committed:
            kinds.append("halt")
        elif after_committed and after_halted:
            kinds.append("reject")
    return tuple(kinds)


def _record_counts(
    record: object,
) -> tuple[Counter[str], Counter[int], Counter[tuple[str, int]], int]:
    targets = record.assessor_only.targets
    initial_packets = tuple(
        _packet_from_value(value, f"initial packet {index}")
        for index, value in enumerate(targets.initial_packets)
    )
    terminal_packets = tuple(
        _packet_from_value(value, f"terminal packet {index}")
        for index, value in enumerate(targets.terminal_packets)
    )
    corners = tuple(
        _corner_from_targets(
            terminal_packets[index],
            targets.transaction_traces[index],
            targets.answer_matrix[index],
            f"corner {index}",
        )
        for index in range(4)
    )
    kind_counts: Counter[str] = Counter()
    cardinalities: Counter[int] = Counter()
    per_kind_cardinalities: Counter[tuple[str, int]] = Counter()
    operations = 0
    for world_index in range(2):
        initial, static_ranks = _project_initial(
            initial_packets[world_index], f"world {world_index}"
        )
        for command_index in range(2):
            corner_index = 2 * world_index + command_index
            state = initial
            for rank, trace in enumerate(corners[corner_index].operation_traces):
                steps = tuple(
                    _encode_mutation(
                        mutation,
                        static_ranks,
                        f"corner {corner_index} operation {rank}",
                    )
                    for mutation in trace.mutations
                )
                after = (
                    _independent_replay(
                        state,
                        steps,
                        f"corner {corner_index} operation {rank}",
                    )[0]
                    if steps
                    else state
                )
                kinds = effect_kinds(state_delta_value(state, after))
                kind_counts.update(kinds)
                cardinalities[len(kinds)] += 1
                operation_kinds = Counter(kinds)
                per_kind_cardinalities.update(
                    (name, operation_kinds[name]) for name in _KIND_FAMILY
                )
                operations += 1
                state = after
    return kind_counts, cardinalities, per_kind_cardinalities, operations


def _audit_shard(
    arguments: tuple[Path, Path, str],
) -> tuple[
    Counter[str],
    Counter[int],
    Counter[tuple[str, int]],
    set[str],
    dict[str, object],
]:
    path, data_root, split = arguments
    kind_counts: Counter[str] = Counter()
    cardinalities: Counter[int] = Counter()
    per_kind_cardinalities: Counter[tuple[str, int]] = Counter()
    core_ids: set[str] = set()
    operations = 0
    rows = 0
    digest, size = _sha256_file(path)
    for payload, record in _iter_records(path):
        if record.canonical_bytes() != payload or record.identity.split != split:
            raise EffectKindBalanceAuditError("semantic-core record differs")
        if record.identity.core_id in core_ids:
            raise EffectKindBalanceAuditError("duplicate semantic-core identity")
        core_ids.add(record.identity.core_id)
        (
            record_kinds,
            record_cardinalities,
            record_per_kind,
            record_operations,
        ) = _record_counts(record)
        kind_counts.update(record_kinds)
        cardinalities.update(record_cardinalities)
        per_kind_cardinalities.update(record_per_kind)
        operations += record_operations
        rows += 1
    return (
        kind_counts,
        cardinalities,
        per_kind_cardinalities,
        core_ids,
        {
            "bytes": size,
            "operation_instances": operations,
            "path": path.relative_to(data_root).as_posix(),
            "rows": rows,
            "sha256": digest,
        },
    )


def _audit_split(
    data_root: Path,
    split: str,
    workers: int,
) -> dict[str, object]:
    root = data_root / split
    paths = tuple(sorted(root.glob("*.jsonl.gz"))) if root.is_dir() else ()
    if not paths or any(path.is_symlink() or not path.is_file() for path in paths):
        raise EffectKindBalanceAuditError(f"split shard set differs: {split}")
    arguments = tuple((path, data_root, split) for path in paths)
    if workers == 1:
        results = tuple(_audit_shard(argument) for argument in arguments)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(paths))) as pool:
            results = tuple(pool.map(_audit_shard, arguments))
    kinds: Counter[str] = Counter()
    cardinalities: Counter[int] = Counter()
    per_kind_cardinalities: Counter[tuple[str, int]] = Counter()
    core_ids: set[str] = set()
    receipts = []
    for (
        shard_kinds,
        shard_cardinalities,
        shard_per_kind,
        shard_ids,
        receipt,
    ) in results:
        if core_ids.intersection(shard_ids):
            raise EffectKindBalanceAuditError(
                "duplicate semantic-core identity across shards"
            )
        kinds.update(shard_kinds)
        cardinalities.update(shard_cardinalities)
        per_kind_cardinalities.update(shard_per_kind)
        core_ids.update(shard_ids)
        receipts.append(receipt)
    families = Counter(
        {
            family: sum(
                kinds[name] for name, value in _KIND_FAMILY.items() if value == family
            )
            for family in sorted(set(_KIND_FAMILY.values()))
        }
    )
    present = [value for value in kinds.values() if value]
    per_kind_histograms = {
        name: {
            str(count): per_kind_cardinalities[(name, count)]
            for count in sorted(
                value for kind, value in per_kind_cardinalities if kind == name
            )
        }
        for name in sorted(_KIND_FAMILY)
    }
    return {
        "cardinality_histogram": {
            str(key): cardinalities[key] for key in sorted(cardinalities)
        },
        "core_rows": len(core_ids),
        "family_histogram": dict(sorted(families.items())),
        "kind_histogram": {key: kinds[key] for key in sorted(_KIND_FAMILY)},
        "maximum_to_minimum_present_kind_ratio": max(present) / min(present),
        "maximum_per_kind": {
            name: max(int(count) for count in histogram)
            for name, histogram in per_kind_histograms.items()
        },
        "nonnoop_effects": sum(kinds.values()),
        "operation_instances": sum(cardinalities.values()),
        "per_kind_cardinality_histograms": per_kind_histograms,
        "shards": receipts,
    }


def _load_capacity(path: Path, expected_sha256: str, data_root: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise EffectKindBalanceAuditError("capacity report receipt differs")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise EffectKindBalanceAuditError("capacity report hash differs")
    report = json.loads(payload)
    if (
        not isinstance(report, dict)
        or report.get("schema") != CAPACITY_SCHEMA
        or report.get("status") != "pass"
        or Path(str(report.get("data_root"))).resolve() != data_root
    ):
        raise EffectKindBalanceAuditError("capacity report contract differs")
    return report


def audit(
    data_root: Path,
    *,
    capacity_report: Path,
    capacity_report_sha256: str,
    workers: int = 1,
) -> dict[str, object]:
    if workers < 1:
        raise EffectKindBalanceAuditError("worker count differs")
    data_root = data_root.resolve()
    capacity_report = capacity_report.resolve()
    capacity = _load_capacity(capacity_report, capacity_report_sha256, data_root)
    splits = {split: _audit_split(data_root, split, workers) for split in _SPLITS}
    for split in _SPLITS:
        expected = int(capacity["effect_set_capacity"][split]["instances"])
        if splits[split]["operation_instances"] != expected:
            raise EffectKindBalanceAuditError("operation instance count differs")
    report = {
        "capacity_report": {
            "path": str(capacity_report),
            "sha256": capacity_report_sha256,
        },
        "data_root": str(data_root),
        "input_contract": {
            "answer_read": False,
            "assessor_mutations_used_as_labels_only": True,
            "query_read": False,
            "target_read": False,
        },
        "schema": REPORT_SCHEMA,
        "splits": splits,
        "status": "pass",
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
        capacity_report=args.capacity_report,
        capacity_report_sha256=args.capacity_report_sha256,
        workers=args.workers,
    )
    _write_no_replace(args.output, canonical_json_bytes(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
