#!/usr/bin/env python3
"""Audit public identifiability of operation-level semantic state changes.

The exact operation-trace audit retains assessor cursor and mutation ordering.
Those details are useful for replay but are not necessarily the semantic object
that a public operation identifies.  This audit replays each operation and
labels two stricter quotients instead:

* the order-independent state difference caused by that operation; and
* the cumulative runtime state after that operation.

Only public WORLD and COMMAND syntax form input signatures.  Assessor traces
and initial packets are used offline to construct labels and never become
model inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from audit_ettr_program_templates import _corner_from_targets, _packet_from_value
from audit_ettr_public_opcode_identifiability import (
    _conditional_summary,
    _source_orbits,
    _train_to_development,
    parse_public_transport,
    public_document_indices,
)
from audit_ettr_public_operation_identifiability import resolved_operations
from ettr_il_v2_materialize import (
    _encode_mutation,
    _independent_replay,
    _project_initial,
)
from ettr_il_v2_token_native_surface import TokenNativeSurfaceCodec
from ettr_il_v3_protocol import canonical_json_bytes
from materialize_ettr_il_v3_corpus import _iter_records, _sha256_file


REPORT_SCHEMA = "r12-ettr-public-operation-state-delta-audit-v2"
_SPLITS = ("train", "development")
_TARGETS = (
    "operation_delta",
    "delta_shape",
    "delta_addresses",
    "delta_payloads",
    "cumulative_runtime_state",
)
_WORLD_MODES = ("alpha_exact", "alpha_operator", "topology")
_CONTEXTS = ("operation", "prefix", "full_command_rank")
_MODES = (
    "resolved_operation",
    *tuple(
        f"world_{world_mode}_{context}"
        for world_mode in _WORLD_MODES
        for context in _CONTEXTS
    ),
)
_RUNTIME_STOP = 48


class OperationStateDeltaAuditError(ValueError):
    """The public source or semantic state quotient differs."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _node_value(state: object, slot: int) -> list[object]:
    return [
        bool(state.active[slot]),
        int(state.type_index[slot]),
        int(state.value_code[slot]),
        bool(state.root[slot]),
    ]


def state_delta_value(before: object, after: object) -> dict[str, object]:
    """Return an ordered-replay-independent semantic state difference."""

    nodes = [
        [slot, _node_value(before, slot), _node_value(after, slot)]
        for slot in range(_RUNTIME_STOP)
        if _node_value(before, slot) != _node_value(after, slot)
    ]
    before_edges = {
        tuple(int(value) for value in edge)
        for edge in before.relations
        if int(edge[1]) < _RUNTIME_STOP and int(edge[2]) < _RUNTIME_STOP
    }
    after_edges = {
        tuple(int(value) for value in edge)
        for edge in after.relations
        if int(edge[1]) < _RUNTIME_STOP and int(edge[2]) < _RUNTIME_STOP
    }
    return {
        "edges_added": sorted(after_edges - before_edges),
        "edges_removed": sorted(before_edges - after_edges),
        "nodes": nodes,
        "status": [
            bool(before.committed),
            bool(before.halted),
            bool(after.committed),
            bool(after.halted),
        ],
    }


def runtime_state_value(state: object) -> dict[str, object]:
    """Return the exact query-independent state through the runtime slots."""

    return {
        "edges": sorted(
            tuple(int(value) for value in edge)
            for edge in state.relations
            if int(edge[1]) < _RUNTIME_STOP and int(edge[2]) < _RUNTIME_STOP
        ),
        "nodes": [
            [slot, *_node_value(state, slot)]
            for slot in range(_RUNTIME_STOP)
            if bool(state.active[slot])
        ],
        "status": [bool(state.committed), bool(state.halted)],
    }


def state_delta_factor_values(
    delta: Mapping[str, object],
) -> dict[str, object]:
    """Factor a semantic delta into action shape, addresses, and payloads."""

    nodes = list(delta["nodes"])
    added = list(delta["edges_added"])
    removed = list(delta["edges_removed"])
    status = list(delta["status"])
    shape_nodes = []
    address_nodes = []
    payload_nodes = []
    for item in nodes:
        slot, before, after = item
        before = list(before)
        after = list(after)
        shape_nodes.append(
            [
                before[0] != after[0],
                before[1] != after[1],
                before[2] != after[2],
                before[3] != after[3],
            ]
        )
        address_nodes.append(int(slot))
        payload_nodes.append([before, after])
    return {
        "delta_shape": {
            "edge_additions": len(added),
            "edge_removals": len(removed),
            "node_field_changes": sorted(shape_nodes),
            "status_changes": [status[0] != status[2], status[1] != status[3]],
        },
        "delta_addresses": {
            "edges_added": sorted(added),
            "edges_removed": sorted(removed),
            "nodes": sorted(address_nodes),
        },
        "delta_payloads": {
            "edge_relations_added": sorted(int(edge[0]) for edge in added),
            "edge_relations_removed": sorted(int(edge[0]) for edge in removed),
            "nodes": sorted(payload_nodes),
            "status_after": status[2:],
        },
    }


def _keys(
    operation: object,
    operations: tuple[object, ...],
    rank: int,
    world: Mapping[str, str],
) -> dict[str, str]:
    operation_digest = _digest(operation)
    result = {"resolved_operation": operation_digest}
    for world_mode in _WORLD_MODES:
        world_digest = world[world_mode]
        result[f"world_{world_mode}_operation"] = _digest(
            {"operation": operation_digest, "world": world_digest}
        )
        result[f"world_{world_mode}_prefix"] = _digest(
            {"prefix": operations[: rank + 1], "world": world_digest}
        )
        result[f"world_{world_mode}_full_command_rank"] = _digest(
            {
                "command": operations,
                "rank": rank,
                "world": world_digest,
            }
        )
    if set(result) != set(_MODES):
        raise OperationStateDeltaAuditError("operation context modes differ")
    return result


def _record_labels(
    record: object,
    codec: TokenNativeSurfaceCodec,
) -> Iterable[tuple[str, str, str, str]]:
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
    worlds, _commands = _source_orbits(record, codec)
    views = tuple(record.source_visible.views)
    for world_index in range(2):
        initial, static_ranks = _project_initial(
            initial_packets[world_index], f"world {world_index}"
        )
        for command_index in range(2):
            corner_index = 2 * world_index + command_index
            source = views[0].command_sources[corner_index]
            command_tree = parse_public_transport(
                public_document_indices(codec, source),
                codebook_size=len(codec.codebook.token_ids),
            )
            operations = resolved_operations(command_tree)
            traces = tuple(corners[corner_index].operation_traces)
            if len(operations) != len(traces):
                raise OperationStateDeltaAuditError(
                    "public and assessor operation depths differ"
                )
            state = initial
            for rank, (operation, trace) in enumerate(
                zip(operations, traces, strict=True)
            ):
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
                delta = state_delta_value(state, after)
                target_values = {
                    "operation_delta": delta,
                    **state_delta_factor_values(delta),
                    "cumulative_runtime_state": runtime_state_value(after),
                }
                keys = _keys(operation, operations, rank, worlds[corner_index])
                for target, target_value in target_values.items():
                    label = _digest(target_value)
                    for mode, key in keys.items():
                        yield target, mode, key, label
                state = after


def _new_counts() -> dict[str, dict[str, dict[str, Counter[str]]]]:
    return {
        target: {mode: {} for mode in _MODES}
        for target in _TARGETS
    }


def _audit_shard(
    arguments: tuple[Path, Path, str, Path],
) -> tuple[
    dict[str, dict[str, dict[str, Counter[str]]]],
    set[str],
    dict[str, object],
]:
    path, data_root, split, tokenizer = arguments
    codec = TokenNativeSurfaceCodec(tokenizer)
    counts = _new_counts()
    core_ids: set[str] = set()
    rows = 0
    operation_instances = 0
    digest, size = _sha256_file(path)
    for payload, record in _iter_records(path):
        if record.canonical_bytes() != payload or record.identity.split != split:
            raise OperationStateDeltaAuditError("semantic-core record differs")
        if record.identity.core_id in core_ids:
            raise OperationStateDeltaAuditError("duplicate semantic-core identity")
        core_ids.add(record.identity.core_id)
        rows += 1
        labels = tuple(_record_labels(record, codec))
        operation_instances += len(labels) // (len(_TARGETS) * len(_MODES))
        for target, mode, key, label in labels:
            counts[target][mode].setdefault(key, Counter())[label] += 1
    return counts, core_ids, {
        "bytes": size,
        "operation_instances": operation_instances,
        "path": path.relative_to(data_root).as_posix(),
        "rows": rows,
        "sha256": digest,
    }


def _shards(data_root: Path, split: str) -> tuple[Path, ...]:
    root = data_root / split
    paths = tuple(sorted(root.glob("*.jsonl.gz"))) if root.is_dir() else ()
    if not paths or any(path.is_symlink() or not path.is_file() for path in paths):
        raise OperationStateDeltaAuditError(f"split shard set differs: {split}")
    return paths


def _audit_split(
    data_root: Path,
    split: str,
    tokenizer: Path,
    workers: int,
) -> dict[str, object]:
    paths = _shards(data_root, split)
    arguments = tuple((path, data_root, split, tokenizer) for path in paths)
    if workers == 1:
        results = tuple(_audit_shard(argument) for argument in arguments)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(paths))) as pool:
            results = tuple(pool.map(_audit_shard, arguments))
    counts = _new_counts()
    core_ids: set[str] = set()
    receipts = []
    for source, shard_ids, receipt in results:
        if core_ids.intersection(shard_ids):
            raise OperationStateDeltaAuditError(
                "duplicate semantic-core identity across shards"
            )
        core_ids.update(shard_ids)
        for target in _TARGETS:
            for mode in _MODES:
                for key, labels in source[target][mode].items():
                    counts[target][mode].setdefault(key, Counter()).update(labels)
        receipts.append(receipt)
    return {"core_rows": len(core_ids), "shards": receipts, "_counts": counts}


def audit(
    data_root: Path,
    tokenizer: Path,
    *,
    workers: int = 1,
) -> dict[str, object]:
    if workers < 1:
        raise OperationStateDeltaAuditError("worker count differs")
    data_root = data_root.resolve()
    tokenizer = tokenizer.resolve()
    tokenizer_sha256, tokenizer_bytes = _sha256_file(tokenizer)
    splits = {
        split: _audit_split(data_root, split, tokenizer, workers)
        for split in _SPLITS
    }
    train = splits["train"].pop("_counts")
    development = splits["development"].pop("_counts")
    analyses = {
        target: {
            mode: {
                "development_oracle": _conditional_summary(
                    development[target][mode]
                ),
                "train_fit": _conditional_summary(train[target][mode]),
                "train_to_development": _train_to_development(
                    train[target][mode], development[target][mode]
                ),
            }
            for mode in _MODES
        }
        for target in _TARGETS
    }
    report = {
        "analyses": analyses,
        "data_root": str(data_root),
        "input_contract": {
            "answer_read": False,
            "assessor_initial_packet_used_as_label_constructor_only": True,
            "assessor_operation_trace_used_as_label_only": True,
            "delta_factors_scored_separately": [
                "shape",
                "addresses",
                "payloads",
            ],
            "mutation_order_and_cursor_removed_from_delta_target": True,
            "query_read": False,
            "terminal_packet_read": False,
            "transaction_program_read": False,
            "world_and_command_public_sources_only": True,
        },
        "schema": REPORT_SCHEMA,
        "splits": splits,
        "status": "pass",
        "tokenizer": {
            "bytes": tokenizer_bytes,
            "path": str(tokenizer),
            "sha256": tokenizer_sha256,
        },
    }
    report["report_payload_sha256"] = _digest(report)
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
        workers=args.workers,
    )
    _write_no_replace(args.output, canonical_json_bytes(report))
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
