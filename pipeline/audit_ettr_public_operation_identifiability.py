#!/usr/bin/env python3
"""Audit operation-level ETTR supervision from permitted public syntax.

The one-shot terminal editor can fail because a complete episode contains up
to six ordered public operations.  This audit asks whether each operation's
assessor-only mutation target is identifiable after resolving the exact public
declarations and conditioning on progressively richer public context.  QUERY,
answers, terminal packets, and low-level transaction programs are never input
features; operation traces are labels only.
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

from audit_ettr_public_opcode_identifiability import (
    PublicNode,
    _conditional_summary,
    _source_orbits,
    _train_to_development,
)
from ettr_il_v2_token_native_surface import TokenNativeSurfaceCodec
from ettr_il_v3_protocol import canonical_json_bytes
from materialize_ettr_il_v3_corpus import _iter_records, _sha256_file


REPORT_SCHEMA = "r12-ettr-public-operation-identifiability-audit-v1"
_SPLITS = ("train", "development")
_MODES = (
    "resolved_operation",
    "world_resolved_operation",
    "world_prefix",
    "world_full_command_rank",
)


class PublicOperationIdentifiabilityError(ValueError):
    """A public operation or audit contract differs."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _call(node: PublicNode, head: int) -> bool:
    return len(node) == 3 and node[0] == "call" and int(node[1]) == head


def _command_parts(root: PublicNode) -> tuple[dict[int, PublicNode], tuple[PublicNode, ...]]:
    if _call(root, 15):
        children = tuple(root[2])
        if len(children) != 2:
            raise PublicOperationIdentifiabilityError("COMMAND wrapper differs")
        root = children[1]
    if _call(root, 13):
        operations = tuple(root[2])
        if not 1 <= len(operations) <= 6 or any(
            not _call(item, 4) for item in operations
        ):
            raise PublicOperationIdentifiabilityError(
                "direct COMMAND operation list differs"
            )
        return {}, operations
    if not _call(root, 14):
        raise PublicOperationIdentifiabilityError("COMMAND semantic root differs")
    children = tuple(root[2])
    if (
        len(children) != 3
        or children[0][0] != "integer"
        or int(children[0][1]) != 2
        or not _call(children[1], 1)
        or not _call(children[2], 13)
    ):
        raise PublicOperationIdentifiabilityError("COMMAND public geometry differs")
    declarations: dict[int, PublicNode] = {}
    for declaration in tuple(children[1][2]):
        if not _call(declaration, 3):
            raise PublicOperationIdentifiabilityError("COMMAND declaration differs")
        fields = tuple(declaration[2])
        if (
            len(fields) != 3
            or fields[0][0] != "symbol"
            or fields[1][0] != "integer"
        ):
            raise PublicOperationIdentifiabilityError(
                "COMMAND declaration fields differ"
            )
        symbol_code = int(fields[0][1])
        if symbol_code in declarations:
            raise PublicOperationIdentifiabilityError(
                "COMMAND identifier has multiple declarations"
            )
        declarations[symbol_code] = declaration
    operations = tuple(children[2][2])
    if not 1 <= len(operations) <= 6 or any(not _call(item, 4) for item in operations):
        raise PublicOperationIdentifiabilityError("COMMAND operation list differs")
    return declarations, operations


def _resolved_node(
    node: PublicNode,
    declarations: Mapping[int, PublicNode],
    *,
    resolving: frozenset[int] = frozenset(),
) -> object:
    kind = str(node[0])
    if kind == "integer":
        return ["integer", int(node[1])]
    if kind == "symbol":
        code = int(node[1])
        declaration = declarations.get(code)
        if declaration is None:
            return ["unbound-symbol"]
        if code in resolving:
            return ["recursive-declaration"]
        fields = tuple(declaration[2])
        return [
            "declared-symbol",
            int(fields[1][1]),
            _resolved_node(
                fields[2],
                declarations,
                resolving=resolving | {code},
            ),
        ]
    if kind == "reify":
        resolved = [
            _resolved_node(node[1], declarations, resolving=resolving),
            *(
                _resolved_node(child, declarations, resolving=resolving)
                for child in tuple(node[2])
            ),
        ]
        return ["reify", resolved]
    if kind != "call":
        raise PublicOperationIdentifiabilityError("COMMAND node differs")
    head = int(node[1])
    children = [
        _resolved_node(child, declarations, resolving=resolving)
        for child in tuple(node[2])
    ]
    if head in {1, 2}:
        children.sort(key=_digest)
    return ["call", head, children]


def resolved_operations(root: PublicNode) -> tuple[object, ...]:
    declarations, operations = _command_parts(root)
    return tuple(_resolved_node(item, declarations) for item in operations)


def _mutation_target(operation: object) -> str:
    mutations = getattr(operation, "mutations", None)
    cursor = getattr(operation, "cursor", None)
    if not isinstance(mutations, tuple) or type(cursor) is not int:
        raise PublicOperationIdentifiabilityError("operation trace differs")
    values = []
    for mutation in mutations:
        value = getattr(mutation, "value", None)
        kind = getattr(getattr(value, "kind", None), "value", None)
        index = getattr(value, "index", None)
        fields = (
            int(getattr(mutation, "opcode")),
            int(getattr(mutation, "source")),
            int(getattr(mutation, "target")),
            int(getattr(mutation, "relation")),
            int(getattr(mutation, "type_index")),
            kind,
            index,
        )
        values.append(fields)
    return _digest({"cursor": cursor, "mutations": values})


def _record_labels(
    record: object,
    codec: TokenNativeSurfaceCodec,
) -> Iterable[tuple[str, str, str]]:
    from audit_ettr_program_templates import (  # noqa: PLC0415
        _corner_from_targets,
        _packet_from_value,
    )
    from audit_ettr_public_opcode_identifiability import (  # noqa: PLC0415
        parse_public_transport,
        public_document_indices,
    )

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
    del initial_packets
    worlds, commands = _source_orbits(record, codec)
    views = tuple(record.source_visible.views)
    for world_index in range(2):
        for command_index in range(2):
            corner = 2 * world_index + command_index
            source = views[0].command_sources[corner]
            command_tree = parse_public_transport(
                public_document_indices(codec, source),
                codebook_size=len(codec.codebook.token_ids),
            )
            operations = resolved_operations(command_tree)
            traces = tuple(corners[corner].operation_traces)
            if len(operations) != len(traces):
                raise PublicOperationIdentifiabilityError(
                    "public and assessor operation depths differ"
                )
            world = worlds[corner]["alpha_exact"]
            full = _digest(operations)
            for rank, (operation, trace) in enumerate(
                zip(operations, traces, strict=True)
            ):
                operation_digest = _digest(operation)
                keys = {
                    "resolved_operation": operation_digest,
                    "world_resolved_operation": _digest(
                        {"operation": operation_digest, "world": world}
                    ),
                    "world_prefix": _digest(
                        {"prefix": operations[: rank + 1], "world": world}
                    ),
                    "world_full_command_rank": _digest(
                        {"command": full, "rank": rank, "world": world}
                    ),
                }
                target = _mutation_target(trace)
                for mode, key in keys.items():
                    yield mode, key, target


def _new_counts() -> dict[str, dict[str, Counter[str]]]:
    return {mode: {} for mode in _MODES}


def _audit_shard(
    arguments: tuple[Path, Path, str, Path],
) -> tuple[dict[str, dict[str, Counter[str]]], set[str], dict[str, object]]:
    path, data_root, split, tokenizer = arguments
    codec = TokenNativeSurfaceCodec(tokenizer)
    counts = _new_counts()
    core_ids: set[str] = set()
    rows = 0
    operation_instances = 0
    digest, size = _sha256_file(path)
    for payload, record in _iter_records(path):
        if record.canonical_bytes() != payload or record.identity.split != split:
            raise PublicOperationIdentifiabilityError("semantic-core record differs")
        if record.identity.core_id in core_ids:
            raise PublicOperationIdentifiabilityError("duplicate semantic-core identity")
        core_ids.add(record.identity.core_id)
        rows += 1
        labels = tuple(_record_labels(record, codec))
        operation_instances += len(labels) // len(_MODES)
        for mode, key, target in labels:
            counts[mode].setdefault(key, Counter())[target] += 1
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
        raise PublicOperationIdentifiabilityError(f"split shard set differs: {split}")
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
            raise PublicOperationIdentifiabilityError(
                "duplicate semantic-core identity across shards"
            )
        core_ids.update(shard_ids)
        for mode in _MODES:
            for key, labels in source[mode].items():
                counts[mode].setdefault(key, Counter()).update(labels)
        receipts.append(receipt)
    return {"core_rows": len(core_ids), "shards": receipts, "_counts": counts}


def audit(
    data_root: Path,
    tokenizer: Path,
    *,
    workers: int = 1,
) -> dict[str, object]:
    if workers < 1:
        raise PublicOperationIdentifiabilityError("worker count differs")
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
        mode: {
            "development_oracle": _conditional_summary(development[mode]),
            "train_fit": _conditional_summary(train[mode]),
            "train_to_development": _train_to_development(
                train[mode], development[mode]
            ),
        }
        for mode in _MODES
    }
    report = {
        "analyses": analyses,
        "data_root": str(data_root),
        "input_contract": {
            "answer_read": False,
            "assessor_operation_trace_used_as_label_only": True,
            "declarations_resolved_from_public_command": True,
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
