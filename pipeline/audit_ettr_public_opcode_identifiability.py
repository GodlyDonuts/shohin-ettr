#!/usr/bin/env python3
"""Measure whether public ETTR syntax identifies a coherent opcode program.

The schedule learner currently predicts transaction fields independently.  A
valid-program projection can remove impossible opcode hybrids, but it cannot
create opcode information that is absent from the permitted input.  This
CPU-only audit estimates that information boundary directly.

Only model-visible WORLD and COMMAND transports are used to construct input
signatures.  Deterministic cover is removed, all four renderers are parsed to
one canonical tree, and opaque identifiers are alpha-renamed by first semantic
occurrence.  Assessor-only traces are used solely as labels after the public
signature is sealed.  QUERY and answer fields are never read.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from audit_ettr_program_templates import _record_programs
from ettr_il_v2_surface import MAX_HEAD
from ettr_il_v2_token_native_surface import (
    MAX_NATIVE_ARITY,
    TokenNativeSurfaceCodec,
)
from ettr_il_v3_protocol import canonical_json_bytes
from materialize_ettr_il_v3_corpus import _iter_records, _sha256_file


REPORT_SCHEMA = "r12-ettr-public-opcode-identifiability-audit-v1"
_SPLITS = ("train", "development")
_SIGNATURE_MODES = ("alpha_exact", "alpha_operator", "topology")
_INPUT_VARIANTS = ("command", "world_command")
_CALL_STRIDE = MAX_NATIVE_ARITY + 1
_CALL_END = (MAX_HEAD + 1) * _CALL_STRIDE
_FRAME_A = _CALL_END
_FRAME_B = _FRAME_A + 1
_FRAME_END = _FRAME_B + 1
_FRAME_FILL = _FRAME_END + 1
_REIFY_BASE = _FRAME_FILL + 1
_REIFY_END = _REIFY_BASE + MAX_NATIVE_ARITY + 1
_INTEGER_BASE = _REIFY_END
_IDENTIFIER_WINDOW = 96
_ROOT_GEOMETRY = {(14, 3), (15, 2)}

PublicNode = tuple[object, ...]
LabelCounts = dict[str, Counter[str]]


class PublicOpcodeIdentifiabilityError(ValueError):
    """A public syntax or audit contract differs."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _leaf_or_operator(code: int, codebook_size: int) -> tuple[str, int, int]:
    if 0 <= code < _CALL_END:
        head, arity = divmod(code, _CALL_STRIDE)
        return "call", head, arity
    if _REIFY_BASE <= code < _REIFY_END:
        return "reify", 2, code - _REIFY_BASE
    identifier_floor = codebook_size - _IDENTIFIER_WINDOW
    if _INTEGER_BASE <= code < identifier_floor:
        return "integer", code - _INTEGER_BASE, 0
    if identifier_floor <= code < codebook_size:
        return "symbol", code, 0
    raise PublicOpcodeIdentifiabilityError("public AST contains an invalid code")


def _is_root(node: PublicNode) -> bool:
    return (
        len(node) == 3
        and node[0] == "call"
        and (int(node[1]), len(tuple(node[2]))) in _ROOT_GEOMETRY
    )


def parse_public_transport(
    physical: Sequence[int],
    *,
    codebook_size: int,
) -> PublicNode:
    """Delete cover and parse one renderer-independent public AST."""

    codes = tuple(int(value) for value in physical)
    if (
        len(codes) < 3
        or codes[0] not in {_FRAME_A, _FRAME_B}
        or codes[1] not in {_FRAME_A, _FRAME_B}
        or codebook_size <= _INTEGER_BASE + _IDENTIFIER_WINDOW
        or min(codes) < 0
        or max(codes) >= codebook_size
    ):
        raise PublicOpcodeIdentifiabilityError("public transport geometry differs")
    prefix = codes[0] == _FRAME_A
    reverse_children = codes[1] == _FRAME_B
    body = codes[2:]

    def node_from(
        kind: str,
        value: int,
        children: tuple[PublicNode, ...] = (),
    ) -> PublicNode:
        if reverse_children:
            children = tuple(reversed(children))
        if kind == "call":
            return ("call", value, children)
        if kind == "reify":
            if len(children) != value + 1 or children[0][0] != "symbol":
                raise PublicOpcodeIdentifiabilityError(
                    "public fused reification differs"
                )
            return ("reify", children[0], children[1:])
        if children:
            raise PublicOpcodeIdentifiabilityError("public leaf has children")
        return (kind, value)

    if prefix:
        offset = 0

        def consume(depth: int) -> PublicNode:
            nonlocal offset
            if depth > len(body) or offset >= len(body):
                raise PublicOpcodeIdentifiabilityError(
                    "public prefix AST does not terminate"
                )
            kind, value, arity = _leaf_or_operator(body[offset], codebook_size)
            offset += 1
            children = tuple(consume(depth + 1) for _ in range(arity))
            return node_from(kind, value, children)

        root = consume(1)
        if not _is_root(root):
            raise PublicOpcodeIdentifiabilityError("public prefix root differs")
        return root

    stack: list[PublicNode] = []
    for code in body:
        kind, value, arity = _leaf_or_operator(code, codebook_size)
        if len(stack) < arity:
            raise PublicOpcodeIdentifiabilityError("public postfix arity underflows")
        children = tuple(stack[-arity:]) if arity else ()
        if arity:
            del stack[-arity:]
        stack.append(node_from(kind, value, children))
        if len(stack) == 1 and _is_root(stack[0]):
            return stack[0]
    raise PublicOpcodeIdentifiabilityError("public postfix AST does not terminate")


def canonical_public_tree(node: PublicNode, *, mode: str) -> object:
    """Return one alpha-normalized public tree at a selected abstraction."""

    if mode not in _SIGNATURE_MODES:
        raise PublicOpcodeIdentifiabilityError("public signature mode differs")
    symbols: dict[int, int] = {}

    def visit(current: PublicNode) -> object:
        kind = str(current[0])
        if kind == "symbol":
            value = int(current[1])
            if mode == "topology":
                return ["symbol"]
            ordinal = symbols.setdefault(value, len(symbols))
            return ["symbol", ordinal]
        if kind == "integer":
            if mode == "alpha_exact":
                return ["integer", int(current[1])]
            return ["integer"]
        if kind == "call":
            return [
                "call",
                int(current[1]),
                [visit(child) for child in tuple(current[2])],
            ]
        if kind == "reify":
            return [
                "reify",
                visit(current[1]),
                [visit(child) for child in tuple(current[2])],
            ]
        raise PublicOpcodeIdentifiabilityError("public tree node differs")

    return visit(node)


def public_source_signatures(
    codec: TokenNativeSurfaceCodec,
    source: str,
) -> dict[str, str]:
    if not isinstance(source, str) or not source.isascii():
        raise PublicOpcodeIdentifiabilityError("public source differs")
    physical = codec._payload_indices(source.encode("ascii"))
    tree = parse_public_transport(
        physical,
        codebook_size=len(codec.codebook.token_ids),
    )
    return {
        mode: _digest(canonical_public_tree(tree, mode=mode))
        for mode in _SIGNATURE_MODES
    }


def _source_orbits(
    record: object,
    codec: TokenNativeSurfaceCodec,
) -> tuple[tuple[dict[str, str], ...], tuple[dict[str, str], ...]]:
    views = tuple(record.source_visible.views)
    if len(views) != 4 or {int(view.renderer) for view in views} != set(range(4)):
        raise PublicOpcodeIdentifiabilityError("public renderer orbit differs")
    worlds = []
    commands = []
    for index in range(2):
        world_orbit = tuple(
            public_source_signatures(codec, view.world_sources[index]) for view in views
        )
        command_orbit = tuple(
            public_source_signatures(codec, view.command_sources[index]) for view in views
        )
        if any(item != world_orbit[0] for item in world_orbit[1:]):
            raise PublicOpcodeIdentifiabilityError(
                "WORLD public signature changes across renderer orbit"
            )
        if any(item != command_orbit[0] for item in command_orbit[1:]):
            raise PublicOpcodeIdentifiabilityError(
                "COMMAND public signature changes across renderer orbit"
            )
        worlds.append(world_orbit[0])
        commands.append(command_orbit[0])
    return tuple(worlds), tuple(commands)


def _record_labels(
    record: object,
    codec: TokenNativeSurfaceCodec,
) -> Iterable[tuple[str, str, str, str]]:
    programs = tuple(_record_programs(record))
    if len(programs) != 4:
        raise PublicOpcodeIdentifiabilityError("program rectangle differs")
    worlds, commands = _source_orbits(record, codec)
    for world_index in range(2):
        for command_index in range(2):
            corner = 2 * world_index + command_index
            target = str(programs[corner]["opcode"])
            for mode in _SIGNATURE_MODES:
                command_key = _digest(
                    {"command": commands[command_index][mode], "mode": mode}
                )
                joint_key = _digest(
                    {
                        "command": commands[command_index][mode],
                        "mode": mode,
                        "world": worlds[world_index][mode],
                    }
                )
                yield mode, "command", command_key, target
                yield mode, "world_command", joint_key, target


def _new_counts() -> dict[str, dict[str, LabelCounts]]:
    return {
        mode: {variant: {} for variant in _INPUT_VARIANTS}
        for mode in _SIGNATURE_MODES
    }


def _audit_shard(
    arguments: tuple[Path, Path, str, Path],
) -> tuple[dict[str, dict[str, LabelCounts]], set[str], dict[str, object]]:
    path, data_root, split, tokenizer = arguments
    codec = TokenNativeSurfaceCodec(tokenizer)
    counts = _new_counts()
    core_ids: set[str] = set()
    rows = 0
    digest, size = _sha256_file(path)
    for payload, record in _iter_records(path):
        if record.canonical_bytes() != payload or record.identity.split != split:
            raise PublicOpcodeIdentifiabilityError("semantic-core record differs")
        if record.identity.core_id in core_ids:
            raise PublicOpcodeIdentifiabilityError("duplicate semantic-core identity")
        core_ids.add(record.identity.core_id)
        rows += 1
        for mode, variant, key, target in _record_labels(record, codec):
            labels = counts[mode][variant].setdefault(key, Counter())
            labels[target] += 1
    return (
        counts,
        core_ids,
        {
            "bytes": size,
            "path": path.relative_to(data_root).as_posix(),
            "rows": rows,
            "sha256": digest,
        },
    )


def _shards(data_root: Path, split: str) -> tuple[Path, ...]:
    root = data_root / split
    paths = tuple(sorted(root.glob("*.jsonl.gz"))) if root.is_dir() else ()
    if not paths or any(path.is_symlink() or not path.is_file() for path in paths):
        raise PublicOpcodeIdentifiabilityError(f"split shard set differs: {split}")
    return paths


def _merge_counts(
    destination: dict[str, dict[str, LabelCounts]],
    source: Mapping[str, Mapping[str, LabelCounts]],
) -> None:
    for mode in _SIGNATURE_MODES:
        for variant in _INPUT_VARIANTS:
            for key, labels in source[mode][variant].items():
                destination[mode][variant].setdefault(key, Counter()).update(labels)


def _audit_split(
    data_root: Path,
    split: str,
    tokenizer: Path,
    workers: int,
) -> dict[str, object]:
    if not isinstance(workers, int) or workers < 1:
        raise PublicOpcodeIdentifiabilityError("worker count differs")
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
    for shard_counts, shard_ids, receipt in results:
        if core_ids.intersection(shard_ids):
            raise PublicOpcodeIdentifiabilityError(
                "duplicate semantic-core identity across shards"
            )
        core_ids.update(shard_ids)
        _merge_counts(counts, shard_counts)
        receipts.append(receipt)
    return {
        "core_rows": len(core_ids),
        "shards": receipts,
        "_counts": counts,
    }


def _conditional_summary(counts: Mapping[str, Counter[str]]) -> dict[str, object]:
    instances = sum(sum(labels.values()) for labels in counts.values())
    correct = sum(max(labels.values()) for labels in counts.values() if labels)
    conditional_entropy = 0.0
    for labels in counts.values():
        total = sum(labels.values())
        conditional_entropy += sum(
            -count * math.log2(count / total) for count in labels.values()
        )
    return {
        "ambiguous_signatures": sum(len(labels) > 1 for labels in counts.values()),
        "bayes_correct": correct,
        "bayes_rate": correct / instances if instances else 0.0,
        "conditional_entropy_bits": (
            conditional_entropy / instances if instances else 0.0
        ),
        "instances": instances,
        "signatures": len(counts),
    }


def _train_to_development(
    train: Mapping[str, Counter[str]],
    development: Mapping[str, Counter[str]],
) -> dict[str, object]:
    total = sum(sum(labels.values()) for labels in development.values())
    seen = 0
    correct = 0
    ties = 0
    for key, labels in development.items():
        train_labels = train.get(key)
        if not train_labels:
            continue
        seen += sum(labels.values())
        maximum = max(train_labels.values())
        modal = sorted(
            label for label, count in train_labels.items() if count == maximum
        )
        ties += len(modal) > 1
        correct += labels[modal[0]]
    return {
        "accuracy_all": correct / total if total else 0.0,
        "accuracy_seen": correct / seen if seen else 0.0,
        "correct": correct,
        "development_instances": total,
        "seen_instances": seen,
        "seen_rate": seen / total if total else 0.0,
        "training_modal_ties": ties,
        "unseen_instances": total - seen,
    }


def audit(
    data_root: Path,
    tokenizer: Path,
    *,
    workers: int = 1,
) -> dict[str, object]:
    data_root = data_root.resolve()
    tokenizer = tokenizer.resolve()
    if tokenizer.is_symlink() or not tokenizer.is_file():
        raise PublicOpcodeIdentifiabilityError("tokenizer artifact differs")
    tokenizer_sha256, tokenizer_bytes = _sha256_file(tokenizer)
    splits = {
        split: _audit_split(data_root, split, tokenizer, workers)
        for split in _SPLITS
    }
    train = splits["train"].pop("_counts")
    development = splits["development"].pop("_counts")
    analyses = {}
    for mode in _SIGNATURE_MODES:
        analyses[mode] = {}
        for variant in _INPUT_VARIANTS:
            analyses[mode][variant] = {
                "development_oracle": _conditional_summary(
                    development[mode][variant]
                ),
                "train_fit": _conditional_summary(train[mode][variant]),
                "train_to_development": _train_to_development(
                    train[mode][variant],
                    development[mode][variant],
                ),
            }
    report = {
        "analyses": analyses,
        "data_root": str(data_root),
        "input_contract": {
            "answer_read": False,
            "assessor_program_used_as_label_only": True,
            "cover_removed": True,
            "opaque_identifiers_alpha_renamed": True,
            "query_read": False,
            "renderer_orbit_required_exact": True,
            "sources": ["world_sources", "command_sources"],
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
