#!/usr/bin/env python3
"""Audit whether public operation/state context identifies NONE/WRITE/LINK."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

from audit_ettr_operation_effect_kind_balance import (
    effect_kinds,
    operation_effect_family,
)
from audit_ettr_program_templates import _corner_from_targets, _packet_from_value
from audit_ettr_public_opcode_identifiability import (
    _conditional_summary,
    _source_orbits,
    _train_to_development,
    parse_public_transport,
    public_document_indices,
)
from audit_ettr_public_operation_identifiability import (
    _digest,
    resolved_operations,
)
from audit_ettr_public_operation_state_delta import state_delta_value
from ettr_il_v2_materialize import (
    _encode_mutation,
    _independent_replay,
    _project_initial,
)
from ettr_il_v2_token_native_surface import TokenNativeSurfaceCodec
from ettr_il_v3_protocol import canonical_json_bytes
from materialize_ettr_il_v3_corpus import _iter_records, _sha256_file


REPORT_SCHEMA = "r12-ettr-operation-effect-family-identifiability-audit-v1"
_SPLITS = ("train", "development")
_MODES = (
    "resolved_operation",
    "abstract_operation",
    "world_topology_abstract_operation",
    "world_alpha_operator_prefix",
    "world_alpha_operator_full_command_rank",
)


class OperationEffectFamilyIdentifiabilityError(ValueError):
    """A public feature, exact family label, or corpus receipt differs."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def abstract_resolved_operation(value: object) -> object:
    """Remove literal payload identity while retaining public typed structure."""

    if not isinstance(value, list) or not value or not isinstance(value[0], str):
        raise OperationEffectFamilyIdentifiabilityError(
            "resolved operation structure differs"
        )
    kind = value[0]
    if kind == "integer":
        if len(value) != 2 or not isinstance(value[1], int):
            raise OperationEffectFamilyIdentifiabilityError("resolved integer differs")
        return ["integer"]
    if kind == "declared-symbol":
        if len(value) != 3 or not isinstance(value[1], int):
            raise OperationEffectFamilyIdentifiabilityError(
                "resolved declaration differs"
            )
        return [
            "declared-symbol",
            int(value[1]),
            abstract_resolved_operation(value[2]),
        ]
    if kind in {"unbound-symbol", "recursive-declaration"}:
        if len(value) != 1:
            raise OperationEffectFamilyIdentifiabilityError(
                "resolved symbol sentinel differs"
            )
        return list(value)
    if kind == "reify":
        if len(value) != 2 or not isinstance(value[1], list):
            raise OperationEffectFamilyIdentifiabilityError("resolved reify differs")
        return [kind, [abstract_resolved_operation(item) for item in value[1]]]
    if kind == "call":
        if (
            len(value) != 3
            or not isinstance(value[1], int)
            or not isinstance(value[2], list)
        ):
            raise OperationEffectFamilyIdentifiabilityError("resolved call differs")
        return [
            kind,
            int(value[1]),
            [abstract_resolved_operation(item) for item in value[2]],
        ]
    raise OperationEffectFamilyIdentifiabilityError("resolved operation kind differs")


def _operation_families(record: object) -> tuple[str, ...]:
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
    families: list[str] = []
    for world_index in range(2):
        initial, static_ranks = _project_initial(
            initial_packets[world_index],
            f"world {world_index}",
        )
        for command_index in range(2):
            corner = 2 * world_index + command_index
            state = initial
            for rank, trace in enumerate(corners[corner].operation_traces):
                steps = tuple(
                    _encode_mutation(
                        mutation,
                        static_ranks,
                        f"corner {corner} operation {rank}",
                    )
                    for mutation in trace.mutations
                )
                after = (
                    _independent_replay(
                        state,
                        steps,
                        f"corner {corner} operation {rank}",
                    )[0]
                    if steps
                    else state
                )
                families.append(
                    operation_effect_family(
                        effect_kinds(state_delta_value(state, after))
                    )
                )
                state = after
    return tuple(families)


def _record_labels(
    record: object,
    codec: TokenNativeSurfaceCodec,
) -> Iterable[tuple[str, str, str]]:
    families = iter(_operation_families(record))
    worlds, _commands = _source_orbits(record, codec)
    views = tuple(record.source_visible.views)
    for world_index in range(2):
        for command_index in range(2):
            corner = 2 * world_index + command_index
            command_tree = parse_public_transport(
                public_document_indices(codec, views[0].command_sources[corner]),
                codebook_size=len(codec.codebook.token_ids),
            )
            operations = resolved_operations(command_tree)
            abstract = tuple(abstract_resolved_operation(item) for item in operations)
            full_abstract = _digest(abstract)
            world_topology = worlds[corner]["topology"]
            world_operator = worlds[corner]["alpha_operator"]
            for rank, (operation, abstract_operation) in enumerate(
                zip(operations, abstract, strict=True)
            ):
                try:
                    family = next(families)
                except StopIteration as exc:
                    raise OperationEffectFamilyIdentifiabilityError(
                        "public operations exceed exact family labels"
                    ) from exc
                keys = {
                    "resolved_operation": _digest(operation),
                    "abstract_operation": _digest(abstract_operation),
                    "world_topology_abstract_operation": _digest(
                        {
                            "operation": abstract_operation,
                            "world": world_topology,
                        }
                    ),
                    "world_alpha_operator_prefix": _digest(
                        {
                            "prefix": abstract[: rank + 1],
                            "world": world_operator,
                        }
                    ),
                    "world_alpha_operator_full_command_rank": _digest(
                        {
                            "command": full_abstract,
                            "rank": rank,
                            "world": world_operator,
                        }
                    ),
                }
                for mode, key in keys.items():
                    yield mode, key, family
    try:
        next(families)
    except StopIteration:
        return
    raise OperationEffectFamilyIdentifiabilityError(
        "exact family labels exceed public operations"
    )


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
    operations = 0
    digest, size = _sha256_file(path)
    for payload, record in _iter_records(path):
        if record.canonical_bytes() != payload or record.identity.split != split:
            raise OperationEffectFamilyIdentifiabilityError(
                "semantic-core record differs"
            )
        if record.identity.core_id in core_ids:
            raise OperationEffectFamilyIdentifiabilityError(
                "duplicate semantic-core identity"
            )
        core_ids.add(record.identity.core_id)
        labels = tuple(_record_labels(record, codec))
        operations += len(labels) // len(_MODES)
        for mode, key, family in labels:
            counts[mode].setdefault(key, Counter())[family] += 1
        rows += 1
    return (
        counts,
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
    tokenizer: Path,
    workers: int,
) -> dict[str, object]:
    root = data_root / split
    paths = tuple(sorted(root.glob("*.jsonl.gz"))) if root.is_dir() else ()
    if not paths or any(path.is_symlink() or not path.is_file() for path in paths):
        raise OperationEffectFamilyIdentifiabilityError(
            f"split shard set differs: {split}"
        )
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
            raise OperationEffectFamilyIdentifiabilityError(
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
        raise OperationEffectFamilyIdentifiabilityError("worker count differs")
    data_root = data_root.resolve()
    tokenizer = tokenizer.resolve()
    tokenizer_sha256, tokenizer_bytes = _sha256_file(tokenizer)
    splits = {
        split: _audit_split(data_root, split, tokenizer, workers) for split in _SPLITS
    }
    train = splits["train"].pop("_counts")
    development = splits["development"].pop("_counts")
    train_family = Counter(
        {
            family: sum(labels.get(family, 0) for labels in train[_MODES[0]].values())
            for family in ("none", "write", "link")
        }
    )
    development_family = Counter(
        {
            family: sum(
                labels.get(family, 0) for labels in development[_MODES[0]].values()
            )
            for family in ("none", "write", "link")
        }
    )
    analyses = {
        mode: {
            "development_oracle": _conditional_summary(development[mode]),
            "train_fit": _conditional_summary(train[mode]),
            "train_to_development": _train_to_development(
                train[mode],
                development[mode],
            ),
        }
        for mode in _MODES
    }
    report = {
        "analyses": analyses,
        "data_root": str(data_root),
        "family_histograms": {
            "development": dict(development_family),
            "development_majority_rate": max(development_family.values())
            / sum(development_family.values()),
            "train": dict(train_family),
            "train_majority_rate": max(train_family.values())
            / sum(train_family.values()),
        },
        "input_contract": {
            "answer_read": False,
            "assessor_operation_trace_used_as_family_label_only": True,
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
        workers=args.workers,
    )
    _write_no_replace(args.output, canonical_json_bytes(report))
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
