#!/usr/bin/env python3
"""Measure whether evolving typed state resolves operation-family ambiguity."""

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

from audit_ettr_operation_effect_family_identifiability import (
    abstract_resolved_operation,
)
from audit_ettr_operation_effect_kind_balance import (
    effect_kinds,
    operation_effect_family,
)
from audit_ettr_program_templates import _corner_from_targets, _packet_from_value
from audit_ettr_public_opcode_identifiability import (
    _source_orbits,
    parse_public_transport,
    public_document_indices,
)
from audit_ettr_public_operation_identifiability import resolved_operations
from audit_ettr_public_operation_state_delta import (
    runtime_state_value,
    state_delta_value,
)
from ettr_il_v2_materialize import (
    _encode_mutation,
    _independent_replay,
    _project_initial,
)
from ettr_il_v2_token_native_surface import TokenNativeSurfaceCodec
from ettr_il_v3_protocol import canonical_json_bytes
from materialize_ettr_il_v3_corpus import _iter_records, _sha256_file


REPORT_SCHEMA = "r12-ettr-operation-family-state-conditioning-audit-v1"
FAMILIES = ("none", "write", "link")
MODES = ("syntax", "syntax_state_topology", "syntax_state_exact")


class OperationFamilyStateConditioningError(ValueError):
    """A state-conditioned family feature or custody receipt differs."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def _feature(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def operation_features(value: object, *, prefix: str = "operation") -> set[str]:
    """Return renderer-invariant factor features for one resolved operation."""

    result: set[str] = set()

    def visit(node: object, path: tuple[int, ...]) -> None:
        if not isinstance(node, (list, tuple)) or not node:
            raise OperationFamilyStateConditioningError(
                "resolved operation node differs"
            )
        kind = node[0]
        if not isinstance(kind, str):
            raise OperationFamilyStateConditioningError(
                "resolved operation kind differs"
            )
        result.add(_feature([prefix, "kind", list(path), kind]))
        if kind == "integer":
            if len(node) != 2 or not isinstance(node[1], int):
                raise OperationFamilyStateConditioningError(
                    "resolved integer differs"
                )
            value = int(node[1])
            bucket = 0 if value == 0 else min(16, int(math.log2(abs(value))) + 1)
            result.add(_feature([prefix, "integer-sign", list(path), value < 0]))
            result.add(_feature([prefix, "integer-bucket", list(path), bucket]))
            return
        if kind == "declared-symbol":
            if len(node) != 3 or not isinstance(node[1], int):
                raise OperationFamilyStateConditioningError(
                    "resolved declaration differs"
                )
            result.add(
                _feature([prefix, "declaration-rank", list(path), int(node[1])])
            )
            visit(node[2], path + (0,))
            return
        if kind in {"unbound-symbol", "recursive-declaration"}:
            if len(node) != 1:
                raise OperationFamilyStateConditioningError(
                    "resolved symbol sentinel differs"
                )
            return
        if kind == "reify":
            if len(node) != 2 or not isinstance(node[1], (list, tuple)):
                raise OperationFamilyStateConditioningError("resolved reify differs")
            result.add(_feature([prefix, "arity", list(path), len(node[1])]))
            for index, child in enumerate(node[1]):
                visit(child, path + (index,))
            return
        if kind == "call":
            if (
                len(node) != 3
                or not isinstance(node[1], int)
                or not isinstance(node[2], (list, tuple))
            ):
                raise OperationFamilyStateConditioningError("resolved call differs")
            result.add(_feature([prefix, "head", list(path), int(node[1])]))
            result.add(_feature([prefix, "arity", list(path), len(node[2])]))
            for index, child in enumerate(node[2]):
                visit(child, path + (index,))
            return
        raise OperationFamilyStateConditioningError(
            "resolved operation kind differs"
        )

    visit(value, ())
    return result


def state_features(state: object, *, exact_values: bool) -> set[str]:
    """Factor one oracle preceding typed state without reading its successor."""

    value = runtime_state_value(state)
    nodes = tuple(value["nodes"])
    edges = tuple(value["edges"])
    status = tuple(value["status"])
    result = {
        _feature(["state", "active-count", len(nodes)]),
        _feature(["state", "edge-count", len(edges)]),
        _feature(["state", "status", list(status)]),
    }
    type_histogram: Counter[int] = Counter()
    for item in nodes:
        slot, active, type_index, value_code, root = item
        if not bool(active):
            raise OperationFamilyStateConditioningError(
                "runtime state contains an inactive listed node"
            )
        type_histogram[int(type_index)] += 1
        result.add(
            _feature(
                ["state", "slot-type-root", int(slot), int(type_index), bool(root)]
            )
        )
        if exact_values:
            result.add(
                _feature(["state", "slot-value", int(slot), int(value_code)])
            )
    for type_index, count in sorted(type_histogram.items()):
        result.add(_feature(["state", "type-count", type_index, count]))
    relation_histogram: Counter[int] = Counter()
    for relation, source, target in edges:
        relation_histogram[int(relation)] += 1
        result.add(
            _feature(
                ["state", "edge", int(relation), int(source), int(target)]
            )
        )
    for relation, count in sorted(relation_histogram.items()):
        result.add(_feature(["state", "relation-count", relation, count]))
    return result


def _operation_contexts(record: object) -> tuple[tuple[str, object], ...]:
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
    result: list[tuple[str, object]] = []
    for world_index in range(2):
        initial, static_ranks = _project_initial(
            initial_packets[world_index], f"world {world_index}"
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
                        state, steps, f"corner {corner} operation {rank}"
                    )[0]
                    if steps
                    else state
                )
                family = operation_effect_family(
                    effect_kinds(state_delta_value(state, after))
                )
                result.append((family, state))
                state = after
    return tuple(result)


def _record_examples(
    record: object,
    codec: TokenNativeSurfaceCodec,
) -> Iterable[tuple[str, str, tuple[str, ...]]]:
    contexts = iter(_operation_contexts(record))
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
            for rank, (operation, abstract_operation) in enumerate(
                zip(operations, abstract, strict=True)
            ):
                try:
                    family, state = next(contexts)
                except StopIteration as exc:
                    raise OperationFamilyStateConditioningError(
                        "public operations exceed exact family contexts"
                    ) from exc
                syntax = operation_features(abstract_operation)
                syntax.add(_feature(["command", "rank", rank]))
                syntax.add(_feature(["command", "length", len(operations)]))
                syntax.add(
                    _feature(
                        ["world", "topology", worlds[corner]["topology"]]
                    )
                )
                for prefix_rank, item in enumerate(abstract[: rank + 1]):
                    syntax.update(
                        operation_features(
                            item,
                            prefix=f"prefix-distance-{rank - prefix_rank}",
                        )
                    )
                topology = syntax | state_features(state, exact_values=False)
                exact = syntax | state_features(state, exact_values=True)
                for mode, features in (
                    ("syntax", syntax),
                    ("syntax_state_topology", topology),
                    ("syntax_state_exact", exact),
                ):
                    yield mode, family, tuple(sorted(features))
    try:
        next(contexts)
    except StopIteration:
        return
    raise OperationFamilyStateConditioningError(
        "exact family contexts exceed public operations"
    )


def _new_feature_counts() -> dict[str, dict[str, Counter[str]]]:
    return {mode: {} for mode in MODES}


def _audit_shard(
    arguments: tuple[Path, Path, str, Path],
) -> tuple[dict[str, object], dict[str, object]]:
    path, data_root, split, tokenizer = arguments
    codec = TokenNativeSurfaceCodec(tokenizer)
    family_counts = {mode: Counter() for mode in MODES}
    feature_counts = _new_feature_counts()
    feature_totals = {mode: Counter() for mode in MODES}
    signatures = {mode: {} for mode in MODES}
    signature_features = {mode: {} for mode in MODES}
    rows = 0
    operations = 0
    core_ids: set[str] = set()
    digest, size = _sha256_file(path)
    for payload, record in _iter_records(path):
        if record.canonical_bytes() != payload or record.identity.split != split:
            raise OperationFamilyStateConditioningError(
                "semantic-core record differs"
            )
        if record.identity.core_id in core_ids:
            raise OperationFamilyStateConditioningError(
                "duplicate semantic-core identity"
            )
        core_ids.add(record.identity.core_id)
        examples = tuple(_record_examples(record, codec))
        operations += len(examples) // len(MODES)
        for mode, family, features in examples:
            family_counts[mode][family] += 1
            signature = _feature([mode, list(features)])
            signatures[mode].setdefault(signature, Counter())[family] += 1
            if split == "development":
                previous = signature_features[mode].setdefault(signature, features)
                if previous != features:
                    raise OperationFamilyStateConditioningError(
                        "operation family feature digest collides"
                    )
            for feature in features:
                feature_counts[mode].setdefault(feature, Counter())[family] += 1
                feature_totals[mode][family] += 1
        rows += 1
    return (
        {
            "core_ids": core_ids,
            "family_counts": family_counts,
            "feature_counts": feature_counts,
            "feature_totals": feature_totals,
            "signature_features": signature_features,
            "signatures": signatures,
        },
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
) -> tuple[dict[str, object], dict[str, object]]:
    root = data_root / split
    paths = tuple(sorted(root.glob("*.jsonl.gz"))) if root.is_dir() else ()
    if not paths or any(path.is_symlink() or not path.is_file() for path in paths):
        raise OperationFamilyStateConditioningError(
            f"split shard set differs: {split}"
        )
    arguments = tuple((path, data_root, split, tokenizer) for path in paths)
    if workers == 1:
        results = tuple(_audit_shard(argument) for argument in arguments)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(paths))) as pool:
            results = tuple(pool.map(_audit_shard, arguments))
    merged = {
        "family_counts": {mode: Counter() for mode in MODES},
        "feature_counts": _new_feature_counts(),
        "feature_totals": {mode: Counter() for mode in MODES},
        "signature_features": {mode: {} for mode in MODES},
        "signatures": {mode: {} for mode in MODES},
    }
    core_ids: set[str] = set()
    receipts = []
    for source, receipt in results:
        shard_ids = set(source["core_ids"])
        if core_ids.intersection(shard_ids):
            raise OperationFamilyStateConditioningError(
                "duplicate semantic-core identity across shards"
            )
        core_ids.update(shard_ids)
        for mode in MODES:
            merged["family_counts"][mode].update(source["family_counts"][mode])
            merged["feature_totals"][mode].update(source["feature_totals"][mode])
            for feature, labels in source["feature_counts"][mode].items():
                merged["feature_counts"][mode].setdefault(
                    feature, Counter()
                ).update(labels)
            for signature, labels in source["signatures"][mode].items():
                merged["signatures"][mode].setdefault(
                    signature, Counter()
                ).update(labels)
                if signature in source["signature_features"][mode]:
                    features = source["signature_features"][mode][signature]
                    previous = merged["signature_features"][mode].setdefault(
                        signature, features
                    )
                    if previous != features:
                        raise OperationFamilyStateConditioningError(
                            "operation family feature digest collides across shards"
                        )
        receipts.append(receipt)
    return merged, {"core_rows": len(core_ids), "shards": receipts}


def _conditional_summary(counts: Mapping[str, Counter[str]]) -> dict[str, object]:
    instances = sum(sum(labels.values()) for labels in counts.values())
    correct = sum(max(labels.values()) for labels in counts.values() if labels)
    ambiguous = sum(
        sum(labels.values())
        for labels in counts.values()
        if sum(int(count > 0) for count in labels.values()) > 1
    )
    return {
        "accuracy": correct / instances if instances else 0.0,
        "ambiguous_instances": ambiguous,
        "instances": instances,
        "keys": len(counts),
    }


def _train_to_development(
    train: Mapping[str, Counter[str]],
    development: Mapping[str, Counter[str]],
) -> dict[str, object]:
    total = sum(sum(labels.values()) for labels in development.values())
    seen = 0
    correct = 0
    for key, labels in development.items():
        source = train.get(key)
        if not source:
            continue
        prediction = max(FAMILIES, key=lambda family: (source[family], -FAMILIES.index(family)))
        seen += sum(labels.values())
        correct += labels[prediction]
    return {
        "all_accuracy": correct / total if total else 0.0,
        "coverage": seen / total if total else 0.0,
        "seen_accuracy": correct / seen if seen else 0.0,
        "seen_instances": seen,
        "total_instances": total,
    }


def _multinomial_nb(
    train: Mapping[str, object],
    development: Mapping[str, object],
    mode: str,
    *,
    alpha: float = 1.0,
) -> dict[str, object]:
    train_family: Counter[str] = train["family_counts"][mode]
    train_features: Mapping[str, Counter[str]] = train["feature_counts"][mode]
    train_totals: Counter[str] = train["feature_totals"][mode]
    development_signatures: Mapping[str, Counter[str]] = development["signatures"][mode]
    development_features: Mapping[str, tuple[str, ...]] = development[
        "signature_features"
    ][mode]
    vocabulary = len(train_features)
    family_total = sum(train_family.values())
    confusion = {family: Counter() for family in FAMILIES}
    correct = 0
    total = 0
    for signature, labels in development_signatures.items():
        features = development_features[signature]

        def score(family: str) -> float:
            result = math.log(
                (train_family[family] + alpha) / (family_total + 3 * alpha)
            )
            denominator = train_totals[family] + alpha * max(1, vocabulary)
            for feature in features:
                result += math.log(
                    (train_features.get(feature, {}).get(family, 0) + alpha)
                    / denominator
                )
            return result

        prediction = max(
            FAMILIES,
            key=lambda family: (score(family), -FAMILIES.index(family)),
        )
        for expected, count in labels.items():
            confusion[expected][prediction] += count
            correct += count if expected == prediction else 0
            total += count
    return {
        "accuracy": correct / total if total else 0.0,
        "confusion": {
            expected: dict(confusion[expected]) for expected in FAMILIES
        },
        "instances": total,
        "method": "multinomial_naive_bayes_factor_features",
        "train_feature_vocabulary": vocabulary,
    }


def audit(
    data_root: Path,
    tokenizer: Path,
    *,
    workers: int = 1,
) -> dict[str, object]:
    if workers < 1:
        raise OperationFamilyStateConditioningError("worker count differs")
    data_root = data_root.resolve()
    tokenizer = tokenizer.resolve()
    tokenizer_sha256, tokenizer_bytes = _sha256_file(tokenizer)
    train, train_receipt = _audit_split(data_root, "train", tokenizer, workers)
    development, development_receipt = _audit_split(
        data_root, "development", tokenizer, workers
    )
    analyses = {}
    for mode in MODES:
        analyses[mode] = {
            "development_oracle": _conditional_summary(
                development["signatures"][mode]
            ),
            "factorized_transfer": _multinomial_nb(
                train, development, mode
            ),
            "train_to_development": _train_to_development(
                train["signatures"][mode], development["signatures"][mode]
            ),
        }
    report = {
        "analyses": analyses,
        "data_root": str(data_root),
        "decision_contract": {
            "family_gate": 0.90,
            "oracle_state_below_gate": "reject_standalone_family_primitive",
            "oracle_state_crosses_gate": "state_conditioned_latent_rail_arbiter",
            "syntax_crosses_gate": "operation_family_island_optimization",
        },
        "input_contract": {
            "answer_read": False,
            "assessor_operation_trace_used_as_family_label_only": True,
            "oracle_preceding_state_used_for_state_modes_only": True,
            "query_read": False,
            "successor_state_used_as_feature": False,
            "terminal_packet_used_as_feature": False,
            "transaction_program_used_as_feature": False,
        },
        "schema": REPORT_SCHEMA,
        "splits": {
            "development": development_receipt,
            "train": train_receipt,
        },
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
