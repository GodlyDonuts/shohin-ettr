#!/usr/bin/env python3
"""Audit whether ETTR v3 queries are representable by the deployed reader.

The original source-deleted reader is permutation-invariant over state slots:
it embeds slot contents and relations but never supplies an address feature.
Several admitted query operators name absolute packet slots or ordered register
positions. One also compares the initial and terminal values of a slot. Those
targets are outside a terminal-only unaddressed reader's hypothesis class even
when the terminal packet is exact. This audit measures the affected query
support without loading model weights or exposing assessor targets to training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence


REPORT_SCHEMA = "shohin-ettr-v3-reader-identifiability-audit-v1"
_ASCII_NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")

# Each operator below requires either an absolute packet slot or the order of
# otherwise exchangeable runtime slots. The original reader has neither.
ADDRESS_SENSITIVE_OPERATORS = frozenset(
    {
        "adjacent_is",
        "pattern_exists",
        "resource_place_ge",
        "same_type_slots_equal",
        "slot_changed",
        "slot_is",
    }
)

# These operators are representable from value/type/relation/status multisets.
UNADDRESSED_REPRESENTABLE_OPERATORS = frozenset(
    {
        "horn_count_ge",
        "horn_has",
        "resource_cursor_ge",
        "resource_halt",
        "type_count_ge",
    }
)

# This predicate compares the terminal register against the episode's initial
# register. Addresses alone cannot make it identifiable from terminal state.
INITIAL_STATE_SENSITIVE_OPERATORS = frozenset({"slot_changed"})


class ReaderIdentifiabilityAuditError(ValueError):
    """The corpus does not satisfy the bounded audit contract."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _query_operators(row: object) -> tuple[str, ...]:
    if type(row) is not dict:
        raise ReaderIdentifiabilityAuditError("shard row is not an object")
    try:
        queries = row["assessor_only"]["semantic_factors"]["queries"]
    except (KeyError, TypeError) as exc:
        raise ReaderIdentifiabilityAuditError(
            "shard row lacks assessor query factors"
        ) from exc
    if type(queries) is not list or len(queries) != 2:
        raise ReaderIdentifiabilityAuditError("shard row query pair differs")
    result = []
    for query in queries:
        if type(query) is not dict or type(query.get("op")) is not str:
            raise ReaderIdentifiabilityAuditError("shard query operation differs")
        operation = query["op"]
        if operation not in (
            ADDRESS_SENSITIVE_OPERATORS | UNADDRESSED_REPRESENTABLE_OPERATORS
        ):
            raise ReaderIdentifiabilityAuditError(
                f"unclassified query operation: {operation}"
            )
        result.append(operation)
    return tuple(result)


def _shards(root: Path) -> tuple[Path, ...]:
    if not root.is_dir() or root.is_symlink():
        raise ReaderIdentifiabilityAuditError("data root must be a physical directory")
    paths = tuple(sorted(root.rglob("*.jsonl.gz")))
    if not paths:
        raise ReaderIdentifiabilityAuditError("data root has no gzip shards")
    return paths


def audit(
    root: Path,
    *,
    max_records_per_shard: int | None = None,
) -> dict[str, object]:
    """Return deterministic operator support and hypothesis-class coverage."""

    if max_records_per_shard is not None and max_records_per_shard < 1:
        raise ReaderIdentifiabilityAuditError("max records per shard must be positive")
    total = Counter()
    by_family: dict[str, Counter[str]] = {}
    shard_rows: dict[str, int] = {}
    digest = hashlib.sha256()
    for path in _shards(root):
        relative = path.relative_to(root).as_posix()
        family = path.name.split("-", 1)[0]
        if _ASCII_NAME.fullmatch(family) is None:
            raise ReaderIdentifiabilityAuditError("shard family differs")
        family_counts = by_family.setdefault(family, Counter())
        rows = 0
        try:
            with gzip.open(path, "rt", encoding="ascii", newline="") as handle:
                for line in handle:
                    if (
                        max_records_per_shard is not None
                        and rows >= max_records_per_shard
                    ):
                        break
                    row = json.loads(line)
                    operations = _query_operators(row)
                    total.update(operations)
                    family_counts.update(operations)
                    digest.update(relative.encode("ascii") + b"\0")
                    digest.update(_canonical_bytes(operations))
                    rows += 1
        except (OSError, EOFError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReaderIdentifiabilityAuditError(
                f"cannot audit shard {relative}"
            ) from exc
        if rows == 0:
            raise ReaderIdentifiabilityAuditError(f"audited shard is empty: {relative}")
        shard_rows[relative] = rows
    query_count = sum(total.values())
    affected = sum(total[op] for op in ADDRESS_SENSITIVE_OPERATORS)
    represented = sum(total[op] for op in UNADDRESSED_REPRESENTABLE_OPERATORS)
    initial_state_sensitive = sum(total[op] for op in INITIAL_STATE_SENSITIVE_OPERATORS)
    addressed_terminal_represented = query_count - initial_state_sensitive
    if query_count != affected + represented:
        raise ReaderIdentifiabilityAuditError("query classification support differs")
    return {
        "address_sensitive_query_count": affected,
        "address_sensitive_query_rate": affected / query_count,
        "addressed_terminal_reader_representable_count": (
            addressed_terminal_represented
        ),
        "addressed_terminal_reader_representable_rate": (
            addressed_terminal_represented / query_count
        ),
        "audit_sha256": digest.hexdigest(),
        "by_family": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(by_family.items())
        },
        "max_records_per_shard": max_records_per_shard,
        "initial_state_sensitive_query_count": initial_state_sensitive,
        "initial_state_sensitive_query_rate": (initial_state_sensitive / query_count),
        "operator_counts": dict(sorted(total.items())),
        "query_count": query_count,
        "reader_without_addresses_representable_count": represented,
        "reader_without_addresses_representable_rate": represented / query_count,
        "schema": REPORT_SCHEMA,
        "shard_count": len(shard_rows),
        "shard_rows": shard_rows,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--max-records-per-shard", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = audit(
        args.data_root,
        max_records_per_shard=args.max_records_per_shard,
    )
    payload = _canonical_bytes(report)
    if args.output is not None:
        if not args.output.parent.is_dir() or args.output.exists():
            raise ReaderIdentifiabilityAuditError("output path differs")
        args.output.write_bytes(payload)
    print(payload.decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADDRESS_SENSITIVE_OPERATORS",
    "INITIAL_STATE_SENSITIVE_OPERATORS",
    "ReaderIdentifiabilityAuditError",
    "UNADDRESSED_REPRESENTABLE_OPERATORS",
    "audit",
]
