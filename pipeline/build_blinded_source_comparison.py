#!/usr/bin/env python3
"""Build a reviewer-blind, length-matched comparison from private packets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


PACKET_SCHEMA = "shohin-private-selected-source-review-v1"
BLINDED_SCHEMA = "shohin-private-blinded-source-comparison-v1"
KEY_SCHEMA = "shohin-private-blinded-source-comparison-key-v1"
RECEIPT_SCHEMA = "shohin-blinded-source-comparison-receipt-v1"
NAMESPACE = "shohin-blinded-source-comparison-v1"
ARM_PATTERN = re.compile(r"([a-z0-9_-]+)=(/[^:]+)::(/.+)")


class BlindedComparisonError(ValueError):
    """The source packets cannot support a reviewer-blind comparison."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _length_bucket(tokens: int) -> str:
    if tokens <= 2_048:
        return "tokens_00000_02048"
    if tokens <= 8_192:
        return "tokens_02049_08192"
    if tokens <= 32_768:
        return "tokens_08193_32768"
    return "tokens_32769_plus"


def _hash_priority(*parts: str) -> str:
    return hashlib.sha256("\x1f".join((NAMESPACE, *parts)).encode("ascii")).hexdigest()


def _load_packet(
    packet_path: Path,
    receipt_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    packet_sha256 = sha256_file(packet_path)
    if (
        receipt.get("schema") != "shohin-selected-source-review-receipt-v1"
        or receipt.get("private_packet_sha256") != packet_sha256
        or receipt.get("private_packet_bytes") != packet_path.stat().st_size
    ):
        raise BlindedComparisonError("source review receipt does not bind packet")

    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    with packet_path.open(encoding="ascii") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BlindedComparisonError(
                    f"{packet_path}:{line_number}: malformed JSON"
                ) from exc
            identity = row.get("stable_identity_sha256")
            selection = row.get("selection")
            text = row.get("review_text")
            if (
                not isinstance(row, dict)
                or row.get("schema") != PACKET_SCHEMA
                or not isinstance(identity, str)
                or len(identity) != 64
                or identity in identities
                or not isinstance(selection, dict)
                or not isinstance(selection.get("tokens"), int)
                or selection["tokens"] < 1
                or not isinstance(text, str)
                or not text
            ):
                raise BlindedComparisonError("source review packet fields differ")
            identities.add(identity)
            rows.append(row)
    if len(rows) != receipt.get("review_rows"):
        raise BlindedComparisonError("source review row count differs")
    return rows, receipt


def _balanced_bucket_quotas(
    rows_by_arm: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    rows_per_arm: int,
) -> dict[str, int]:
    capacities: dict[str, int] = {}
    counts_by_arm = {
        arm: Counter(
            _length_bucket(int(row["selection"]["tokens"])) for row in rows
        )
        for arm, rows in rows_by_arm.items()
    }
    buckets = sorted(
        set.intersection(*(set(counts) for counts in counts_by_arm.values()))
    )
    for bucket in buckets:
        capacities[bucket] = min(
            counts[bucket] for counts in counts_by_arm.values()
        )
    capacities = {
        bucket: capacity
        for bucket, capacity in capacities.items()
        if capacity > 0
    }
    if sum(capacities.values()) < rows_per_arm:
        raise BlindedComparisonError("matched length strata are too small")
    if rows_per_arm < len(capacities):
        raise BlindedComparisonError("row target cannot represent every length stratum")

    quotas = {bucket: 1 for bucket in capacities}
    total_capacity = sum(capacities.values())
    while sum(quotas.values()) < rows_per_arm:
        available = [
            bucket
            for bucket in capacities
            if quotas[bucket] < capacities[bucket]
        ]
        if not available:
            raise BlindedComparisonError("matched length allocation exhausted")
        bucket = max(
            available,
            key=lambda value: (
                capacities[value] / total_capacity
                - quotas[value] / rows_per_arm,
                capacities[value] - quotas[value],
                value,
            ),
        )
        quotas[bucket] += 1
    return dict(sorted(quotas.items()))


def build_comparison(
    arms: Mapping[str, tuple[Path, Path]],
    *,
    rows_per_arm: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if len(arms) < 2:
        raise BlindedComparisonError("at least two arms are required")
    if rows_per_arm < 100:
        raise BlindedComparisonError("human comparison requires at least 100 rows per arm")

    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    packet_paths: dict[str, Path] = {}
    receipt_paths: dict[str, Path] = {}
    dataset_config: set[tuple[Any, Any]] = set()
    all_identities: set[str] = set()
    for arm, (packet_path, receipt_path) in sorted(arms.items()):
        rows, receipt = _load_packet(packet_path, receipt_path)
        identities = {str(row["stable_identity_sha256"]) for row in rows}
        if all_identities & identities:
            raise BlindedComparisonError("source arms share document identities")
        all_identities.update(identities)
        rows_by_arm[arm] = rows
        receipts[arm] = receipt
        packet_paths[arm] = packet_path
        receipt_paths[arm] = receipt_path
        dataset_config.add((receipt.get("dataset"), receipt.get("config")))
    if len(dataset_config) != 1:
        raise BlindedComparisonError("source arms mix datasets or configurations")

    quotas = _balanced_bucket_quotas(rows_by_arm, rows_per_arm=rows_per_arm)
    blinded_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    selected_counts: dict[str, Counter[str]] = {}
    for arm, rows in sorted(rows_by_arm.items()):
        by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_bucket[_length_bucket(int(row["selection"]["tokens"]))].append(row)
        selected_counts[arm] = Counter()
        for bucket, quota in quotas.items():
            selected = sorted(
                by_bucket[bucket],
                key=lambda row: (
                    _hash_priority(
                        "select",
                        arm,
                        str(row["stable_identity_sha256"]),
                    ),
                    row["stable_identity_sha256"],
                ),
            )[:quota]
            if len(selected) != quota:
                raise BlindedComparisonError("matched source selection is incomplete")
            for row in selected:
                identity = str(row["stable_identity_sha256"])
                blind_id = _hash_priority("blind-id", identity)
                blinded_rows.append(
                    {
                        "schema": BLINDED_SCHEMA,
                        "blind_id": blind_id,
                        "review_text": row["review_text"],
                        "review_text_truncated": bool(
                            row.get("review_text_truncated")
                        ),
                    }
                )
                key_rows.append(
                    {
                        "schema": KEY_SCHEMA,
                        "blind_id": blind_id,
                        "arm": arm,
                        "stable_identity_sha256": identity,
                        "document_sha256": row["document_sha256"],
                        "length_bucket": bucket,
                        "source_packet_sha256": receipts[arm][
                            "private_packet_sha256"
                        ],
                    }
                )
                selected_counts[arm][bucket] += 1
    blinded_rows.sort(key=lambda row: _hash_priority("order", row["blind_id"]))
    key_rows.sort(key=lambda row: row["blind_id"])
    key = {
        "schema": KEY_SCHEMA,
        "contains_document_text": False,
        "rows": key_rows,
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "private_human_review_only_not_training_admission",
        "contains_document_text": False,
        "dataset": next(iter(dataset_config))[0],
        "config": next(iter(dataset_config))[1],
        "rows_per_arm": rows_per_arm,
        "total_rows": len(blinded_rows),
        "matched_length_bucket_quotas": quotas,
        "selected_counts": {
            arm: dict(sorted(counts.items()))
            for arm, counts in sorted(selected_counts.items())
        },
        "sources": {
            arm: {
                "packet_basename": packet_paths[arm].name,
                "packet_sha256": receipts[arm]["private_packet_sha256"],
                "receipt_basename": receipt_paths[arm].name,
                "receipt_sha256": sha256_file(receipt_paths[arm]),
            }
            for arm in sorted(arms)
        },
        "selection_rule": (
            "minimum shared length-stratum capacity, proportional deterministic "
            "allocation with every shared stratum represented, then lowest "
            "namespace-bound identity priorities per arm and stratum"
        ),
    }
    return blinded_rows, key, receipt


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], mode: int) -> None:
    if path.exists() or path.is_symlink():
        raise BlindedComparisonError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="ascii") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _write_json(path: Path, payload: Mapping[str, Any], mode: int) -> None:
    if path.exists() or path.is_symlink():
        raise BlindedComparisonError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="ascii") as output:
        json.dump(payload, output, indent=2, sort_keys=True, ensure_ascii=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def _parse_arm(value: str) -> tuple[str, tuple[Path, Path]]:
    match = ARM_PATTERN.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError(
            "arm must be lowercase-name=/absolute/packet::/absolute/receipt"
        )
    return match.group(1), (Path(match.group(2)), Path(match.group(3)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", type=_parse_arm, required=True)
    parser.add_argument("--rows-per-arm", type=int, default=100)
    parser.add_argument("--private-out", type=Path, required=True)
    parser.add_argument("--key-out", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    args = parser.parse_args()
    arms = dict(args.arm)
    if len(arms) != len(args.arm):
        raise BlindedComparisonError("duplicate arm name")
    blinded_rows, key, receipt = build_comparison(
        arms,
        rows_per_arm=args.rows_per_arm,
    )
    _write_jsonl(args.private_out, blinded_rows, 0o600)
    _write_json(args.key_out, key, 0o600)
    receipt = dict(receipt)
    receipt["private_packet_basename"] = args.private_out.name
    receipt["private_packet_bytes"] = args.private_out.stat().st_size
    receipt["private_packet_sha256"] = sha256_file(args.private_out)
    receipt["private_key_basename"] = args.key_out.name
    receipt["private_key_bytes"] = args.key_out.stat().st_size
    receipt["private_key_sha256"] = sha256_file(args.key_out)
    _write_json(args.receipt_out, receipt, 0o444)
    print(
        json.dumps(
            {
                "rows": receipt["total_rows"],
                "private_packet_sha256": receipt["private_packet_sha256"],
                "private_key_sha256": receipt["private_key_sha256"],
                "receipt": str(args.receipt_out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
