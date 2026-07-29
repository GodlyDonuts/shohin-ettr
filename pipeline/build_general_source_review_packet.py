#!/usr/bin/env python3
"""Build a private semantic-review packet from an exact v3 tokenized corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import heapq
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import zstandard as zstd
from datasets import load_dataset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.profile_general_source import (  # noqa: E402
    review_excerpt,
    review_metadata,
    text_metrics,
)
from pipeline.tokenize_shards import (  # noqa: E402
    DOCUMENT_LEDGER_NAME,
    DOCUMENT_LEDGER_SCHEMA,
    canonical_payload_sha256,
    exact_text_hash,
    local_input_format,
    sha256_file,
    stable_document_identity,
)
from pipeline.verify_tokenized_shards import verify_manifest  # noqa: E402


PACKET_SCHEMA = "shohin-private-selected-source-review-v1"
RECEIPT_SCHEMA = "shohin-selected-source-review-receipt-v1"


class ReviewPacketError(ValueError):
    """The selected corpus cannot produce an exact private review packet."""


def _length_bucket(tokens: int) -> str:
    if tokens <= 2_048:
        return "tokens_00000_02048"
    if tokens <= 8_192:
        return "tokens_02049_08192"
    if tokens <= 32_768:
        return "tokens_08193_32768"
    return "tokens_32769_plus"


def _priority(identity: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"shohin-review-v1\x1f{identity}".encode("ascii")).digest(),
        "big",
    )


def iter_document_ledger(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("rb") as source:
        with zstd.ZstdDecompressor().stream_reader(source) as reader:
            with io.TextIOWrapper(reader, encoding="ascii") as text:
                for line_number, line in enumerate(text, 1):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ReviewPacketError(
                            f"document ledger row {line_number} is malformed"
                        ) from exc
                    if (
                        not isinstance(row, dict)
                        or row.get("schema") != DOCUMENT_LEDGER_SCHEMA
                    ):
                        raise ReviewPacketError("document ledger schema differs")
                    yield row


def select_review_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    count: int,
) -> list[dict[str, Any]]:
    if count < 1:
        raise ReviewPacketError("review row count must be positive")
    # Retain the best `count` hash priorities in every license/length stratum.
    # This bounds memory while keeping rare strata eligible.
    heaps: dict[tuple[str, str], list[tuple[int, str, dict[str, Any]]]] = (
        defaultdict(list)
    )
    total = 0
    for raw in rows:
        row = dict(raw)
        identity = row.get("stable_identity_sha256")
        tokens = row.get("tokens")
        if not isinstance(identity, str) or not isinstance(tokens, int):
            raise ReviewPacketError("document ledger review fields differ")
        stratum = (
            str(row.get("allowed_value") or "<missing>"),
            _length_bucket(tokens),
        )
        priority = _priority(identity)
        item = (-priority, identity, row)
        heap = heaps[stratum]
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, item)
        total += 1
    if total < count:
        raise ReviewPacketError(
            f"requested {count} review rows from only {total} documents"
        )

    candidates: dict[str, tuple[int, tuple[str, str], dict[str, Any]]] = {}
    for stratum, heap in heaps.items():
        for negative, identity, row in heap:
            candidates[identity] = (-negative, stratum, row)
    ordered = sorted(
        candidates.values(),
        key=lambda item: (item[0], item[2]["stable_identity_sha256"]),
    )

    selected: list[dict[str, Any]] = []
    selected_identities: set[str] = set()
    represented: set[tuple[str, str]] = set()
    for _priority_value, stratum, row in ordered:
        if stratum in represented:
            continue
        selected.append(row)
        selected_identities.add(row["stable_identity_sha256"])
        represented.add(stratum)
        if len(selected) == count:
            return selected

    domain_cap = max(5, math.ceil(count / 100))
    domains: Counter[str] = Counter(str(row.get("domain")) for row in selected)
    for _priority_value, _stratum, row in ordered:
        identity = row["stable_identity_sha256"]
        domain = str(row.get("domain"))
        if identity in selected_identities or domains[domain] >= domain_cap:
            continue
        selected.append(row)
        selected_identities.add(identity)
        domains[domain] += 1
        if len(selected) == count:
            return selected

    # Very small-domain corpora may not satisfy the diversity cap. Fill the
    # packet deterministically rather than silently shrinking it.
    for _priority_value, _stratum, row in ordered:
        identity = row["stable_identity_sha256"]
        if identity in selected_identities:
            continue
        selected.append(row)
        selected_identities.add(identity)
        if len(selected) == count:
            return selected
    raise ReviewPacketError("review candidate pool cannot satisfy row count")


def _iter_json_rows(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ReviewPacketError(
                        f"{path}:{line_number}: malformed source JSON"
                    ) from exc
                if not isinstance(row, dict):
                    raise ReviewPacketError(
                        f"{path}:{line_number}: source row is not an object"
                    )
                yield row


def iter_source_rows(
    paths: Iterable[Path],
    *,
    dataset_loader=load_dataset,
) -> Iterable[dict[str, Any]]:
    """Replay physical source rows with the tokenizer's format and ordering."""
    resolved = tuple(sorted(Path(path).resolve() for path in paths))
    source_format = local_input_format([str(path) for path in resolved])
    if source_format == "json":
        yield from _iter_json_rows(resolved)
        return
    dataset = dataset_loader(
        "parquet",
        data_files=[str(path) for path in resolved],
        split="train",
        streaming=True,
    )
    for row_number, row in enumerate(dataset, 1):
        if not isinstance(row, Mapping):
            raise ReviewPacketError(
                f"Parquet source row {row_number} is not an object"
            )
        yield dict(row)


def _row_text(row: Mapping[str, Any], filters: Mapping[str, Any]) -> str:
    text_columns = filters.get("text_cols")
    if text_columns:
        if not isinstance(text_columns, list):
            raise ReviewPacketError("text column list differs")
        return "\n\n".join(
            str(row.get(column) or "")
            for column in text_columns
            if row.get(column)
        )
    text_column = filters.get("text_col")
    if not isinstance(text_column, str) or not text_column:
        raise ReviewPacketError("text column differs")
    value = row.get(text_column)
    return value if isinstance(value, str) else ""


def materialize_review_rows(
    source_rows: Iterable[Mapping[str, Any]],
    selected: Iterable[Mapping[str, Any]],
    *,
    dataset: str,
    config: str,
    filters: Mapping[str, Any],
    max_review_chars: int,
) -> list[dict[str, Any]]:
    if max_review_chars < 1:
        raise ReviewPacketError("review excerpt limit must be positive")
    selected_by_index = {
        int(row["source_row_index"]): dict(row) for row in selected
    }
    if len(selected_by_index) == 0:
        raise ReviewPacketError("selected review rows are empty")
    materialized: list[dict[str, Any]] = []
    for source_index, source_row in enumerate(source_rows):
        ledger = selected_by_index.get(source_index)
        if ledger is None:
            continue
        text = _row_text(source_row, filters)
        document_sha256 = exact_text_hash(text).hex()
        identity = stable_document_identity(source_row, document_sha256)
        if (
            document_sha256 != ledger["document_sha256"]
            or identity != ledger["stable_identity_sha256"]
        ):
            raise ReviewPacketError(
                "source document differs from selected ledger identity"
            )
        excerpt, truncated = review_excerpt(text, max_review_chars)
        materialized.append(
            {
                "schema": PACKET_SCHEMA,
                "admission_status": "private_human_review_only_not_training_data",
                "dataset": dataset,
                "config": config,
                "stable_identity_sha256": identity,
                "document_sha256": document_sha256,
                "metadata": review_metadata(source_row),
                "metrics": text_metrics(text),
                "selection": ledger,
                "review_text": excerpt,
                "review_text_truncated": truncated,
            }
        )
        if len(materialized) == len(selected_by_index):
            break
    if len(materialized) != len(selected_by_index):
        missing = len(selected_by_index) - len(materialized)
        raise ReviewPacketError(f"{missing} selected source documents were not found")
    return sorted(
        materialized,
        key=lambda row: row["stable_identity_sha256"],
    )


def _write_packet(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise ReviewPacketError(f"refusing existing private packet: {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ReviewPacketError(f"refusing existing review receipt: {path}")
    material = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as output:
        output.write(material)
        output.flush()
        os.fsync(output.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--selection-code", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1_000)
    parser.add_argument("--max-review-chars", type=int, default=12_000)
    parser.add_argument("--private-out", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    args = parser.parse_args()

    verification = verify_manifest(
        args.shard_dir,
        selection_code=args.selection_code,
        require_external_inputs=True,
    )
    manifest_path = args.shard_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "shohin-tokenized-shards-v3":
        raise ReviewPacketError("semantic review requires a v3 corpus")
    ledger_record = manifest.get("document_ledger")
    if not isinstance(ledger_record, dict):
        raise ReviewPacketError("v3 document ledger receipt is absent")
    ledger_path = args.shard_dir / DOCUMENT_LEDGER_NAME
    selected = select_review_rows(
        iter_document_ledger(ledger_path),
        count=args.rows,
    )
    sources = manifest.get("source_files")
    if not isinstance(sources, list) or not sources:
        raise ReviewPacketError(
            "review packet requires physical hash-bound source files"
        )
    source_paths = [Path(str(record["path"])) for record in sources]
    filters = manifest.get("filters")
    if not isinstance(filters, dict):
        raise ReviewPacketError("selection filters are absent")
    rows = materialize_review_rows(
        iter_source_rows(source_paths),
        selected,
        dataset=str(manifest.get("dataset")),
        config=str(manifest.get("config")),
        filters=filters,
        max_review_chars=args.max_review_chars,
    )
    _write_packet(args.private_out, rows)
    packet_sha256 = sha256_file(args.private_out)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "contains_document_text": False,
        "dataset": manifest.get("dataset"),
        "config": manifest.get("config"),
        "manifest_payload_sha256": manifest.get("payload_sha256"),
        "manifest_recomputed_payload_sha256": canonical_payload_sha256(
            {
                key: value
                for key, value in manifest.items()
                if key != "payload_sha256"
            }
        ),
        "document_ledger_sha256": ledger_record.get("sha256"),
        "private_packet_basename": args.private_out.name,
        "private_packet_bytes": args.private_out.stat().st_size,
        "private_packet_sha256": packet_sha256,
        "review_rows": len(rows),
        "selection_rule": (
            "lowest stable-identity hash priorities per license/length "
            "stratum, then domain-capped deterministic fill"
        ),
        "license_counts": dict(
            Counter(str(row["selection"]["allowed_value"]) for row in rows)
        ),
        "length_bucket_counts": dict(
            Counter(_length_bucket(row["selection"]["tokens"]) for row in rows)
        ),
        "unique_domains": len(
            {str(row["selection"]["domain"]) for row in rows}
        ),
        "verification": verification,
    }
    _write_receipt(args.receipt_out, receipt)
    print(
        json.dumps(
            {
                "packet_sha256": packet_sha256,
                "receipt": str(args.receipt_out),
                "rows": len(rows),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
