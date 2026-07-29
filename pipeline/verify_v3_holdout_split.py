#!/usr/bin/env python3
"""Independently verify a Shohin v3 train/document/domain holdout split."""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator

from pipeline.build_general_source_review_packet import iter_document_ledger
from pipeline.materialize_v3_holdout_split import (
    RECEIPT_SCHEMA,
    SPLIT_NAMES,
    classify_document,
)
from pipeline.tokenize_shards import (
    DOCUMENT_LEDGER_NAME,
    canonical_payload_sha256,
    sha256_file,
)
from pipeline.verify_tokenized_shards import verify_manifest


HEX64 = re.compile(r"^[0-9a-f]{64}$")
IGNORED_LOCATION_FIELDS = {"shard", "token_start", "token_end"}


class HoldoutVerificationError(ValueError):
    """An immutable holdout split differs from its parent or receipt."""


def _load_receipt(path: Path) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HoldoutVerificationError("split receipt is unreadable") from exc
    if not isinstance(receipt, dict):
        raise HoldoutVerificationError("split receipt is not an object")
    claimed = receipt.get("payload_sha256")
    unsigned = dict(receipt)
    unsigned.pop("payload_sha256", None)
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or not isinstance(claimed, str)
        or HEX64.fullmatch(claimed) is None
        or canonical_payload_sha256(unsigned) != claimed
    ):
        raise HoldoutVerificationError("split receipt contract differs")
    return receipt


def _policy(receipt: dict[str, Any]) -> dict[str, Any]:
    policy = receipt.get("policy")
    if (
        not isinstance(policy, dict)
        or policy.get("algorithm") != "sha256_first_64_bits_mod_10000"
        or policy.get("assignment_order")
        != ["domain_validation", "document_validation", "train"]
        or policy.get("document_namespace")
        != "shohin-document-holdout-v1"
        or policy.get("domain_namespace") != "shohin-domain-holdout-v1"
        or policy.get("domain_missing_policy")
        != "never_domain_holdout_then_document_hash"
        or not isinstance(policy.get("seed"), str)
        or not policy["seed"]
        or not isinstance(
            policy.get("document_validation_basis_points"),
            int,
        )
        or not isinstance(policy.get("domain_validation_basis_points"), int)
    ):
        raise HoldoutVerificationError("split policy differs")
    return policy


def _normalized_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in IGNORED_LOCATION_FIELDS
    }


def _merged_split_rows(
    output_dir: Path,
) -> Iterator[tuple[str, dict[str, Any]]]:
    iterators = {
        name: iter(
            iter_document_ledger(
                output_dir / name / DOCUMENT_LEDGER_NAME
            )
        )
        for name in SPLIT_NAMES
    }
    heap: list[tuple[int, str, dict[str, Any]]] = []
    for name, iterator in iterators.items():
        try:
            row = next(iterator)
        except StopIteration as exc:
            raise HoldoutVerificationError(f"{name} split is empty") from exc
        heapq.heappush(heap, (int(row["source_row_index"]), name, row))
    while heap:
        _source_index, name, row = heapq.heappop(heap)
        yield name, row
        try:
            following = next(iterators[name])
        except StopIteration:
            continue
        heapq.heappush(
            heap,
            (int(following["source_row_index"]), name, following),
        )


def verify_holdout_split(
    *,
    output_dir: Path,
    source_selection_code: Path,
    selection_code: Path,
) -> dict[str, Any]:
    receipt_path = output_dir / "split_receipt.json"
    receipt = _load_receipt(receipt_path)
    expected_entries = {*SPLIT_NAMES, "split_receipt.json"}
    if (
        not output_dir.is_dir()
        or output_dir.is_symlink()
        or {path.name for path in output_dir.iterdir()} != expected_entries
    ):
        raise HoldoutVerificationError("split root contains unbound entries")
    source = receipt.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("path"), str):
        raise HoldoutVerificationError("source receipt differs")
    source_dir = Path(source["path"])
    source_verification = verify_manifest(
        source_dir,
        selection_code=source_selection_code,
        require_external_inputs=True,
    )
    source_manifest = json.loads((source_dir / "manifest.json").read_text())
    if (
        source_manifest.get("payload_sha256")
        != source.get("manifest_payload_sha256")
        or source_manifest.get("kept") != source.get("documents")
        or source_manifest.get("tokens") != source.get("tokens")
        or source_verification.get("document_ledger_verified") is not True
    ):
        raise HoldoutVerificationError("source binding differs")
    policy = _policy(receipt)
    split_records = receipt.get("splits")
    if (
        not isinstance(split_records, dict)
        or set(split_records) != set(SPLIT_NAMES)
    ):
        raise HoldoutVerificationError("split records differ")

    verifications: dict[str, dict[str, Any]] = {}
    for name in SPLIT_NAMES:
        record = split_records[name]
        manifest_path = output_dir / name / "manifest.json"
        if (
            not isinstance(record, dict)
            or record.get("path") != name
            or sha256_file(manifest_path) != record.get("manifest_sha256")
        ):
            raise HoldoutVerificationError(f"{name} receipt differs")
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("payload_sha256")
            != record.get("manifest_payload_sha256")
            or manifest.get("kept") != record.get("documents")
            or manifest.get("tokens") != record.get("tokens")
            or manifest.get("holdout_split", {}).get("name") != name
            or manifest.get("holdout_split", {}).get("policy") != policy
        ):
            raise HoldoutVerificationError(f"{name} manifest binding differs")
        verifications[name] = verify_manifest(
            output_dir / name,
            selection_code=selection_code,
            require_external_inputs=True,
        )

    merged = _merged_split_rows(output_dir)
    documents = tokens = 0
    split_documents = {name: 0 for name in SPLIT_NAMES}
    split_tokens = {name: 0 for name in SPLIT_NAMES}
    for source_row in iter_document_ledger(
        source_dir / DOCUMENT_LEDGER_NAME
    ):
        try:
            name, split_row = next(merged)
        except StopIteration as exc:
            raise HoldoutVerificationError(
                "split partition ends before source"
            ) from exc
        expected_name = classify_document(
            source_row,
            seed=policy["seed"],
            document_validation_bps=policy[
                "document_validation_basis_points"
            ],
            domain_validation_bps=policy[
                "domain_validation_basis_points"
            ],
        )
        if (
            name != expected_name
            or _normalized_row(source_row) != _normalized_row(split_row)
        ):
            raise HoldoutVerificationError(
                "split partition differs from source classification"
            )
        row_tokens = int(source_row["tokens"])
        documents += 1
        tokens += row_tokens
        split_documents[name] += 1
        split_tokens[name] += row_tokens
    try:
        next(merged)
    except StopIteration:
        pass
    else:
        raise HoldoutVerificationError("split partition extends past source")
    if (
        documents != source["documents"]
        or tokens != source["tokens"]
        or any(
            split_documents[name] != split_records[name]["documents"]
            or split_tokens[name] != split_records[name]["tokens"]
            for name in SPLIT_NAMES
        )
    ):
        raise HoldoutVerificationError("partition accounting differs")
    return {
        "schema": "shohin-v3-holdout-split-verification-v1",
        "receipt_payload_sha256": receipt["payload_sha256"],
        "documents": documents,
        "tokens": tokens,
        "split_documents": split_documents,
        "split_tokens": split_tokens,
        "source_verification": source_verification,
        "split_verifications": verifications,
        "partition_verified": True,
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-selection-code", type=Path, required=True)
    parser.add_argument(
        "--selection-code",
        type=Path,
        default=Path(__file__).with_name(
            "materialize_v3_holdout_split.py"
        ),
    )
    arguments = parser.parse_args(argv)
    result = verify_holdout_split(
        output_dir=arguments.output_dir,
        source_selection_code=arguments.source_selection_code,
        selection_code=arguments.selection_code,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
