#!/usr/bin/env python3
"""Split one verified v3 corpus into train, document, and domain holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping

from pipeline.build_general_source_review_packet import iter_document_ledger
from pipeline.materialize_cross_source_exact_residual import (
    _source_shard_bytes,
    _write_shard,
)
from pipeline.tokenize_shards import (
    DOCUMENT_LEDGER_NAME,
    DocumentLedgerWriter,
    canonical_payload_sha256,
    file_receipt,
    sha256_file,
)
from pipeline.verify_tokenized_shards import verify_manifest


SPLIT_SCHEMA = "shohin-v3-holdout-split-v1"
RECEIPT_SCHEMA = "shohin-v3-holdout-split-receipt-v1"
SPLIT_NAMES = ("train", "document_validation", "domain_validation")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MISSING_DOMAINS = {"", "<missing>", "none", "null"}


class HoldoutSplitError(ValueError):
    """The source corpus cannot produce a valid immutable holdout split."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HoldoutSplitError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise HoldoutSplitError(f"{label} is not an object")
    return value


def _hash_bucket(namespace: str, seed: str, value: str) -> int:
    material = f"{namespace}\x1f{seed}\x1f{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 10_000


def classify_document(
    row: Mapping[str, Any],
    *,
    seed: str,
    document_validation_bps: int,
    domain_validation_bps: int,
) -> str:
    """Classify a document with stable, order-independent hash thresholds."""
    identity = row.get("stable_identity_sha256")
    if (
        not isinstance(identity, str)
        or HEX64.fullmatch(identity) is None
        or not 0 <= document_validation_bps < 10_000
        or not 0 <= domain_validation_bps < 10_000
        or document_validation_bps + domain_validation_bps >= 10_000
        or not seed
    ):
        raise HoldoutSplitError("holdout classification contract differs")
    domain_value = row.get("domain")
    domain = (
        str(domain_value).strip().lower() if domain_value is not None else ""
    )
    if (
        domain not in MISSING_DOMAINS
        and _hash_bucket("shohin-domain-holdout-v1", seed, domain)
        < domain_validation_bps
    ):
        return "domain_validation"
    if (
        _hash_bucket("shohin-document-holdout-v1", seed, identity)
        < document_validation_bps
    ):
        return "document_validation"
    return "train"


class _SplitWriter:
    def __init__(
        self,
        root: Path,
        *,
        split_name: str,
        shard_tokens: int,
    ) -> None:
        self.split_name = split_name
        self.output_dir = root / split_name
        self.output_dir.mkdir()
        self.shard_tokens = shard_tokens
        self.ledger = DocumentLedgerWriter(
            self.output_dir / DOCUMENT_LEDGER_NAME
        )
        self.shard_records: list[dict[str, Any]] = []
        self.payload = bytearray()
        self.shard_index = 0
        self.documents = 0
        self.tokens = 0

    def write(self, row: Mapping[str, Any], document_payload: bytes) -> None:
        tokens = int(row["tokens"])
        if (
            len(document_payload) != tokens * 2
            or hashlib.sha256(document_payload).hexdigest()
            != row["token_sha256"]
        ):
            raise HoldoutSplitError("source document token span differs")
        token_start = len(self.payload) // 2
        self.payload.extend(document_payload)
        output_row = dict(row)
        output_row.update(
            {
                "shard": f"shard_{self.shard_index:05d}.u16.zst",
                "token_start": token_start,
                "token_end": token_start + tokens,
            }
        )
        self.ledger.write(output_row)
        self.documents += 1
        self.tokens += tokens
        if len(self.payload) // 2 >= self.shard_tokens:
            self._flush()

    def _flush(self) -> None:
        if not self.payload:
            return
        self.shard_records.append(
            _write_shard(
                self.output_dir,
                index=self.shard_index,
                payload=self.payload,
            )
        )
        self.payload = bytearray()
        self.shard_index += 1

    def close(
        self,
        *,
        source_manifest: Mapping[str, Any],
        source_dir: Path,
        source_verification: Mapping[str, Any],
        source_selection_receipt: Mapping[str, Any],
        selection_code: Path,
        policy: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._flush()
        ledger_receipt = self.ledger.close()
        if (
            self.documents < 1
            or self.tokens < 1
            or ledger_receipt["rows"] != self.documents
            or ledger_receipt["tokens"] != self.tokens
            or sum(item["tokens"] for item in self.shard_records)
            != self.tokens
        ):
            raise HoldoutSplitError(f"{self.split_name} split is empty or inconsistent")
        manifest = {
            key: value
            for key, value in source_manifest.items()
            if key
            not in {
                "payload_sha256",
                "selection_code_sha256",
                "tokens",
                "shards",
                "shard_files",
                "document_ledger",
                "kept",
            }
        }
        manifest.update(
            {
                "schema": "shohin-tokenized-shards-v3",
                "selection_code_sha256": sha256_file(selection_code),
                "tokens": self.tokens,
                "shards": len(self.shard_records),
                "shard_files": self.shard_records,
                "document_ledger": ledger_receipt,
                "kept": self.documents,
                "holdout_split": {
                    "schema": SPLIT_SCHEMA,
                    "name": self.split_name,
                    "policy": dict(policy),
                    "source_path": str(source_dir.resolve()),
                    "source_manifest_payload_sha256": source_manifest[
                        "payload_sha256"
                    ],
                    "source_selection_code": dict(source_selection_receipt),
                    "source_verification": dict(source_verification),
                },
            }
        )
        manifest["payload_sha256"] = canonical_payload_sha256(manifest)
        manifest_path = self.output_dir / "manifest.json"
        with manifest_path.open("x") as destination:
            json.dump(manifest, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        verification = verify_manifest(
            self.output_dir,
            selection_code=selection_code,
            require_external_inputs=True,
        )
        if (
            verification["tokens"] != self.tokens
            or verification["document_rows"] != self.documents
        ):
            raise HoldoutSplitError(
                f"{self.split_name} published verification differs"
            )
        return manifest, verification


def materialize_holdout_split(
    *,
    source_dir: Path,
    source_selection_code: Path,
    selection_code: Path,
    output_dir: Path,
    seed: str,
    document_validation_bps: int = 100,
    domain_validation_bps: int = 100,
    shard_tokens: int = 100_000_000,
) -> dict[str, Any]:
    if (
        not seed
        or shard_tokens < 1
        or not 0 <= document_validation_bps < 10_000
        or not 0 <= domain_validation_bps < 10_000
        or document_validation_bps + domain_validation_bps >= 10_000
        or not source_selection_code.is_file()
        or source_selection_code.is_symlink()
        or not selection_code.is_file()
        or selection_code.is_symlink()
    ):
        raise HoldoutSplitError("holdout split arguments differ")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing existing output: {output_dir}")
    source_verification = verify_manifest(
        source_dir,
        selection_code=source_selection_code,
        require_external_inputs=True,
    )
    source_manifest = _load_json(source_dir / "manifest.json", "source manifest")
    if (
        source_manifest.get("schema") != "shohin-tokenized-shards-v3"
        or not source_verification.get("document_ledger_verified")
        or source_manifest.get("filters", {}).get("exact_dedup") is not True
    ):
        raise HoldoutSplitError(
            "source is not a verified exact-deduplicated v3 corpus"
        )
    source_selection_receipt = file_receipt(source_selection_code)
    if (
        source_selection_receipt["sha256"]
        != source_manifest.get("selection_code_sha256")
    ):
        raise HoldoutSplitError("source selection code differs from manifest")

    policy = {
        "algorithm": "sha256_first_64_bits_mod_10000",
        "assignment_order": [
            "domain_validation",
            "document_validation",
            "train",
        ],
        "document_namespace": "shohin-document-holdout-v1",
        "document_validation_basis_points": document_validation_bps,
        "domain_missing_policy": "never_domain_holdout_then_document_hash",
        "domain_namespace": "shohin-domain-holdout-v1",
        "domain_validation_basis_points": domain_validation_bps,
        "seed": seed,
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.partial-",
            dir=output_dir.parent,
        )
    )
    try:
        writers = {
            name: _SplitWriter(
                staging,
                split_name=name,
                shard_tokens=shard_tokens,
            )
            for name in SPLIT_NAMES
        }
        source_shard: str | None = None
        source_payload = b""
        source_documents = source_tokens = 0
        for row in iter_document_ledger(source_dir / DOCUMENT_LEDGER_NAME):
            split_name = classify_document(
                row,
                seed=seed,
                document_validation_bps=document_validation_bps,
                domain_validation_bps=domain_validation_bps,
            )
            if row["shard"] != source_shard:
                source_shard = row["shard"]
                source_payload = _source_shard_bytes(source_dir, source_shard)
            start = int(row["token_start"]) * 2
            end = int(row["token_end"]) * 2
            writers[split_name].write(row, source_payload[start:end])
            source_documents += 1
            source_tokens += int(row["tokens"])
        if (
            source_documents != source_manifest.get("kept")
            or source_tokens != source_manifest.get("tokens")
        ):
            raise HoldoutSplitError("source accounting differs")

        manifests: dict[str, dict[str, Any]] = {}
        verifications: dict[str, dict[str, Any]] = {}
        for name in SPLIT_NAMES:
            manifests[name], verifications[name] = writers[name].close(
                source_manifest=source_manifest,
                source_dir=source_dir,
                source_verification=source_verification,
                source_selection_receipt=source_selection_receipt,
                selection_code=selection_code,
                policy=policy,
            )
        if (
            sum(writers[name].documents for name in SPLIT_NAMES)
            != source_documents
            or sum(writers[name].tokens for name in SPLIT_NAMES)
            != source_tokens
        ):
            raise HoldoutSplitError("split accounting differs")

        receipt = {
            "schema": RECEIPT_SCHEMA,
            "source": {
                "path": str(source_dir.resolve()),
                "manifest_payload_sha256": source_manifest["payload_sha256"],
                "selection_code": source_selection_receipt,
                "documents": source_documents,
                "tokens": source_tokens,
            },
            "policy": policy,
            "partition_invariant": (
                "every_source_ledger_row_assigned_exactly_once_in_source_order"
            ),
            "splits": {
                name: {
                    "path": name,
                    "manifest_payload_sha256": manifests[name][
                        "payload_sha256"
                    ],
                    "manifest_sha256": sha256_file(
                        staging / name / "manifest.json"
                    ),
                    "documents": writers[name].documents,
                    "tokens": writers[name].tokens,
                    "verification": verifications[name],
                }
                for name in SPLIT_NAMES
            },
        }
        receipt["payload_sha256"] = canonical_payload_sha256(receipt)
        receipt_path = staging / "split_receipt.json"
        with receipt_path.open("x") as destination:
            json.dump(receipt, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(staging, output_dir)
        return receipt
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-selection-code", type=Path, required=True)
    parser.add_argument("--selection-code", type=Path, default=Path(__file__))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--document-validation-bps", type=int, default=100)
    parser.add_argument("--domain-validation-bps", type=int, default=100)
    parser.add_argument("--shard-tokens", type=int, default=100_000_000)
    arguments = parser.parse_args(argv)
    result = materialize_holdout_split(
        source_dir=arguments.source_dir,
        source_selection_code=arguments.source_selection_code,
        selection_code=arguments.selection_code,
        output_dir=arguments.output_dir,
        seed=arguments.seed,
        document_validation_bps=arguments.document_validation_bps,
        domain_validation_bps=arguments.domain_validation_bps,
        shard_tokens=arguments.shard_tokens,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
