#!/usr/bin/env python3
"""Verify every byte bound by a Shohin tokenized-shard manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Any

import zstandard as zstd

from pipeline.tokenize_shards import (
    DOCUMENT_LEDGER_NAME,
    DOCUMENT_LEDGER_SCHEMA,
    canonical_payload_sha256,
)


SHARD_NAME = re.compile(r"shard_[0-9]{5}\.u16\.zst")
HEX64 = re.compile(r"[0-9a-f]{64}")


class ShardVerificationError(ValueError):
    """A tokenized shard corpus differs from its immutable manifest."""


def _regular_identity(path: Path, label: str) -> tuple[int, ...]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ShardVerificationError(f"{label} cannot be inspected: {path}") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise ShardVerificationError(
            f"{label} is not a single-link regular non-symlink file: {path}"
        )
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def _stable_sha256_and_tokens(
    path: Path,
    label: str,
    *,
    count_tokens: bool,
    expected_sha256: object | None = None,
    expected_bytes: object | None = None,
) -> tuple[str, int | None, int]:
    before = _regular_identity(path, label)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if expected_bytes is not None and before[2] != expected_bytes:
        raise ShardVerificationError(f"{label} byte count differs: {path}")
    measured_digest = digest.hexdigest()
    if expected_sha256 is not None and measured_digest != expected_sha256:
        raise ShardVerificationError(f"{label} SHA-256 differs: {path}")
    tokens = _decompressed_uint16_tokens(path) if count_tokens else None
    after = _regular_identity(path, label)
    if before != after:
        raise ShardVerificationError(f"{label} changed while being verified: {path}")
    return measured_digest, tokens, before[2]


def _decompressed_uint16_tokens(path: Path) -> int:
    raw_bytes = 0
    try:
        with path.open("rb") as source:
            with zstd.ZstdDecompressor().stream_reader(source) as reader:
                for block in iter(lambda: reader.read(8 * 1024 * 1024), b""):
                    raw_bytes += len(block)
    except (OSError, zstd.ZstdError) as exc:
        raise ShardVerificationError(
            f"compressed shard cannot be decoded: {path}"
        ) from exc
    if raw_bytes % 2:
        raise ShardVerificationError(f"decompressed shard has odd byte length: {path}")
    return raw_bytes // 2


def _decompressed_bytes(path: Path) -> bytes:
    try:
        with path.open("rb") as source:
            with zstd.ZstdDecompressor().stream_reader(source) as reader:
                return reader.read()
    except (OSError, zstd.ZstdError) as exc:
        raise ShardVerificationError(
            f"compressed shard cannot be decoded: {path}"
        ) from exc


def _verify_external_file(record: dict[str, Any], label: str) -> None:
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ShardVerificationError(f"{label} path is missing")
    path = Path(path_value)
    digest, _tokens, size = _stable_sha256_and_tokens(
        path,
        label,
        count_tokens=False,
        expected_sha256=record.get("sha256"),
        expected_bytes=record.get("bytes"),
    )
    if size != record.get("bytes"):
        raise ShardVerificationError(f"{label} byte count differs: {path}")
    if digest != record.get("sha256"):
        raise ShardVerificationError(f"{label} SHA-256 differs: {path}")


def _verify_document_ledger(
    shard_dir: Path,
    record: dict[str, Any],
    shard_records: list[dict[str, Any]],
    *,
    kept: object,
    document_policy: dict[str, Any] | None,
) -> dict[str, int]:
    if (
        record.get("path") != DOCUMENT_LEDGER_NAME
        or record.get("schema") != DOCUMENT_LEDGER_SCHEMA
        or record.get("contains_document_text") is not False
    ):
        raise ShardVerificationError("document ledger receipt differs")
    path = shard_dir / DOCUMENT_LEDGER_NAME
    _digest, _tokens, _bytes = _stable_sha256_and_tokens(
        path,
        "document ledger",
        count_tokens=False,
        expected_sha256=record.get("sha256"),
        expected_bytes=record.get("bytes"),
    )
    shard_tokens = {
        str(shard["path"]): int(shard["tokens"]) for shard in shard_records
    }
    offsets = {name: 0 for name in shard_tokens}
    rows = 0
    tokens = 0
    policy_tier_counts: dict[str, int] = {}
    expected_policy_tiers: set[str] | None = None
    expected_policy_counts: dict[str, int] | None = None
    if document_policy is not None:
        allowed_tiers = document_policy.get("allowed_tiers")
        retained_tiers = document_policy.get("retained_tiers")
        if (
            not isinstance(allowed_tiers, list)
            or not allowed_tiers
            or any(
                not isinstance(tier, str) or not tier
                for tier in allowed_tiers
            )
            or len(allowed_tiers) != len(set(allowed_tiers))
            or not isinstance(retained_tiers, dict)
            or any(
                not isinstance(tier, str)
                or not tier
                or type(count) is not int
                or count < 0
                for tier, count in retained_tiers.items()
            )
        ):
            raise ShardVerificationError(
                "document-policy ledger contract differs"
            )
        expected_policy_tiers = set(allowed_tiers)
        if not set(retained_tiers).issubset(expected_policy_tiers):
            raise ShardVerificationError(
                "document-policy retained tiers are not allowed"
            )
        expected_policy_counts = dict(retained_tiers)
    last_source_row = -1
    last_shard_index = -1
    current_shard: str | None = None
    current_shard_bytes = b""
    try:
        with path.open("rb") as source:
            with zstd.ZstdDecompressor().stream_reader(source) as reader:
                pending = b""
                while True:
                    block = reader.read(8 * 1024 * 1024)
                    if not block:
                        break
                    pending += block
                    lines = pending.split(b"\n")
                    pending = lines.pop()
                    for line in lines:
                        if not line:
                            raise ShardVerificationError(
                                "document ledger contains an empty row"
                            )
                        try:
                            value = json.loads(line)
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            raise ShardVerificationError(
                                "document ledger row is malformed"
                            ) from exc
                        if not isinstance(value, dict):
                            raise ShardVerificationError(
                                "document ledger row is not an object"
                            )
                        required = {
                            "allowed_value",
                            "chars",
                            "document_sha256",
                            "domain",
                            "schema",
                            "shard",
                            "source_row_index",
                            "stable_identity_sha256",
                            "token_end",
                            "token_sha256",
                            "token_start",
                            "tokens",
                        }
                        policy_bound_fields = required | {
                            "document_policy_tier"
                        }
                        value_fields = frozenset(value)
                        if value_fields not in {
                            frozenset(required),
                            frozenset(policy_bound_fields),
                        }:
                            raise ShardVerificationError(
                                "document ledger row fields differ"
                            )
                        has_policy_tier = "document_policy_tier" in value
                        policy_tier = value.get("document_policy_tier")
                        if expected_policy_tiers is None:
                            if has_policy_tier and policy_tier is not None:
                                raise ShardVerificationError(
                                    "unbound document-policy tier is present"
                                )
                        elif (
                            not has_policy_tier
                            or not isinstance(policy_tier, str)
                            or policy_tier not in expected_policy_tiers
                        ):
                            raise ShardVerificationError(
                                "document-policy tier differs"
                            )
                        if isinstance(policy_tier, str):
                            policy_tier_counts[policy_tier] = (
                                policy_tier_counts.get(policy_tier, 0) + 1
                            )
                        source_row = value["source_row_index"]
                        shard_name = value["shard"]
                        if (
                            not isinstance(shard_name, str)
                            or shard_name not in shard_tokens
                            or SHARD_NAME.fullmatch(shard_name) is None
                        ):
                            raise ShardVerificationError(
                                "document ledger shard is not admitted"
                            )
                        try:
                            shard_index = int(
                                shard_name
                                .removeprefix("shard_")
                                .removesuffix(".u16.zst")
                            )
                        except ValueError as exc:
                            raise ShardVerificationError(
                                "document ledger shard is malformed"
                            ) from exc
                        if shard_name != current_shard:
                            current_shard_bytes = _decompressed_bytes(
                                shard_dir / str(shard_name)
                            )
                            current_shard = str(shard_name)
                        token_start = value["token_start"]
                        token_end = value["token_end"]
                        token_slice_sha256 = (
                            hashlib.sha256(
                                current_shard_bytes[
                                    token_start * 2 : token_end * 2
                                ]
                            ).hexdigest()
                            if (
                                isinstance(token_start, int)
                                and isinstance(token_end, int)
                                and 0 <= token_start <= token_end
                            )
                            else None
                        )
                        if (
                            value["schema"] != DOCUMENT_LEDGER_SCHEMA
                            or not isinstance(source_row, int)
                            or source_row <= last_source_row
                            or shard_index < last_shard_index
                            or not isinstance(value["chars"], int)
                            or value["chars"] <= 0
                            or not isinstance(value["tokens"], int)
                            or value["tokens"] <= 0
                            or value["token_start"] != offsets[shard_name]
                            or value["token_end"]
                            != value["token_start"] + value["tokens"]
                            or value["token_end"] > shard_tokens[shard_name]
                            or not isinstance(value["token_sha256"], str)
                            or not HEX64.fullmatch(value["token_sha256"])
                            or token_slice_sha256 != value["token_sha256"]
                            or not isinstance(
                                value["stable_identity_sha256"], str
                            )
                            or not HEX64.fullmatch(
                                value["stable_identity_sha256"]
                            )
                            or not isinstance(value["document_sha256"], str)
                            or not HEX64.fullmatch(value["document_sha256"])
                        ):
                            raise ShardVerificationError(
                                "document ledger row contract differs"
                            )
                        offsets[shard_name] = value["token_end"]
                        last_source_row = source_row
                        last_shard_index = shard_index
                        rows += 1
                        tokens += value["tokens"]
                if pending:
                    raise ShardVerificationError(
                        "document ledger lacks a terminal newline"
                    )
    except (OSError, zstd.ZstdError) as exc:
        raise ShardVerificationError(
            "document ledger cannot be decompressed"
        ) from exc
    if (
        offsets != shard_tokens
        or rows != record.get("rows")
        or rows != kept
        or tokens != record.get("tokens")
        or tokens != sum(shard_tokens.values())
        or (
            expected_policy_counts is not None
            and policy_tier_counts != expected_policy_counts
        )
    ):
        raise ShardVerificationError(
            "document ledger does not reconcile with token shards"
        )
    return {"rows": rows, "tokens": tokens}


def verify_manifest(
    shard_dir: Path,
    *,
    selection_code: Path | None = None,
    require_external_inputs: bool = False,
) -> dict[str, Any]:
    manifest_path = shard_dir / "manifest.json"
    _regular_identity(manifest_path, "manifest")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ShardVerificationError("manifest is unreadable") from exc
    schema = manifest.get("schema")
    if schema not in {
        "shohin-tokenized-shards-v2",
        "shohin-tokenized-shards-v3",
    }:
        raise ShardVerificationError("manifest schema is unsupported")
    filters = manifest.get("filters")
    if filters is not None and not isinstance(filters, dict):
        raise ShardVerificationError("filter contract is malformed")
    document_policy = (
        filters.get("document_policy")
        if isinstance(filters, dict)
        else None
    )
    if document_policy is not None and not isinstance(document_policy, dict):
        raise ShardVerificationError(
            "document-policy contract is malformed"
        )

    claimed_payload = manifest.get("payload_sha256")
    if not isinstance(claimed_payload, str) or not HEX64.fullmatch(claimed_payload):
        raise ShardVerificationError("manifest payload SHA-256 is invalid")
    unsigned = dict(manifest)
    unsigned.pop("payload_sha256", None)
    if canonical_payload_sha256(unsigned) != claimed_payload:
        raise ShardVerificationError("manifest payload SHA-256 differs")

    if selection_code is not None:
        selection_digest, _tokens, _size = _stable_sha256_and_tokens(
            selection_code,
            "selection code",
            count_tokens=False,
            expected_sha256=manifest.get("selection_code_sha256"),
        )
        if selection_digest != manifest.get("selection_code_sha256"):
            raise ShardVerificationError("selection code SHA-256 differs")

    records = manifest.get("shard_files")
    if not isinstance(records, list) or not records:
        raise ShardVerificationError("manifest has no shard-file ledger")
    if manifest.get("shards") != len(records):
        raise ShardVerificationError("manifest shard count differs from ledger")

    expected_names: list[str] = []
    token_total = 0
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ShardVerificationError("shard record is not an object")
        name = record.get("path")
        expected_name = f"shard_{index:05d}.u16.zst"
        if name != expected_name or not SHARD_NAME.fullmatch(str(name)):
            raise ShardVerificationError("shard ledger is not canonical and contiguous")
        path = shard_dir / expected_name
        shard_digest, tokens, size = _stable_sha256_and_tokens(
            path,
            "shard",
            count_tokens=True,
            expected_sha256=record.get("sha256"),
            expected_bytes=record.get("bytes"),
        )
        if size != record.get("bytes"):
            raise ShardVerificationError(f"compressed byte count differs: {expected_name}")
        digest = record.get("sha256")
        if not isinstance(digest, str) or not HEX64.fullmatch(digest):
            raise ShardVerificationError(f"invalid shard SHA-256: {expected_name}")
        if shard_digest != digest:
            raise ShardVerificationError(f"shard SHA-256 differs: {expected_name}")
        if tokens != record.get("tokens"):
            raise ShardVerificationError(f"shard token count differs: {expected_name}")
        expected_names.append(expected_name)
        token_total += tokens

    actual_names = sorted(path.name for path in shard_dir.glob("*.u16.zst"))
    if actual_names != expected_names:
        raise ShardVerificationError("shard directory has missing or unbound shard files")
    if token_total != manifest.get("tokens"):
        raise ShardVerificationError("manifest token total differs from shard ledger")

    document_ledger_verified = False
    document_rows = 0
    if schema == "shohin-tokenized-shards-v3":
        ledger = manifest.get("document_ledger")
        if not isinstance(ledger, dict):
            raise ShardVerificationError("v3 manifest has no document ledger")
        ledger_totals = _verify_document_ledger(
            shard_dir,
            ledger,
            records,
            kept=manifest.get("kept"),
            document_policy=document_policy,
        )
        document_ledger_verified = True
        document_rows = ledger_totals["rows"]
        actual_bound_names = sorted(
            path.name
            for path in shard_dir.iterdir()
            if path.name != "manifest.json"
        )
        expected_bound_names = sorted([*expected_names, DOCUMENT_LEDGER_NAME])
        if actual_bound_names != expected_bound_names:
            raise ShardVerificationError(
                "v3 shard directory contains unbound files"
            )

    if require_external_inputs:
        source_files = manifest.get("source_files")
        if not isinstance(source_files, list):
            raise ShardVerificationError("source-file ledger is absent")
        for index, record in enumerate(source_files):
            if not isinstance(record, dict):
                raise ShardVerificationError("source-file record is malformed")
            _verify_external_file(record, f"source file {index}")
        tokenizer = manifest.get("tokenizer")
        if not isinstance(tokenizer, dict):
            raise ShardVerificationError("tokenizer receipt is absent")
        _verify_external_file(tokenizer, "tokenizer")
        decontamination = manifest.get("decontamination")
        if not isinstance(decontamination, dict):
            raise ShardVerificationError("decontamination receipt is absent")
        pickle_path = decontamination.get("pickle_path")
        if pickle_path is not None:
            _verify_external_file(
                {
                    "path": pickle_path,
                    "bytes": decontamination.get("pickle_bytes"),
                    "sha256": decontamination.get("pickle_sha256"),
                },
                "decontamination pickle",
            )
        eval_files = decontamination.get("eval_files")
        if not isinstance(eval_files, list):
            raise ShardVerificationError("evaluation-file ledger is absent")
        for index, record in enumerate(eval_files):
            if not isinstance(record, dict):
                raise ShardVerificationError("evaluation-file record is malformed")
            _verify_external_file(record, f"evaluation file {index}")
        if document_policy is not None:
            source = document_policy.get("source")
            if not isinstance(source, dict):
                raise ShardVerificationError(
                    "document-policy source receipt is absent"
                )
            _verify_external_file(source, "document-policy source")

    return {
        "schema": "shohin-tokenized-shard-verification-v1",
        "manifest_payload_sha256": claimed_payload,
        "shards": len(records),
        "tokens": token_total,
        "all_shards_hash_and_token_verified": True,
        "document_ledger_verified": document_ledger_verified,
        "document_rows": document_rows,
        "external_inputs_verified": require_external_inputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--selection-code")
    parser.add_argument("--require-external-inputs", action="store_true")
    args = parser.parse_args()
    receipt = verify_manifest(
        Path(args.shard_dir),
        selection_code=Path(args.selection_code) if args.selection_code else None,
        require_external_inputs=args.require_external_inputs,
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
