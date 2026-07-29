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

from pipeline.tokenize_shards import canonical_payload_sha256


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
    if manifest.get("schema") != "shohin-tokenized-shards-v2":
        raise ShardVerificationError("manifest schema is not v2")

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

    if require_external_inputs:
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

    return {
        "schema": "shohin-tokenized-shard-verification-v1",
        "manifest_payload_sha256": claimed_payload,
        "shards": len(records),
        "tokens": token_total,
        "all_shards_hash_and_token_verified": True,
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
