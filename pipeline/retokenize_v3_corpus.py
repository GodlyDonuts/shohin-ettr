#!/usr/bin/env python3
"""Retokenize a verified Shohin v3 corpus without recovering source files."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable
import unicodedata

import numpy as np
from tokenizers import Tokenizer
import zstandard as zstd

from pipeline.tokenize_shards import (
    DOCUMENT_LEDGER_NAME,
    DOCUMENT_LEDGER_SCHEMA,
    DocumentLedgerWriter,
    canonical_payload_sha256,
    file_receipt,
    sha256_file,
)
from pipeline.verify_tokenized_shards import verify_manifest


RETOKENIZATION_SCHEMA = "shohin-v3-retokenization-v1"


class RetokenizationError(ValueError):
    """A source corpus cannot be retokenized without changing its documents."""


def _differs_only_by_control_deletions(source: str, target: str) -> bool:
    source_index = target_index = 0
    while source_index < len(source) and target_index < len(target):
        if source[source_index] == target[target_index]:
            source_index += 1
            target_index += 1
            continue
        character = source[source_index]
        if unicodedata.category(character) != "Cc" or character in "\n\r\t":
            return False
        source_index += 1
    if target_index != len(target):
        return False
    return all(
        unicodedata.category(character) == "Cc" and character not in "\n\r\t"
        for character in source[source_index:]
    )


def _iter_document_ledger(path: Path) -> Iterable[dict[str, Any]]:
    try:
        with path.open("rb") as source:
            with zstd.ZstdDecompressor().stream_reader(source) as reader:
                with io.TextIOWrapper(reader, encoding="ascii") as text:
                    for line_number, line in enumerate(text, 1):
                        row = json.loads(line)
                        if (
                            not isinstance(row, dict)
                            or row.get("schema") != DOCUMENT_LEDGER_SCHEMA
                        ):
                            raise RetokenizationError(
                                f"document ledger row {line_number} differs"
                            )
                        yield row
    except (OSError, zstd.ZstdError, json.JSONDecodeError) as exc:
        raise RetokenizationError("document ledger cannot be decoded") from exc


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RetokenizationError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise RetokenizationError(f"{label} is not an object")
    return value


def _source_shard_bytes(source_dir: Path, name: str) -> bytes:
    path = source_dir / name
    try:
        with path.open("rb") as source:
            with zstd.ZstdDecompressor().stream_reader(source) as reader:
                payload = reader.read()
    except (OSError, zstd.ZstdError) as exc:
        raise RetokenizationError(f"source shard cannot be decoded: {name}") from exc
    if len(payload) % 2:
        raise RetokenizationError(f"source shard has odd byte length: {name}")
    return payload


def _write_shard(
    output_dir: Path,
    *,
    index: int,
    token_ids: list[int],
) -> dict[str, Any]:
    if not token_ids:
        raise RetokenizationError("retokenized shard is empty")
    if min(token_ids) < 0 or max(token_ids) > np.iinfo(np.uint16).max:
        raise RetokenizationError("retokenized token ID exceeds uint16")
    payload = np.asarray(token_ids, dtype="<u2").tobytes()
    compressed = zstd.ZstdCompressor(level=3).compress(payload)
    path = output_dir / f"shard_{index:05d}.u16.zst"
    with path.open("xb") as destination:
        if destination.write(compressed) != len(compressed):
            raise RetokenizationError("retokenized shard write differs")
        destination.flush()
        os.fsync(destination.fileno())
    return {
        "path": path.name,
        "bytes": len(compressed),
        "tokens": len(token_ids),
        "sha256": hashlib.sha256(compressed).hexdigest(),
    }


def _source_document_ids(
    *,
    row: dict[str, Any],
    shard_payload: bytes,
    source_eos_id: int | None,
) -> list[int]:
    start = int(row["token_start"]) * 2
    end = int(row["token_end"]) * 2
    payload = shard_payload[start:end]
    if (
        len(payload) != int(row["tokens"]) * 2
        or hashlib.sha256(payload).hexdigest() != row["token_sha256"]
    ):
        raise RetokenizationError("source document token span differs")
    ids = np.frombuffer(payload, dtype="<u2").astype(np.int64).tolist()
    if source_eos_id is not None:
        if not ids or ids[-1] != source_eos_id:
            raise RetokenizationError("source document lacks its bound EOS token")
        ids.pop()
    if not ids:
        raise RetokenizationError("source document is empty after removing EOS")
    return ids


def _retokenize_batch(
    *,
    rows_and_ids: list[tuple[dict[str, Any], list[int]]],
    source_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    target_eos_id: int,
) -> tuple[
    list[tuple[dict[str, Any], list[int]]],
    list[tuple[dict[str, Any], int]],
]:
    source_id_batches = [ids for _row, ids in rows_and_ids]
    texts = source_tokenizer.decode_batch(
        source_id_batches,
        skip_special_tokens=False,
    )
    source_reencodings = source_tokenizer.encode_batch(
        texts,
        add_special_tokens=False,
    )
    target_encodings = target_tokenizer.encode_batch(
        texts,
        add_special_tokens=False,
    )
    target_id_batches = [encoding.ids for encoding in target_encodings]
    target_texts = target_tokenizer.decode_batch(
        target_id_batches,
        skip_special_tokens=False,
    )
    target_reencodings = target_tokenizer.encode_batch(
        target_texts,
        add_special_tokens=False,
    )
    output: list[tuple[dict[str, Any], list[int]]] = []
    dropped: list[tuple[dict[str, Any], int]] = []
    for index, ((row, source_ids), text) in enumerate(zip(rows_and_ids, texts)):
        if (
            source_reencodings[index].ids != source_ids
            or len(text) != row["chars"]
            or hashlib.sha256(text.encode("utf-8")).hexdigest()
            != row["document_sha256"]
            or target_reencodings[index].ids != target_id_batches[index]
        ):
            raise RetokenizationError(
                "document text or tokenizer round trip differs"
            )
        target_ids = [*target_id_batches[index], target_eos_id]
        if not target_ids or max(target_ids) > np.iinfo(np.uint16).max:
            raise RetokenizationError("target document token IDs exceed uint16")
        if target_texts[index] != text:
            if not _differs_only_by_control_deletions(
                text,
                target_texts[index],
            ):
                raise RetokenizationError(
                    "target tokenizer changes printable document content"
                )
            dropped.append((row, len(target_ids)))
            continue
        output.append((row, target_ids))
    return output, dropped


def retokenize_corpus(
    *,
    source_dir: Path,
    source_selection_code: Path,
    target_tokenizer_path: Path,
    target_eos_token: str,
    selection_code: Path,
    output_dir: Path,
    shard_tokens: int = 100_000_000,
    batch_size: int = 1_024,
) -> dict[str, Any]:
    if (
        shard_tokens < 1
        or batch_size < 1
        or not target_eos_token
        or not source_selection_code.is_file()
        or source_selection_code.is_symlink()
        or not target_tokenizer_path.is_file()
        or target_tokenizer_path.is_symlink()
        or not selection_code.is_file()
        or selection_code.is_symlink()
    ):
        raise RetokenizationError("retokenization arguments differ")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing existing output: {output_dir}")

    source_verification = verify_manifest(
        source_dir,
        selection_code=source_selection_code,
        require_external_inputs=True,
    )
    source_manifest = _load_json(source_dir / "manifest.json", "source manifest")
    source_tokenizer_record = source_manifest.get("tokenizer")
    if (
        source_manifest.get("schema") != "shohin-tokenized-shards-v3"
        or not source_verification.get("document_ledger_verified")
        or not isinstance(source_tokenizer_record, dict)
        or not isinstance(source_tokenizer_record.get("path"), str)
        or source_manifest.get("kept") != source_verification.get("document_rows")
    ):
        raise RetokenizationError("source is not a verified v3 corpus")

    source_tokenizer_path = Path(source_tokenizer_record["path"])
    source_tokenizer = Tokenizer.from_file(str(source_tokenizer_path))
    target_tokenizer = Tokenizer.from_file(str(target_tokenizer_path))
    source_eos_id = source_tokenizer_record.get("eos_id")
    if source_eos_id is not None and not isinstance(source_eos_id, int):
        raise RetokenizationError("source EOS receipt differs")
    target_eos_id = target_tokenizer.token_to_id(target_eos_token)
    target_vocab_size = target_tokenizer.get_vocab_size(with_added_tokens=True)
    if (
        target_eos_id is None
        or target_vocab_size < 1
        or target_vocab_size > np.iinfo(np.uint16).max + 1
    ):
        raise RetokenizationError("target tokenizer cannot use uint16 shards")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.partial-",
            dir=output_dir.parent,
        )
    )
    try:
        ledger = DocumentLedgerWriter(staging / DOCUMENT_LEDGER_NAME)
        shard_records: list[dict[str, Any]] = []
        output_ids: list[int] = []
        output_shard_index = 0
        source_shard_name: str | None = None
        source_shard_payload = b""
        pending: list[tuple[dict[str, Any], list[int]]] = []
        documents = 0
        target_tokens = 0
        next_progress = 100_000
        dropped_documents = 0
        dropped_source_tokens = 0
        dropped_target_candidate_tokens = 0
        retained_domains: set[str] = set()
        retained_policy_tiers: Counter[str] = Counter()

        def flush_shard() -> None:
            nonlocal output_ids, output_shard_index
            if not output_ids:
                return
            shard_records.append(
                _write_shard(
                    staging,
                    index=output_shard_index,
                    token_ids=output_ids,
                )
            )
            output_ids = []
            output_shard_index += 1

        def commit_pending() -> None:
            nonlocal documents, dropped_documents, dropped_source_tokens
            nonlocal dropped_target_candidate_tokens, next_progress, target_tokens
            converted, dropped = _retokenize_batch(
                rows_and_ids=pending,
                source_tokenizer=source_tokenizer,
                target_tokenizer=target_tokenizer,
                target_eos_id=target_eos_id,
            )
            dropped_documents += len(dropped)
            dropped_source_tokens += sum(int(row["tokens"]) for row, _ in dropped)
            dropped_target_candidate_tokens += sum(tokens for _row, tokens in dropped)
            for source_row, target_ids in converted:
                token_start = len(output_ids)
                token_payload = np.asarray(target_ids, dtype="<u2").tobytes()
                output_row = dict(source_row)
                output_row.update(
                    {
                        "tokens": len(target_ids),
                        "shard": f"shard_{output_shard_index:05d}.u16.zst",
                        "token_start": token_start,
                        "token_end": token_start + len(target_ids),
                        "token_sha256": hashlib.sha256(token_payload).hexdigest(),
                    }
                )
                output_ids.extend(target_ids)
                ledger.write(output_row)
                documents += 1
                target_tokens += len(target_ids)
                domain = source_row.get("domain")
                if isinstance(domain, str):
                    retained_domains.add(domain)
                tier = source_row.get("document_policy_tier")
                if isinstance(tier, str):
                    retained_policy_tiers[tier] += 1
                if len(output_ids) >= shard_tokens:
                    flush_shard()
            if documents >= next_progress:
                print(
                    json.dumps(
                        {"documents": documents, "target_tokens": target_tokens}
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                next_progress = ((documents // 100_000) + 1) * 100_000

        for row in _iter_document_ledger(source_dir / DOCUMENT_LEDGER_NAME):
            if row["shard"] != source_shard_name:
                if pending:
                    commit_pending()
                    pending.clear()
                source_shard_name = row["shard"]
                source_shard_payload = _source_shard_bytes(
                    source_dir,
                    source_shard_name,
                )
            pending.append(
                (
                    row,
                    _source_document_ids(
                        row=row,
                        shard_payload=source_shard_payload,
                        source_eos_id=source_eos_id,
                    ),
                )
            )
            if len(pending) >= batch_size:
                commit_pending()
                pending.clear()
        if pending:
            commit_pending()
        flush_shard()
        ledger_receipt = ledger.close()
        if (
            documents + dropped_documents != source_manifest["kept"]
            or documents < 1
            or ledger_receipt["rows"] != documents
            or ledger_receipt["tokens"] != target_tokens
            or sum(record["tokens"] for record in shard_records) != target_tokens
        ):
            raise RetokenizationError("retokenized corpus accounting differs")

        target_receipt = file_receipt(target_tokenizer_path)
        manifest = {
            key: value
            for key, value in json.loads(json.dumps(source_manifest)).items()
            if key
            not in {
                "payload_sha256",
                "selection_code_sha256",
                "tokenizer",
                "tokens",
                "shards",
                "shard_files",
                "document_ledger",
            }
        }
        manifest.update(
            {
                "schema": "shohin-tokenized-shards-v3",
                "selection_code_sha256": sha256_file(selection_code),
                "tokenizer": {
                    **target_receipt,
                    "vocab_size": target_vocab_size,
                    "eos_token": target_eos_token,
                    "eos_id": target_eos_id,
                    "batch_size": batch_size,
                },
                "tokens": target_tokens,
                "shards": len(shard_records),
                "shard_files": shard_records,
                "document_ledger": ledger_receipt,
                "kept": documents,
                "dropped_retokenization_nonroundtrip": dropped_documents,
                "dropped_retokenization_nonroundtrip_source_tokens": (
                    dropped_source_tokens
                ),
                "retokenization": {
                    "schema": RETOKENIZATION_SCHEMA,
                    "source_path": str(source_dir.resolve()),
                    "source_manifest_sha256": sha256_file(
                        source_dir / "manifest.json"
                    ),
                    "source_manifest_payload_sha256": source_manifest[
                        "payload_sha256"
                    ],
                    "source_selection_code_sha256": source_manifest[
                        "selection_code_sha256"
                    ],
                    "source_tokenizer": source_tokenizer_record,
                    "source_tokens": source_manifest["tokens"],
                    "target_tokens": target_tokens,
                    "documents": documents,
                    "dropped_documents": dropped_documents,
                    "dropped_source_tokens": dropped_source_tokens,
                    "dropped_target_candidate_tokens": (
                        dropped_target_candidate_tokens
                    ),
                    "drop_policy": (
                        "drop_whole_document_only_when_target_decode_differs_"
                        "solely_by_removed_non_tab_newline_carriage_return_"
                        "unicode_control_characters"
                    ),
                    "all_source_text_sha256_verified": True,
                    "all_source_roundtrips_verified": True,
                    "all_target_roundtrips_verified": True,
                    "contains_document_text": False,
                },
            }
        )
        filters = manifest.get("filters")
        if isinstance(filters, dict):
            if "retained_domains" in filters:
                filters["retained_domains"] = len(retained_domains)
            policy = filters.get("document_policy")
            if isinstance(policy, dict):
                policy["retained_tiers"] = dict(sorted(retained_policy_tiers.items()))
        manifest["payload_sha256"] = canonical_payload_sha256(manifest)
        manifest_path = staging / "manifest.json"
        with manifest_path.open("x", encoding="ascii") as destination:
            json.dump(manifest, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        verification = verify_manifest(
            staging,
            selection_code=selection_code,
            require_external_inputs=True,
        )
        if (
            verification["tokens"] != target_tokens
            or verification["document_rows"] != documents
        ):
            raise RetokenizationError("published corpus verification differs")
        os.replace(staging, output_dir)
        report = {
            "schema": RETOKENIZATION_SCHEMA,
            "output": str(output_dir),
            "source_manifest_payload_sha256": source_manifest["payload_sha256"],
            "manifest_payload_sha256": manifest["payload_sha256"],
            "documents": documents,
            "dropped_documents": dropped_documents,
            "dropped_source_tokens": dropped_source_tokens,
            "source_tokens": source_manifest["tokens"],
            "target_tokens": target_tokens,
            "token_reduction": source_manifest["tokens"] - target_tokens,
            "verification": verification,
        }
        report["payload_sha256"] = canonical_payload_sha256(report)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-selection-code", type=Path, required=True)
    parser.add_argument("--target-tokenizer", type=Path, required=True)
    parser.add_argument("--target-eos-token", default="<|endoftext|>")
    parser.add_argument("--selection-code", type=Path, default=Path(__file__))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--shard-tokens", type=int, default=100_000_000)
    parser.add_argument("--batch-size", type=int, default=1_024)
    arguments = parser.parse_args(argv)
    report = retokenize_corpus(
        source_dir=arguments.source_dir,
        source_selection_code=arguments.source_selection_code,
        target_tokenizer_path=arguments.target_tokenizer,
        target_eos_token=arguments.target_eos_token,
        selection_code=arguments.selection_code,
        output_dir=arguments.output_dir,
        shard_tokens=arguments.shard_tokens,
        batch_size=arguments.batch_size,
    )
    if arguments.report.exists() or arguments.report.is_symlink():
        raise FileExistsError(f"refusing existing report: {arguments.report}")
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.report.with_name(
        f".{arguments.report.name}.tmp.{os.getpid()}"
    )
    try:
        with temporary.open("x", encoding="ascii") as destination:
            json.dump(report, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, arguments.report)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
