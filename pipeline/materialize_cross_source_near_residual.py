#!/usr/bin/env python3
"""Apply an exact-confirmed near-duplicate ledger to one verified v3 corpus."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping

import zstandard as zstd

from pipeline.audit_cross_source_near_dedup import (
    REMOVAL_SCHEMA,
    REPORT_SCHEMA,
)
from pipeline.build_general_source_review_packet import iter_document_ledger
from pipeline.materialize_cross_source_exact_residual import (
    _source_shard_bytes,
    _write_shard,
)
from pipeline.tokenize_shards import (
    DOCUMENT_LEDGER_NAME,
    DocumentLedgerWriter,
    canonical_payload_sha256,
    sha256_file,
)
from pipeline.verify_tokenized_shards import verify_manifest


RESIDUAL_SCHEMA = "shohin-cross-source-near-residual-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class NearResidualError(ValueError):
    """The near-duplicate receipt cannot produce an admitted residual."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise NearResidualError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise NearResidualError(f"{label} is not an object")
    return value


def _verified_near_report(near_dir: Path) -> dict[str, Any]:
    report_path = near_dir / "report.json"
    report = _load_json(report_path, "near-dedup report")
    claimed = report.get("payload_sha256")
    unsigned = dict(report)
    unsigned.pop("payload_sha256", None)
    algorithm = report.get("algorithm")
    if (
        report.get("schema") != REPORT_SCHEMA
        or not isinstance(claimed, str)
        or HEX64.fullmatch(claimed) is None
        or canonical_payload_sha256(unsigned) != claimed
        or report.get("external_inputs_verified") is not True
        or report.get("exact_duplicate_status")
        != "duplicate_document_sha256_rejected_before_publication"
        or not isinstance(algorithm, dict)
        or algorithm.get("exact_confirmation_required") is not True
    ):
        raise NearResidualError("near-dedup report contract differs")
    removals = report.get("removals")
    if (
        not isinstance(removals, dict)
        or removals.get("path") != "near_duplicate_removals.jsonl.zst"
        or removals.get("contains_document_text") is not False
        or not isinstance(removals.get("rows"), int)
        or removals["rows"] < 0
    ):
        raise NearResidualError("near-dedup removal receipt differs")
    removals_path = near_dir / removals["path"]
    if (
        not removals_path.is_file()
        or removals_path.is_symlink()
        or removals_path.stat().st_nlink != 1
        or removals_path.stat().st_size != removals.get("bytes")
        or sha256_file(removals_path) != removals.get("sha256")
    ):
        raise NearResidualError("near-dedup removal artifact differs")
    return report


def _corpus_record(
    report: Mapping[str, Any],
    *,
    corpus_name: str,
    source_dir: Path,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    corpora = report.get("corpora")
    if not isinstance(corpora, list):
        raise NearResidualError("near-dedup corpus ledger differs")
    matches = [
        item
        for item in corpora
        if isinstance(item, dict) and item.get("name") == corpus_name
    ]
    if len(matches) != 1:
        raise NearResidualError("near-dedup corpus name is not unique")
    record = matches[0]
    if (
        Path(str(record.get("path"))).resolve() != source_dir.resolve()
        or record.get("manifest_payload_sha256")
        != source_manifest.get("payload_sha256")
        or record.get("selection_code_sha256")
        != source_manifest.get("selection_code_sha256")
        or record.get("documents") != source_manifest.get("kept")
        or record.get("tokens") != source_manifest.get("tokens")
    ):
        raise NearResidualError("near-dedup source corpus binding differs")
    return record


def _removals_for_corpus(
    near_dir: Path,
    report: Mapping[str, Any],
    *,
    corpus_name: str,
    record: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    path = near_dir / str(report["removals"]["path"])
    removals: dict[str, dict[str, Any]] = {}
    total_rows = 0
    try:
        with path.open("rb") as source:
            with zstd.ZstdDecompressor().stream_reader(source) as reader:
                with io.TextIOWrapper(reader, encoding="ascii") as text:
                    for line_number, line in enumerate(text, 1):
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise NearResidualError(
                                f"near-removal row {line_number} is malformed"
                            ) from exc
                        drop = row.get("drop") if isinstance(row, dict) else None
                        keep = row.get("keep") if isinstance(row, dict) else None
                        comparison = (
                            row.get("comparison")
                            if isinstance(row, dict)
                            else None
                        )
                        if (
                            not isinstance(row, dict)
                            or row.get("schema") != REMOVAL_SCHEMA
                            or not isinstance(drop, dict)
                            or not isinstance(keep, dict)
                            or not isinstance(comparison, dict)
                            or not isinstance(comparison.get("jaccard"), float)
                            or not isinstance(
                                comparison.get("containment"),
                                float,
                            )
                        ):
                            raise NearResidualError(
                                "near-removal row contract differs"
                            )
                        total_rows += 1
                        if drop.get("corpus") != corpus_name:
                            continue
                        identity = drop.get("stable_identity_sha256")
                        document = drop.get("document_sha256")
                        tokens = drop.get("tokens")
                        if (
                            not isinstance(identity, str)
                            or HEX64.fullmatch(identity) is None
                            or not isinstance(document, str)
                            or HEX64.fullmatch(document) is None
                            or not isinstance(tokens, int)
                            or tokens < 1
                            or identity in removals
                        ):
                            raise NearResidualError(
                                "near-removal identity contract differs"
                            )
                        removals[identity] = {
                            "document_sha256": document,
                            "tokens": tokens,
                        }
    except (OSError, zstd.ZstdError) as exc:
        raise NearResidualError(
            "near-removal artifact cannot be decoded"
        ) from exc
    if (
        total_rows != report["removals"]["rows"]
        or len(removals)
        != record.get("near_duplicate_documents_dropped")
        or sum(item["tokens"] for item in removals.values())
        != record.get("near_duplicate_tokens_dropped")
    ):
        raise NearResidualError("near-removal accounting differs")
    return removals


def materialize_near_residual(
    *,
    source_dir: Path,
    near_dir: Path,
    corpus_name: str,
    source_selection_code: Path,
    selection_code: Path,
    output_dir: Path,
    shard_tokens: int = 100_000_000,
) -> dict[str, Any]:
    if (
        not corpus_name
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
            for character in corpus_name
        )
        or shard_tokens < 1
        or not source_selection_code.is_file()
        or source_selection_code.is_symlink()
        or not selection_code.is_file()
        or selection_code.is_symlink()
    ):
        raise NearResidualError("near-residual arguments differ")
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing existing output: {output_dir}")
    source_verification = verify_manifest(
        source_dir,
        selection_code=source_selection_code,
        require_external_inputs=True,
    )
    source_manifest = _load_json(
        source_dir / "manifest.json",
        "source manifest",
    )
    filters = source_manifest.get("filters")
    if (
        source_manifest.get("schema") != "shohin-tokenized-shards-v3"
        or not source_verification.get("document_ledger_verified")
        or not isinstance(filters, dict)
        or filters.get("exact_dedup") is not True
    ):
        raise NearResidualError(
            "source is not a verified exact-deduplicated v3 corpus"
        )
    report = _verified_near_report(near_dir)
    record = _corpus_record(
        report,
        corpus_name=corpus_name,
        source_dir=source_dir,
        source_manifest=source_manifest,
    )
    removals = _removals_for_corpus(
        near_dir,
        report,
        corpus_name=corpus_name,
        record=record,
    )
    expected_kept = int(record["residual_documents"])
    expected_tokens = int(record["residual_tokens"])
    if expected_kept < 1 or expected_tokens < 1:
        raise NearResidualError("cross-source near residual would be empty")

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
        output_payload = bytearray()
        output_shard = 0
        source_shard: str | None = None
        source_payload = b""
        retained_rows = retained_tokens = dropped_rows = dropped_tokens = 0
        seen_removals: set[str] = set()
        for row in iter_document_ledger(source_dir / DOCUMENT_LEDGER_NAME):
            identity = row["stable_identity_sha256"]
            removal = removals.get(identity)
            if removal is not None:
                if (
                    row["document_sha256"] != removal["document_sha256"]
                    or row["tokens"] != removal["tokens"]
                ):
                    raise NearResidualError(
                        "near-removed document differs from source"
                    )
                seen_removals.add(identity)
                dropped_rows += 1
                dropped_tokens += row["tokens"]
                continue
            if row["shard"] != source_shard:
                source_shard = row["shard"]
                source_payload = _source_shard_bytes(
                    source_dir,
                    source_shard,
                )
            start = int(row["token_start"]) * 2
            end = int(row["token_end"]) * 2
            document_payload = source_payload[start:end]
            if (
                len(document_payload) != int(row["tokens"]) * 2
                or hashlib.sha256(document_payload).hexdigest()
                != row["token_sha256"]
            ):
                raise NearResidualError(
                    "retained document token span differs"
                )
            token_start = len(output_payload) // 2
            output_payload.extend(document_payload)
            output_row = dict(row)
            output_row.update(
                {
                    "shard": f"shard_{output_shard:05d}.u16.zst",
                    "token_start": token_start,
                    "token_end": token_start + row["tokens"],
                }
            )
            ledger.write(output_row)
            retained_rows += 1
            retained_tokens += row["tokens"]
            if len(output_payload) // 2 >= shard_tokens:
                shard_records.append(
                    _write_shard(
                        staging,
                        index=output_shard,
                        payload=output_payload,
                    )
                )
                output_payload = bytearray()
                output_shard += 1
        if output_payload:
            shard_records.append(
                _write_shard(
                    staging,
                    index=output_shard,
                    payload=output_payload,
                )
            )
        ledger_receipt = ledger.close()
        if (
            seen_removals != set(removals)
            or dropped_rows != len(removals)
            or dropped_tokens != record["near_duplicate_tokens_dropped"]
            or retained_rows != expected_kept
            or retained_tokens != expected_tokens
            or ledger_receipt["rows"] != retained_rows
            or ledger_receipt["tokens"] != retained_tokens
            or sum(item["tokens"] for item in shard_records) != retained_tokens
        ):
            raise NearResidualError("near-residual accounting differs")

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
                "tokens": retained_tokens,
                "shards": len(shard_records),
                "shard_files": shard_records,
                "document_ledger": ledger_receipt,
                "kept": retained_rows,
                "dropped_cross_source_near": dropped_rows,
                "dropped_cross_source_near_tokens": dropped_tokens,
                "cross_source_near_residual": {
                    "schema": RESIDUAL_SCHEMA,
                    "corpus_name": corpus_name,
                    "source_path": str(source_dir.resolve()),
                    "source_manifest_payload_sha256": source_manifest[
                        "payload_sha256"
                    ],
                    "source_selection_code_sha256": source_manifest[
                        "selection_code_sha256"
                    ],
                    "source_verification": source_verification,
                    "near_report_path": str(
                        (near_dir / "report.json").resolve()
                    ),
                    "near_report_sha256": sha256_file(
                        near_dir / "report.json"
                    ),
                    "near_report_payload_sha256": report[
                        "payload_sha256"
                    ],
                    "removals_path": str(
                        (
                            near_dir
                            / str(report["removals"]["path"])
                        ).resolve()
                    ),
                    "removals_sha256": report["removals"]["sha256"],
                    "retention_policy": report["retention_policy"],
                    "algorithm": report["algorithm"],
                },
            }
        )
        manifest["payload_sha256"] = canonical_payload_sha256(manifest)
        manifest_path = staging / "manifest.json"
        with manifest_path.open("x") as destination:
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
            verification["tokens"] != retained_tokens
            or verification["document_rows"] != retained_rows
        ):
            raise NearResidualError(
                "published near residual verification differs"
            )
        os.replace(staging, output_dir)
        return {
            "schema": RESIDUAL_SCHEMA,
            "output": str(output_dir),
            "manifest_payload_sha256": manifest["payload_sha256"],
            "documents": retained_rows,
            "tokens": retained_tokens,
            "dropped_documents": dropped_rows,
            "dropped_tokens": dropped_tokens,
            "verification": verification,
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--near-dir", type=Path, required=True)
    parser.add_argument("--corpus-name", required=True)
    parser.add_argument("--source-selection-code", type=Path, required=True)
    parser.add_argument("--selection-code", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-tokens", type=int, default=100_000_000)
    arguments = parser.parse_args(argv)
    result = materialize_near_residual(
        source_dir=arguments.source_dir,
        near_dir=arguments.near_dir,
        corpus_name=arguments.corpus_name,
        source_selection_code=arguments.source_selection_code,
        selection_code=arguments.selection_code,
        output_dir=arguments.output_dir,
        shard_tokens=arguments.shard_tokens,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
