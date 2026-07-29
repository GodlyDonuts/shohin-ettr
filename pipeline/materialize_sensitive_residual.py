#!/usr/bin/env python3
"""Remove audit-confirmed secrets from one verified v3 token corpus."""

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

from pipeline.audit_v3_sensitive_content import (
    FINDING_SCHEMA,
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


RESIDUAL_SCHEMA = "shohin-sensitive-content-residual-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class SensitiveResidualError(ValueError):
    """A sensitive-content audit cannot produce a verified residual."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SensitiveResidualError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise SensitiveResidualError(f"{label} is not an object")
    return value


def _verified_audit(
    audit_dir: Path,
    *,
    source_dir: Path,
    source_manifest: Mapping[str, Any],
    source_selection_code: Path,
) -> dict[str, Any]:
    report_path = audit_dir / "report.json"
    report = _load_json(report_path, "sensitive audit report")
    claimed = report.get("payload_sha256")
    unsigned = dict(report)
    unsigned.pop("payload_sha256", None)
    corpus = report.get("corpus")
    selection = report.get("selection_code")
    findings = report.get("findings")
    summary = report.get("summary")
    automatic_categories = report.get("automatic_exclusion_categories")
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("status") != "pass"
        or not isinstance(claimed, str)
        or HEX64.fullmatch(claimed) is None
        or canonical_payload_sha256(unsigned) != claimed
        or not isinstance(corpus, dict)
        or Path(str(corpus.get("path"))).resolve() != source_dir.resolve()
        or corpus.get("manifest_payload_sha256")
        != source_manifest.get("payload_sha256")
        or corpus.get("documents") != source_manifest.get("kept")
        or corpus.get("tokens") != source_manifest.get("tokens")
        or not isinstance(corpus.get("verification"), dict)
        or corpus["verification"].get("document_ledger_verified") is not True
        or corpus["verification"].get("external_inputs_verified") is not True
        or not isinstance(selection, dict)
        or Path(str(selection.get("path"))).resolve()
        != source_selection_code.resolve()
        or selection.get("sha256") != sha256_file(source_selection_code)
        or selection.get("sha256")
        != source_manifest.get("selection_code_sha256")
        or not isinstance(findings, dict)
        or findings.get("path") != "sensitive_findings.jsonl.zst"
        or findings.get("contains_document_text") is not False
        or not isinstance(findings.get("rows"), int)
        or isinstance(findings.get("rows"), bool)
        or findings["rows"] < 0
        or not isinstance(automatic_categories, list)
        or automatic_categories != sorted(set(automatic_categories))
        or not automatic_categories
        or any(
            not isinstance(category, str) or not category
            for category in automatic_categories
        )
        or not isinstance(summary, dict)
        or not isinstance(summary.get("automatic_exclusion_documents"), int)
        or isinstance(summary.get("automatic_exclusion_documents"), bool)
        or not isinstance(summary.get("automatic_exclusion_tokens"), int)
        or isinstance(summary.get("automatic_exclusion_tokens"), bool)
        or summary["automatic_exclusion_documents"] < 1
        or summary["automatic_exclusion_tokens"] < 1
        or summary["automatic_exclusion_documents"] > source_manifest.get("kept")
        or summary["automatic_exclusion_tokens"] > source_manifest.get("tokens")
    ):
        raise SensitiveResidualError("sensitive audit contract differs")
    findings_path = audit_dir / findings["path"]
    try:
        metadata = findings_path.lstat()
    except OSError as exc:
        raise SensitiveResidualError("sensitive findings are unavailable") from exc
    if (
        findings_path.is_symlink()
        or not findings_path.is_file()
        or metadata.st_nlink != 1
        or metadata.st_size != findings.get("bytes")
        or sha256_file(findings_path) != findings.get("sha256")
    ):
        raise SensitiveResidualError("sensitive findings receipt differs")
    return report


def _automatic_removals(
    audit_dir: Path,
    report: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    findings = report["findings"]
    path = audit_dir / str(findings["path"])
    removals: dict[str, dict[str, Any]] = {}
    rows = 0
    try:
        with path.open("rb") as source:
            with zstd.ZstdDecompressor().stream_reader(source) as reader:
                with io.TextIOWrapper(reader, encoding="ascii") as text:
                    for line_number, line in enumerate(text, 1):
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise SensitiveResidualError(
                                f"sensitive finding {line_number} is malformed"
                            ) from exc
                        categories = row.get("automatic_exclusion_categories")
                        identity = row.get("stable_identity_sha256")
                        document = row.get("document_sha256")
                        tokens = row.get("tokens")
                        if (
                            not isinstance(row, dict)
                            or row.get("schema") != FINDING_SCHEMA
                            or not isinstance(categories, list)
                            or categories != sorted(set(categories))
                            or any(
                                category
                                not in report["automatic_exclusion_categories"]
                                for category in categories
                            )
                            or not isinstance(identity, str)
                            or HEX64.fullmatch(identity) is None
                            or not isinstance(document, str)
                            or HEX64.fullmatch(document) is None
                            or not isinstance(tokens, int)
                            or isinstance(tokens, bool)
                            or tokens < 1
                        ):
                            raise SensitiveResidualError(
                                "sensitive finding contract differs"
                            )
                        rows += 1
                        if not categories:
                            continue
                        if identity in removals:
                            raise SensitiveResidualError(
                                "sensitive removal identity repeats"
                            )
                        removals[identity] = {
                            "categories": categories,
                            "document_sha256": document,
                            "tokens": tokens,
                        }
    except (OSError, zstd.ZstdError) as exc:
        raise SensitiveResidualError(
            "sensitive findings cannot be decoded"
        ) from exc
    summary = report["summary"]
    if (
        rows != findings["rows"]
        or len(removals) != summary["automatic_exclusion_documents"]
        or sum(item["tokens"] for item in removals.values())
        != summary["automatic_exclusion_tokens"]
    ):
        raise SensitiveResidualError("sensitive removal accounting differs")
    return removals


def materialize_sensitive_residual(
    *,
    source_dir: Path,
    source_selection_code: Path,
    audit_dir: Path,
    selection_code: Path,
    output_dir: Path,
    shard_tokens: int = 100_000_000,
) -> dict[str, Any]:
    if (
        shard_tokens < 1
        or not source_selection_code.is_file()
        or source_selection_code.is_symlink()
        or not selection_code.is_file()
        or selection_code.is_symlink()
        or output_dir.exists()
        or output_dir.is_symlink()
    ):
        raise SensitiveResidualError("sensitive residual arguments differ")
    source_verification = verify_manifest(
        source_dir,
        selection_code=source_selection_code,
        require_external_inputs=True,
    )
    source_manifest = _load_json(source_dir / "manifest.json", "source manifest")
    if (
        source_manifest.get("schema") != "shohin-tokenized-shards-v3"
        or source_verification.get("document_ledger_verified") is not True
    ):
        raise SensitiveResidualError("source is not a verified v3 corpus")
    report = _verified_audit(
        audit_dir,
        source_dir=source_dir,
        source_manifest=source_manifest,
        source_selection_code=source_selection_code,
    )
    removals = _automatic_removals(audit_dir, report)
    expected_rows = int(source_manifest["kept"]) - len(removals)
    expected_tokens = int(source_manifest["tokens"]) - sum(
        item["tokens"] for item in removals.values()
    )
    if expected_rows < 1 or expected_tokens < 1:
        raise SensitiveResidualError("sensitive residual would be empty")

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
        removed_categories: dict[str, int] = {}
        for row in iter_document_ledger(source_dir / DOCUMENT_LEDGER_NAME):
            identity = str(row["stable_identity_sha256"])
            removal = removals.get(identity)
            if removal is not None:
                if (
                    row["document_sha256"] != removal["document_sha256"]
                    or row["tokens"] != removal["tokens"]
                ):
                    raise SensitiveResidualError(
                        "removed document differs from source"
                    )
                seen_removals.add(identity)
                dropped_rows += 1
                dropped_tokens += int(row["tokens"])
                for category in removal["categories"]:
                    removed_categories[category] = (
                        removed_categories.get(category, 0) + 1
                    )
                continue
            if row["shard"] != source_shard:
                source_shard = str(row["shard"])
                source_payload = _source_shard_bytes(source_dir, source_shard)
            start = int(row["token_start"]) * 2
            end = int(row["token_end"]) * 2
            document_payload = source_payload[start:end]
            if (
                len(document_payload) != int(row["tokens"]) * 2
                or hashlib.sha256(document_payload).hexdigest()
                != row["token_sha256"]
            ):
                raise SensitiveResidualError(
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
            retained_tokens += int(row["tokens"])
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
            or dropped_tokens
            != report["summary"]["automatic_exclusion_tokens"]
            or retained_rows != expected_rows
            or retained_tokens != expected_tokens
            or ledger_receipt["rows"] != retained_rows
            or ledger_receipt["tokens"] != retained_tokens
            or sum(item["tokens"] for item in shard_records) != retained_tokens
        ):
            raise SensitiveResidualError("sensitive residual accounting differs")

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
                "dropped_sensitive_content": dropped_rows,
                "dropped_sensitive_content_tokens": dropped_tokens,
                "sensitive_content_residual": {
                    "schema": RESIDUAL_SCHEMA,
                    "source_path": str(source_dir.resolve()),
                    "source_manifest_payload_sha256": source_manifest[
                        "payload_sha256"
                    ],
                    "source_selection_code_sha256": source_manifest[
                        "selection_code_sha256"
                    ],
                    "source_verification": source_verification,
                    "audit_report_path": str(
                        (audit_dir / "report.json").resolve()
                    ),
                    "audit_report_sha256": sha256_file(
                        audit_dir / "report.json"
                    ),
                    "audit_report_payload_sha256": report["payload_sha256"],
                    "findings_path": str(
                        (audit_dir / report["findings"]["path"]).resolve()
                    ),
                    "findings_sha256": report["findings"]["sha256"],
                    "automatic_exclusion_categories": report[
                        "automatic_exclusion_categories"
                    ],
                    "removed_category_documents": dict(
                        sorted(removed_categories.items())
                    ),
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
            raise SensitiveResidualError(
                "published sensitive residual verification differs"
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
            "removed_category_documents": dict(
                sorted(removed_categories.items())
            ),
            "verification": verification,
        }
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-selection-code", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--selection-code", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-tokens", type=int, default=100_000_000)
    arguments = parser.parse_args(argv)
    result = materialize_sensitive_residual(
        source_dir=arguments.source_dir,
        source_selection_code=arguments.source_selection_code,
        audit_dir=arguments.audit_dir,
        selection_code=arguments.selection_code,
        output_dir=arguments.output_dir,
        shard_tokens=arguments.shard_tokens,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
