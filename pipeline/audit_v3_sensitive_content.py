#!/usr/bin/env python3
"""Audit exact optimizer-visible text for secrets and sensitive identifiers."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
from tokenizers import Tokenizer
import zstandard as zstd

from pipeline.build_general_source_review_packet import iter_document_ledger
from pipeline.tokenize_shards import (
    DOCUMENT_LEDGER_NAME,
    canonical_payload_sha256,
    sha256_file,
)
from pipeline.verify_tokenized_shards import verify_manifest


REPORT_SCHEMA = "shohin-v3-sensitive-content-audit-v1"
FINDING_SCHEMA = "shohin-v3-sensitive-content-finding-v1"
AUTOMATIC_EXCLUSION_CATEGORIES = frozenset(
    {
        "aws_access_key",
        "credential_assignment",
        "github_token",
        "google_api_key",
        "jwt",
        "private_key",
        "slack_token",
    }
)

_PATTERNS = {
    "private_key": re.compile(
        r"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github_token": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{30,255}|github_pat_[A-Za-z0-9_]{40,255})\b"
    ),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{20,255}\b"),
    "jwt": re.compile(
        r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"
    ),
    "email": re.compile(
        r"(?i)(?<![A-Z0-9._%+-])[A-Z0-9._%+-]{1,64}"
        r"@[A-Z0-9-]{1,63}(?:\.[A-Z0-9-]{1,63})*"
        r"\.[A-Z]{2,24}(?![A-Z0-9-]|\.[A-Z0-9])"
    ),
    "ipv4": re.compile(
        r"(?<!\d)(?:25[0-5]|2[0-4]\d|1?\d?\d)"
        r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?!\d)"
    ),
    "ssn_candidate": re.compile(
        r"(?<!\d)(?!000|666|9\d\d)\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}(?!\d)"
    ),
}
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"""(?ix)
    \b(?:api[_ -]?key|access[_ -]?token|auth[_ -]?token|client[_ -]?secret|
        password|passwd|secret[_ -]?key)\b
    \s*(?::|=|=>)\s*
    ["']?([A-Za-z0-9_./+=-]{16,512})
    """
)
_CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_PLACEHOLDER = re.compile(
    r"(?i)(?:example|placeholder|replace|sample|dummy|your[_-]|xxx|redacted|"
    r"changeme|notasecret|<|>|\{|\})"
)


class SensitiveContentAuditError(ValueError):
    """A v3 corpus cannot produce a trustworthy sensitive-content receipt."""


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads((path / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SensitiveContentAuditError("corpus manifest is unreadable") from exc
    if not isinstance(value, dict):
        raise SensitiveContentAuditError("corpus manifest is not an object")
    return value


def _tokenizer(manifest: Mapping[str, Any]) -> tuple[Tokenizer, Path]:
    receipt = manifest.get("tokenizer")
    if not isinstance(receipt, dict):
        raise SensitiveContentAuditError("tokenizer receipt differs")
    path_value = receipt.get("path")
    if not isinstance(path_value, str):
        raise SensitiveContentAuditError("tokenizer path differs")
    path = Path(path_value)
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or _digest(path) != receipt.get("sha256")
    ):
        raise SensitiveContentAuditError("tokenizer identity differs")
    return Tokenizer.from_file(str(path)), path


def _source_shard(path: Path) -> bytes:
    try:
        with path.open("rb") as source:
            with zstd.ZstdDecompressor().stream_reader(source) as reader:
                payload = reader.read()
    except (OSError, zstd.ZstdError) as exc:
        raise SensitiveContentAuditError(
            f"token shard cannot be decoded: {path.name}"
        ) from exc
    if len(payload) % 2:
        raise SensitiveContentAuditError("token shard has odd byte length")
    return payload


def _entropy_bits(value: str) -> float:
    counts = Counter(value)
    length = len(value)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def _credential_values(text: str) -> list[str]:
    values: list[str] = []
    for match in _CREDENTIAL_ASSIGNMENT.finditer(text):
        value = match.group(1)
        classes = sum(
            bool(re.search(pattern, value))
            for pattern in (r"[a-z]", r"[A-Z]", r"\d", r"[^A-Za-z0-9]")
        )
        if (
            len(value) >= 20
            and classes >= 2
            and len(set(value)) >= 8
            and _entropy_bits(value) >= 3.0
            and _PLACEHOLDER.search(value) is None
        ):
            values.append(value)
    return values


def _luhn(value: str) -> bool:
    digits = [int(character) for character in value if character.isdigit()]
    if not 13 <= len(digits) <= 19 or len(set(digits)) < 3:
        return False
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def classify_sensitive_text(text: str) -> dict[str, int]:
    """Return text-free category counts for one optimizer-visible document."""
    counts = {
        category: len(pattern.findall(text))
        for category, pattern in _PATTERNS.items()
    }
    counts["credential_assignment"] = len(_credential_values(text))
    counts["payment_card_candidate"] = sum(
        _luhn(match.group(0)) for match in _CARD_CANDIDATE.finditer(text)
    )
    return {category: count for category, count in counts.items() if count}


def _write_json_line(stream: io.TextIOBase, value: Mapping[str, Any]) -> None:
    stream.write(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
    stream.write("\n")


def audit_sensitive_content(
    *,
    corpus_dir: Path,
    selection_code: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if (
        not selection_code.is_file()
        or selection_code.is_symlink()
        or output_dir.exists()
        or output_dir.is_symlink()
    ):
        raise SensitiveContentAuditError("audit arguments differ")
    verification = verify_manifest(
        corpus_dir,
        selection_code=selection_code,
        require_external_inputs=True,
    )
    manifest = _load_manifest(corpus_dir)
    if (
        manifest.get("schema") != "shohin-tokenized-shards-v3"
        or not verification.get("document_ledger_verified")
        or manifest.get("selection_code_sha256") != sha256_file(selection_code)
    ):
        raise SensitiveContentAuditError("source corpus contract differs")
    tokenizer, tokenizer_path = _tokenizer(manifest)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise SensitiveContentAuditError("refusing existing output") from exc
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.partial-",
            dir=output_dir.parent,
        )
    )
    findings_path = staging / "sensitive_findings.jsonl.zst"
    report_path = staging / "report.json"
    try:
        documents = tokens = flagged_documents = flagged_tokens = 0
        excluded_documents = excluded_tokens = 0
        category_documents: Counter[str] = Counter()
        category_occurrences: Counter[str] = Counter()
        active_shard: str | None = None
        shard_payload = b""
        compressor = zstd.ZstdCompressor(level=3)
        with findings_path.open("xb") as raw:
            with compressor.stream_writer(raw, closefd=False) as compressed:
                with io.TextIOWrapper(
                    compressed,
                    encoding="ascii",
                    write_through=True,
                ) as findings:
                    for row in iter_document_ledger(
                        corpus_dir / DOCUMENT_LEDGER_NAME
                    ):
                        if row["shard"] != active_shard:
                            active_shard = str(row["shard"])
                            shard_payload = _source_shard(
                                corpus_dir / active_shard
                            )
                        start = int(row["token_start"]) * 2
                        end = int(row["token_end"]) * 2
                        payload = shard_payload[start:end]
                        row_tokens = int(row["tokens"])
                        if (
                            len(payload) != row_tokens * 2
                            or hashlib.sha256(payload).hexdigest()
                            != row["token_sha256"]
                        ):
                            raise SensitiveContentAuditError(
                                "document token span differs"
                            )
                        token_ids = np.frombuffer(payload, dtype="<u2").tolist()
                        text = tokenizer.decode(
                            token_ids,
                            skip_special_tokens=True,
                        )
                        categories = classify_sensitive_text(text)
                        documents += 1
                        tokens += row_tokens
                        if not categories:
                            continue
                        automatic = sorted(
                            set(categories) & AUTOMATIC_EXCLUSION_CATEGORIES
                        )
                        flagged_documents += 1
                        flagged_tokens += row_tokens
                        category_documents.update(categories)
                        category_occurrences.update(categories)
                        if automatic:
                            excluded_documents += 1
                            excluded_tokens += row_tokens
                        _write_json_line(
                            findings,
                            {
                                "schema": FINDING_SCHEMA,
                                "automatic_exclusion_categories": automatic,
                                "category_occurrences": categories,
                                "document_sha256": row["document_sha256"],
                                "stable_identity_sha256": row[
                                    "stable_identity_sha256"
                                ],
                                "tokens": row_tokens,
                            },
                        )
        if (
            documents != manifest.get("kept")
            or tokens != manifest.get("tokens")
        ):
            raise SensitiveContentAuditError("audit document accounting differs")
        findings_receipt = {
            "bytes": findings_path.stat().st_size,
            "contains_document_text": False,
            "path": findings_path.name,
            "rows": flagged_documents,
            "sha256": _digest(findings_path),
        }
        report: dict[str, Any] = {
            "schema": REPORT_SCHEMA,
            "automatic_exclusion_categories": sorted(
                AUTOMATIC_EXCLUSION_CATEGORIES
            ),
            "corpus": {
                "documents": documents,
                "manifest_payload_sha256": manifest["payload_sha256"],
                "path": str(corpus_dir.resolve()),
                "tokens": tokens,
                "verification": verification,
            },
            "findings": findings_receipt,
            "policy": {
                "contact_identifiers": (
                    "report_for_source_specific_review_not_automatic_exclusion"
                ),
                "credential_assignments": (
                    "minimum_20_characters_two_character_classes_"
                    "eight_unique_symbols_entropy_at_least_3_bits"
                ),
                "malware_and_unsafe_code": (
                    "not_claimed_by_this_scanner_requires_code_specific_gate"
                ),
                "optimizer_visible_surface": (
                    "decode_exact_hash_verified_uint16_document_spans"
                ),
                "payment_cards_and_ssn": (
                    "report_for_review_not_automatic_exclusion"
                ),
            },
            "selection_code": {
                "path": str(selection_code.resolve()),
                "sha256": sha256_file(selection_code),
            },
            "status": "pass",
            "summary": {
                "automatic_exclusion_documents": excluded_documents,
                "automatic_exclusion_tokens": excluded_tokens,
                "category_documents": dict(sorted(category_documents.items())),
                "category_occurrences": dict(
                    sorted(category_occurrences.items())
                ),
                "flagged_documents": flagged_documents,
                "flagged_tokens": flagged_tokens,
            },
            "tokenizer": {
                "path": str(tokenizer_path.resolve()),
                "sha256": manifest["tokenizer"]["sha256"],
            },
        }
        report["payload_sha256"] = canonical_payload_sha256(report)
        with report_path.open("x") as destination:
            json.dump(report, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.link(findings_path, output_dir / findings_path.name)
        os.link(report_path, output_dir / report_path.name)
        (output_dir / findings_path.name).chmod(0o400)
        (output_dir / report_path.name).chmod(0o400)
        output_dir.chmod(0o500)
        directory = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        shutil.rmtree(staging)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        try:
            output_dir.rmdir()
        except OSError:
            pass
        raise


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--selection-code", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    report = audit_sensitive_content(
        corpus_dir=arguments.corpus_dir,
        selection_code=arguments.selection_code,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
