#!/usr/bin/env python3
"""Build a text-free exact-duplicate receipt across verified v3 corpora."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any, Iterable, Sequence

import zstandard as zstd

from pipeline.build_general_source_review_packet import iter_document_ledger
from pipeline.tokenize_shards import (
    DOCUMENT_LEDGER_NAME,
    canonical_payload_sha256,
    sha256_file,
)
from pipeline.verify_tokenized_shards import verify_manifest


REPORT_SCHEMA = "shohin-cross-source-exact-dedup-report-v1"
REMOVAL_SCHEMA = "shohin-cross-source-exact-duplicate-removal-v1"


class CrossSourceDedupError(ValueError):
    """The cross-source exact-dedup audit cannot be admitted."""


@dataclass(frozen=True, slots=True)
class CorpusSpec:
    name: str
    path: Path
    selection_code: Path | None = None


def _parse_corpus(value: str) -> CorpusSpec:
    name, separator, raw_path = value.partition("=")
    raw_path, code_separator, raw_selection_code = raw_path.partition("::")
    if (
        separator != "="
        or not name
        or not raw_path
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
            for character in name
        )
    ):
        raise argparse.ArgumentTypeError(
            "corpus must be lowercase-name=/absolute/path"
        )
    path = Path(raw_path)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("corpus path must be absolute")
    selection_code = (
        Path(raw_selection_code) if code_separator == "::" else None
    )
    if (
        selection_code is not None
        and (
            not raw_selection_code
            or not selection_code.is_absolute()
        )
    ):
        raise argparse.ArgumentTypeError(
            "corpus selection-code path must be absolute"
        )
    return CorpusSpec(
        name=name,
        path=path,
        selection_code=selection_code,
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads((path / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CrossSourceDedupError(f"manifest is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise CrossSourceDedupError(f"manifest is not an object: {path}")
    return value


def _write_json_line(stream: io.TextIOBase, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")))
    stream.write("\n")


def _corpus_record(
    spec: CorpusSpec,
    *,
    default_selection_code: Path,
    require_external_inputs: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection_code = spec.selection_code or default_selection_code
    verification = verify_manifest(
        spec.path,
        selection_code=selection_code,
        require_external_inputs=require_external_inputs,
    )
    manifest = _load_manifest(spec.path)
    if (
        manifest.get("schema") != "shohin-tokenized-shards-v3"
        or not verification.get("document_ledger_verified")
        or manifest.get("filters", {}).get("exact_dedup") is not True
    ):
        raise CrossSourceDedupError(
            f"corpus is not an internally exact-deduplicated v3 payload: {spec.name}"
        )
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict) or not isinstance(
        tokenizer.get("sha256"), str
    ):
        raise CrossSourceDedupError(f"tokenizer receipt differs: {spec.name}")
    return manifest, verification


def audit_exact_duplicates(
    corpora: Sequence[CorpusSpec],
    *,
    selection_code: Path,
    output_dir: Path,
    require_external_inputs: bool = True,
) -> dict[str, Any]:
    if len(corpora) < 2:
        raise CrossSourceDedupError("at least two corpora are required")
    if len({corpus.name for corpus in corpora}) != len(corpora):
        raise CrossSourceDedupError("corpus names must be unique")
    manifests: list[dict[str, Any]] = []
    verifications: list[dict[str, Any]] = []
    tokenizer_sha256: str | None = None
    for spec in corpora:
        manifest, verification = _corpus_record(
            spec,
            default_selection_code=selection_code,
            require_external_inputs=require_external_inputs,
        )
        current_tokenizer = str(manifest["tokenizer"]["sha256"])
        if tokenizer_sha256 is None:
            tokenizer_sha256 = current_tokenizer
        elif tokenizer_sha256 != current_tokenizer:
            raise CrossSourceDedupError("corpus tokenizer identities differ")
        manifests.append(manifest)
        verifications.append(verification)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing existing output directory: {output_dir}"
        ) from exc
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent)
    )
    database_path = staging / "seen.sqlite3"
    removals_path = staging / "exact_duplicate_removals.jsonl.zst"
    report_path = staging / "report.json"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(database_path)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute(
            """
            CREATE TABLE seen (
                document_sha256 TEXT PRIMARY KEY NOT NULL,
                corpus_name TEXT NOT NULL,
                stable_identity_sha256 TEXT NOT NULL,
                tokens INTEGER NOT NULL
            ) WITHOUT ROWID
            """
        )
        duplicate_rows = 0
        duplicate_tokens = 0
        corpus_stats: list[dict[str, Any]] = []
        compressor = zstd.ZstdCompressor(level=3)
        with removals_path.open("xb") as raw:
            with compressor.stream_writer(raw, closefd=False) as compressed:
                with io.TextIOWrapper(
                    compressed,
                    encoding="ascii",
                    write_through=True,
                ) as text:
                    for priority, (spec, manifest, verification) in enumerate(
                        zip(corpora, manifests, verifications, strict=True)
                    ):
                        rows = tokens = dropped_rows = dropped_tokens = 0
                        for row in iter_document_ledger(
                            spec.path / DOCUMENT_LEDGER_NAME
                        ):
                            document_sha256 = row["document_sha256"]
                            identity = row["stable_identity_sha256"]
                            document_tokens = row["tokens"]
                            if (
                                not isinstance(document_sha256, str)
                                or not isinstance(identity, str)
                                or not isinstance(document_tokens, int)
                                or document_tokens < 1
                            ):
                                raise CrossSourceDedupError(
                                    f"document ledger fields differ: {spec.name}"
                                )
                            rows += 1
                            tokens += document_tokens
                            try:
                                connection.execute(
                                    "INSERT INTO seen VALUES (?, ?, ?, ?)",
                                    (
                                        document_sha256,
                                        spec.name,
                                        identity,
                                        document_tokens,
                                    ),
                                )
                            except sqlite3.IntegrityError:
                                keeper = connection.execute(
                                    """
                                    SELECT corpus_name, stable_identity_sha256, tokens
                                    FROM seen WHERE document_sha256 = ?
                                    """,
                                    (document_sha256,),
                                ).fetchone()
                                if keeper is None or keeper[0] == spec.name:
                                    raise CrossSourceDedupError(
                                        f"within-source exact duplicate survived: {spec.name}"
                                    )
                                _write_json_line(
                                    text,
                                    {
                                        "schema": REMOVAL_SCHEMA,
                                        "document_sha256": document_sha256,
                                        "drop": {
                                            "corpus": spec.name,
                                            "stable_identity_sha256": identity,
                                            "tokens": document_tokens,
                                        },
                                        "keep": {
                                            "corpus": keeper[0],
                                            "stable_identity_sha256": keeper[1],
                                            "tokens": keeper[2],
                                        },
                                    },
                                )
                                dropped_rows += 1
                                dropped_tokens += document_tokens
                        if (
                            rows != manifest.get("document_ledger", {}).get("rows")
                            or rows != verification.get("document_rows")
                            or tokens != manifest.get("tokens")
                        ):
                            raise CrossSourceDedupError(
                                f"document accounting differs: {spec.name}"
                            )
                        duplicate_rows += dropped_rows
                        duplicate_tokens += dropped_tokens
                        corpus_stats.append(
                            {
                                "name": spec.name,
                                "priority": priority,
                                "path": str(spec.path.resolve()),
                                "selection_code_path": str(
                                    (
                                        spec.selection_code
                                        or selection_code
                                    ).resolve()
                                ),
                                "selection_code_sha256": manifest[
                                    "selection_code_sha256"
                                ],
                                "manifest_payload_sha256": manifest[
                                    "payload_sha256"
                                ],
                                "documents": rows,
                                "tokens": tokens,
                                "exact_duplicate_documents_dropped": dropped_rows,
                                "exact_duplicate_tokens_dropped": dropped_tokens,
                                "residual_documents": rows - dropped_rows,
                                "residual_tokens": tokens - dropped_tokens,
                                "verification": verification,
                            }
                        )
        connection.close()
        connection = None
        database_path.unlink()
        report = {
            "schema": REPORT_SCHEMA,
            "selection_code_sha256": sha256_file(selection_code),
            "tokenizer_sha256": tokenizer_sha256,
            "retention_policy": "first_corpus_in_declared_order_wins",
            "corpora": corpus_stats,
            "totals": {
                "input_documents": sum(item["documents"] for item in corpus_stats),
                "input_tokens": sum(item["tokens"] for item in corpus_stats),
                "exact_duplicate_documents_dropped": duplicate_rows,
                "exact_duplicate_tokens_dropped": duplicate_tokens,
                "residual_documents": sum(
                    item["residual_documents"] for item in corpus_stats
                ),
                "residual_tokens": sum(item["residual_tokens"] for item in corpus_stats),
            },
            "removals": {
                "path": removals_path.name,
                "bytes": removals_path.stat().st_size,
                "sha256": sha256_file(removals_path),
                "rows": duplicate_rows,
                "contains_document_text": False,
            },
            "external_inputs_verified": require_external_inputs,
            "near_duplicate_status": (
                "not_measured_exact_audit_is_not_near_duplicate_admission"
            ),
        }
        report["payload_sha256"] = canonical_payload_sha256(report)
        with report_path.open("x") as destination:
            json.dump(report, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.link(
            removals_path,
            output_dir / "exact_duplicate_removals.jsonl.zst",
        )
        os.link(report_path, output_dir / "report.json")
        directory_fd = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        shutil.rmtree(staging)
        return report
    except BaseException:
        if connection is not None:
            connection.close()
        shutil.rmtree(staging, ignore_errors=True)
        try:
            output_dir.rmdir()
        except OSError:
            # A partially published directory has no report completion marker
            # and is deliberately left for forensic inspection.
            pass
        raise


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", type=_parse_corpus, required=True)
    parser.add_argument("--selection-code", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--skip-external-input-verification",
        action="store_true",
        help="diagnostic only; production admission must not set this flag",
    )
    arguments = parser.parse_args(argv)
    report = audit_exact_duplicates(
        arguments.corpus,
        selection_code=arguments.selection_code,
        output_dir=arguments.output_dir,
        require_external_inputs=not arguments.skip_external_input_verification,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
