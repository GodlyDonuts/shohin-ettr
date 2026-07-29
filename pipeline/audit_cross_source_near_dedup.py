#!/usr/bin/env python3
"""Find cross-source near duplicates in verified v3 token corpora.

Candidate localization uses sixteen deterministic bottom hashes of five-token
shingles. A document is removed only after exact unique-shingle comparison.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import zstandard as zstd

from pipeline.audit_cross_source_exact_dedup import CorpusSpec
from pipeline.build_general_source_review_packet import iter_document_ledger
from pipeline.tokenize_shards import (
    DOCUMENT_LEDGER_NAME,
    canonical_payload_sha256,
    sha256_file,
)
from pipeline.verify_tokenized_shards import verify_manifest


REPORT_SCHEMA = "shohin-cross-source-near-dedup-report-v1"
REMOVAL_SCHEMA = "shohin-cross-source-near-duplicate-removal-v1"
SHINGLE_TOKENS = 5
SIGNATURE_HASHES = 16
DEFAULT_JACCARD = 0.80
DEFAULT_CONTAINMENT = 0.90
DEFAULT_LENGTH_RATIO = 0.50
DEFAULT_MIN_TOKENS = 32
DEFAULT_BATCH_DOCUMENTS = 20_000
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MASK64 = (1 << 64) - 1
_MIX_A = np.uint64(0xBF58476D1CE4E5B9)
_MIX_B = np.uint64(0x94D049BB133111EB)
_SHINGLE_CONSTANTS = tuple(
    np.uint64(value)
    for value in (
        0x9E3779B97F4A7C15,
        0xD1B54A32D192ED03,
        0x94D049BB133111EB,
        0xDB4F0B9175AE2165,
        0xA24BAED4963EE407,
    )
)


class NearDedupError(ValueError):
    """The near-duplicate audit cannot produce an admissible receipt."""


@dataclass(frozen=True, slots=True)
class Document:
    corpus_index: int
    corpus_name: str
    source_path: Path
    stable_identity_sha256: str
    document_sha256: str
    tokens: int
    shard: str
    token_start: int
    token_end: int
    token_payload: bytes
    signature: tuple[int, ...]


class ShardCache:
    """Small LRU for exact candidate confirmation."""

    def __init__(self, corpora: Sequence[CorpusSpec], maximum: int = 3) -> None:
        self.corpora = tuple(corpora)
        self.maximum = maximum
        self.values: OrderedDict[tuple[int, str], bytes] = OrderedDict()

    def get(self, corpus_index: int, shard: str) -> bytes:
        key = (corpus_index, shard)
        existing = self.values.pop(key, None)
        if existing is not None:
            self.values[key] = existing
            return existing
        path = self.corpora[corpus_index].path / shard
        try:
            with path.open("rb") as source:
                with zstd.ZstdDecompressor().stream_reader(source) as reader:
                    payload = reader.read()
        except (OSError, zstd.ZstdError) as exc:
            raise NearDedupError(
                f"candidate-confirmation shard cannot be decoded: {path}"
            ) from exc
        self.values[key] = payload
        while len(self.values) > self.maximum:
            self.values.popitem(last=False)
        return payload


def _mix64(values: np.ndarray) -> np.ndarray:
    values = values.copy()
    with np.errstate(over="ignore"):
        values ^= values >> np.uint64(30)
        values *= _MIX_A
        values ^= values >> np.uint64(27)
        values *= _MIX_B
        values ^= values >> np.uint64(31)
    return values


def _shingle_hashes(token_payload: bytes, eos_id: int | None) -> np.ndarray:
    if len(token_payload) % 2:
        raise NearDedupError("document token payload has odd length")
    tokens = np.frombuffer(token_payload, dtype="<u2").astype(np.uint64)
    if eos_id is not None and len(tokens) and int(tokens[-1]) == eos_id:
        tokens = tokens[:-1]
    if len(tokens) < SHINGLE_TOKENS:
        return np.empty(0, dtype=np.uint64)
    count = len(tokens) - SHINGLE_TOKENS + 1
    values = np.zeros(count, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for offset, constant in enumerate(_SHINGLE_CONSTANTS):
            values ^= _mix64(
                tokens[offset : offset + count]
                + constant
                + np.uint64(offset)
            )
    return np.unique(_mix64(values))


def _signature(shingles: np.ndarray) -> tuple[int, ...]:
    if len(shingles) < SIGNATURE_HASHES:
        return ()
    indices = np.argpartition(shingles, SIGNATURE_HASHES - 1)[
        :SIGNATURE_HASHES
    ]
    return tuple(sorted(int(value) for value in shingles[indices]))


def _similarity(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[float, float]:
    intersection = int(np.intersect1d(left, right, assume_unique=True).size)
    union = len(left) + len(right) - intersection
    jaccard = intersection / union if union else 1.0
    containment = intersection / min(len(left), len(right))
    return jaccard, containment


def _document_payload(
    shard_payload: bytes,
    row: Mapping[str, Any],
) -> bytes:
    start = int(row["token_start"]) * 2
    end = int(row["token_end"]) * 2
    payload = shard_payload[start:end]
    if (
        len(payload) != int(row["tokens"]) * 2
        or hashlib.sha256(payload).hexdigest() != row["token_sha256"]
    ):
        raise NearDedupError("document token span differs from ledger")
    return payload


def _manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads((path / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise NearDedupError(f"manifest is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise NearDedupError(f"manifest is not an object: {path}")
    return value


def _initialize_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        """
        CREATE TABLE documents (
            document_id INTEGER PRIMARY KEY,
            corpus_index INTEGER NOT NULL,
            stable_identity_sha256 TEXT NOT NULL,
            document_sha256 TEXT NOT NULL UNIQUE,
            tokens INTEGER NOT NULL,
            shard TEXT NOT NULL,
            token_start INTEGER NOT NULL,
            token_end INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE exact_document_hashes (
            document_sha256 TEXT PRIMARY KEY NOT NULL
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        """
        CREATE TABLE fingerprints (
            fingerprint BLOB NOT NULL,
            document_id INTEGER NOT NULL,
            PRIMARY KEY (fingerprint, document_id)
        ) WITHOUT ROWID
        """
    )
    connection.execute(
        "CREATE INDEX fingerprints_document ON fingerprints(document_id)"
    )
    connection.execute(
        """
        CREATE TEMP TABLE current_fingerprints (
            local_id INTEGER NOT NULL,
            fingerprint BLOB NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX current_fingerprint ON current_fingerprints(fingerprint)"
    )
    return connection


def _fingerprint_blob(value: int) -> bytes:
    return value.to_bytes(8, "big", signed=False)


def _prior_candidates(
    connection: sqlite3.Connection,
    documents: Sequence[Document],
) -> dict[int, list[int]]:
    connection.execute("DELETE FROM current_fingerprints")
    connection.executemany(
        "INSERT INTO current_fingerprints VALUES (?, ?)",
        (
            (local_id, _fingerprint_blob(value))
            for local_id, document in enumerate(documents)
            for value in document.signature
        ),
    )
    candidates: dict[int, list[int]] = defaultdict(list)
    for local_id, document_id, _shared in connection.execute(
        """
        SELECT current.local_id, prior.document_id, COUNT(*) AS shared
        FROM current_fingerprints AS current
        JOIN fingerprints AS prior
          ON prior.fingerprint = current.fingerprint
        GROUP BY current.local_id, prior.document_id
        ORDER BY current.local_id, prior.document_id
        """
    ):
        candidates[int(local_id)].append(int(document_id))
    return candidates


def _prior_metadata(
    connection: sqlite3.Connection,
    document_ids: set[int],
) -> dict[int, tuple[Any, ...]]:
    if not document_ids:
        return {}
    connection.execute("DROP TABLE IF EXISTS requested_documents")
    connection.execute(
        "CREATE TEMP TABLE requested_documents(document_id INTEGER PRIMARY KEY)"
    )
    connection.executemany(
        "INSERT INTO requested_documents VALUES (?)",
        ((value,) for value in sorted(document_ids)),
    )
    return {
        int(row[0]): tuple(row[1:])
        for row in connection.execute(
            """
            SELECT documents.document_id, documents.corpus_index,
                   documents.stable_identity_sha256,
                   documents.document_sha256, documents.tokens,
                   documents.shard, documents.token_start,
                   documents.token_end
            FROM requested_documents
            JOIN documents USING(document_id)
            """
        )
    }


def _prior_shingles(
    metadata: tuple[Any, ...],
    *,
    cache: ShardCache,
    eos_ids: Sequence[int | None],
) -> np.ndarray:
    (
        corpus_index,
        _identity,
        _document_sha256,
        _tokens,
        shard,
        token_start,
        token_end,
    ) = metadata
    payload = cache.get(int(corpus_index), str(shard))
    document_payload = payload[int(token_start) * 2 : int(token_end) * 2]
    return _shingle_hashes(document_payload, eos_ids[int(corpus_index)])


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


def audit_near_duplicates(
    corpora: Sequence[CorpusSpec],
    *,
    selection_code: Path,
    output_dir: Path,
    jaccard_threshold: float = DEFAULT_JACCARD,
    containment_threshold: float = DEFAULT_CONTAINMENT,
    minimum_length_ratio: float = DEFAULT_LENGTH_RATIO,
    minimum_tokens: int = DEFAULT_MIN_TOKENS,
    batch_documents: int = DEFAULT_BATCH_DOCUMENTS,
    require_external_inputs: bool = True,
) -> dict[str, Any]:
    if (
        len(corpora) < 2
        or len({corpus.name for corpus in corpora}) != len(corpora)
        or not 0.0 < jaccard_threshold <= 1.0
        or not 0.0 < containment_threshold <= 1.0
        or not 0.0 < minimum_length_ratio <= 1.0
        or minimum_tokens < SHINGLE_TOKENS
        or batch_documents < 1
    ):
        raise NearDedupError("near-dedup arguments differ")
    manifests: list[dict[str, Any]] = []
    verifications: list[dict[str, Any]] = []
    tokenizer_sha256: str | None = None
    eos_ids: list[int | None] = []
    for spec in corpora:
        corpus_selection_code = spec.selection_code or selection_code
        verification = verify_manifest(
            spec.path,
            selection_code=corpus_selection_code,
            require_external_inputs=require_external_inputs,
        )
        manifest = _manifest(spec.path)
        tokenizer = manifest.get("tokenizer")
        filters = manifest.get("filters")
        if (
            manifest.get("schema") != "shohin-tokenized-shards-v3"
            or not verification.get("document_ledger_verified")
            or not isinstance(filters, dict)
            or filters.get("exact_dedup") is not True
            or not isinstance(tokenizer, dict)
            or not isinstance(tokenizer.get("sha256"), str)
        ):
            raise NearDedupError(
                f"corpus is not an exact-deduplicated verified v3 payload: {spec.name}"
            )
        if tokenizer_sha256 is None:
            tokenizer_sha256 = tokenizer["sha256"]
        elif tokenizer_sha256 != tokenizer["sha256"]:
            raise NearDedupError("corpus tokenizer identities differ")
        eos_id = tokenizer.get("eos_id")
        if eos_id is not None and (
            not isinstance(eos_id, int) or isinstance(eos_id, bool)
        ):
            raise NearDedupError("tokenizer EOS identity differs")
        eos_ids.append(eos_id)
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
    database_path = staging / "near.sqlite3"
    removals_path = staging / "near_duplicate_removals.jsonl.zst"
    report_path = staging / "report.json"
    connection: sqlite3.Connection | None = None
    try:
        connection = _initialize_database(database_path)
        cache = ShardCache(corpora)
        next_document_id = 0
        global_kept_rows = 0
        global_removed_rows = 0
        global_removed_tokens = 0
        corpus_stats: list[dict[str, Any]] = []
        reason_counts: Counter[str] = Counter()
        compressor = zstd.ZstdCompressor(level=3)
        with removals_path.open("xb") as raw:
            with compressor.stream_writer(raw, closefd=False) as compressed:
                with io.TextIOWrapper(
                    compressed,
                    encoding="ascii",
                    write_through=True,
                ) as text:
                    for corpus_index, (spec, manifest, verification) in enumerate(
                        zip(corpora, manifests, verifications, strict=True)
                    ):
                        input_rows = input_tokens = 0
                        retained_rows = retained_tokens = 0
                        removed_rows = removed_tokens = 0
                        current_shard: str | None = None
                        current_payload = b""
                        batch: list[Document] = []

                        def process_batch() -> None:
                            nonlocal next_document_id
                            nonlocal retained_rows, retained_tokens
                            nonlocal removed_rows, removed_tokens
                            nonlocal global_kept_rows, global_removed_rows
                            nonlocal global_removed_tokens
                            if not batch:
                                return
                            try:
                                connection.executemany(
                                    "INSERT INTO exact_document_hashes VALUES (?)",
                                    (
                                        (document.document_sha256,)
                                        for document in batch
                                    ),
                                )
                            except sqlite3.IntegrityError as exc:
                                raise NearDedupError(
                                    "cross-source exact duplicate survived the exact gate"
                                ) from exc
                            external = _prior_candidates(connection, batch)
                            prior_ids = {
                                value
                                for values in external.values()
                                for value in values
                            }
                            prior = _prior_metadata(connection, prior_ids)
                            local_index: dict[int, list[int]] = defaultdict(list)
                            local_kept: dict[int, Document] = {}
                            local_shingles: dict[int, np.ndarray] = {}
                            database_rows: list[tuple[Any, ...]] = []
                            database_fingerprints: list[
                                tuple[bytes, int]
                            ] = []
                            for local_id, document in enumerate(batch):
                                candidates: list[
                                    tuple[int, int, Document | tuple[Any, ...]]
                                ] = [
                                    (0, candidate, prior[candidate])
                                    for candidate in external.get(local_id, ())
                                ]
                                local_candidates = {
                                    candidate
                                    for value in document.signature
                                    for candidate in local_index.get(value, ())
                                }
                                candidates.extend(
                                    (
                                        1,
                                        candidate,
                                        local_kept[candidate],
                                    )
                                    for candidate in sorted(local_candidates)
                                )
                                current_shingles: np.ndarray | None = None
                                matched: tuple[
                                    int,
                                    Document | tuple[Any, ...],
                                    float,
                                    float,
                                    float,
                                ] | None = None
                                for kind, candidate_id, candidate in candidates:
                                    candidate_tokens = int(
                                        candidate.tokens
                                        if isinstance(candidate, Document)
                                        else candidate[3]
                                    )
                                    length_ratio = min(
                                        document.tokens,
                                        candidate_tokens,
                                    ) / max(document.tokens, candidate_tokens)
                                    if length_ratio < minimum_length_ratio:
                                        continue
                                    if current_shingles is None:
                                        current_shingles = _shingle_hashes(
                                            document.token_payload,
                                            eos_ids[document.corpus_index],
                                        )
                                    if isinstance(candidate, Document):
                                        candidate_shingles = local_shingles.get(
                                            candidate_id
                                        )
                                        if candidate_shingles is None:
                                            candidate_shingles = _shingle_hashes(
                                                candidate.token_payload,
                                                eos_ids[candidate.corpus_index],
                                            )
                                            local_shingles[
                                                candidate_id
                                            ] = candidate_shingles
                                    else:
                                        candidate_shingles = _prior_shingles(
                                            candidate,
                                            cache=cache,
                                            eos_ids=eos_ids,
                                        )
                                    jaccard, containment = _similarity(
                                        current_shingles,
                                        candidate_shingles,
                                    )
                                    if (
                                        jaccard >= jaccard_threshold
                                        or containment
                                        >= containment_threshold
                                    ):
                                        matched = (
                                            kind,
                                            candidate_id,
                                            candidate,
                                            jaccard,
                                            containment,
                                            length_ratio,
                                        )
                                        break
                                if matched is not None:
                                    (
                                        _kind,
                                        _candidate_id,
                                        keeper,
                                        jaccard,
                                        containment,
                                        length_ratio,
                                    ) = matched
                                    if isinstance(keeper, Document):
                                        keeper_corpus = keeper.corpus_name
                                        keeper_identity = (
                                            keeper.stable_identity_sha256
                                        )
                                        keeper_document = keeper.document_sha256
                                        keeper_tokens = keeper.tokens
                                    else:
                                        keeper_corpus = corpora[
                                            int(keeper[0])
                                        ].name
                                        keeper_identity = str(keeper[1])
                                        keeper_document = str(keeper[2])
                                        keeper_tokens = int(keeper[3])
                                    _write_json_line(
                                        text,
                                        {
                                            "schema": REMOVAL_SCHEMA,
                                            "drop": {
                                                "corpus": document.corpus_name,
                                                "stable_identity_sha256": (
                                                    document.stable_identity_sha256
                                                ),
                                                "document_sha256": (
                                                    document.document_sha256
                                                ),
                                                "tokens": document.tokens,
                                            },
                                            "keep": {
                                                "corpus": keeper_corpus,
                                                "stable_identity_sha256": (
                                                    keeper_identity
                                                ),
                                                "document_sha256": keeper_document,
                                                "tokens": keeper_tokens,
                                            },
                                            "comparison": {
                                                "containment": containment,
                                                "jaccard": jaccard,
                                                "length_ratio": length_ratio,
                                                "shingle_tokens": SHINGLE_TOKENS,
                                            },
                                        },
                                    )
                                    removed_rows += 1
                                    removed_tokens += document.tokens
                                    global_removed_rows += 1
                                    global_removed_tokens += document.tokens
                                    reason_counts[
                                        (
                                            "jaccard"
                                            if jaccard >= jaccard_threshold
                                            else "containment"
                                        )
                                    ] += 1
                                    continue

                                assigned = next_document_id
                                next_document_id += 1
                                local_kept[local_id] = document
                                if current_shingles is not None:
                                    local_shingles[local_id] = current_shingles
                                for value in document.signature:
                                    local_index[value].append(local_id)
                                    database_fingerprints.append(
                                        (_fingerprint_blob(value), assigned)
                                    )
                                database_rows.append(
                                    (
                                        assigned,
                                        document.corpus_index,
                                        document.stable_identity_sha256,
                                        document.document_sha256,
                                        document.tokens,
                                        document.shard,
                                        document.token_start,
                                        document.token_end,
                                    )
                                )
                                retained_rows += 1
                                retained_tokens += document.tokens
                                global_kept_rows += 1
                            connection.executemany(
                                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                database_rows,
                            )
                            connection.executemany(
                                "INSERT INTO fingerprints VALUES (?, ?)",
                                database_fingerprints,
                            )
                            connection.commit()
                            batch.clear()

                        for row in iter_document_ledger(
                            spec.path / DOCUMENT_LEDGER_NAME
                        ):
                            if row["shard"] != current_shard:
                                process_batch()
                                current_shard = row["shard"]
                                current_payload = cache.get(
                                    corpus_index,
                                    current_shard,
                                )
                            token_payload = _document_payload(
                                current_payload,
                                row,
                            )
                            signature = (
                                _signature(
                                    _shingle_hashes(
                                        token_payload,
                                        eos_ids[corpus_index],
                                    )
                                )
                                if row["tokens"] >= minimum_tokens
                                else ()
                            )
                            document = Document(
                                corpus_index=corpus_index,
                                corpus_name=spec.name,
                                source_path=spec.path,
                                stable_identity_sha256=row[
                                    "stable_identity_sha256"
                                ],
                                document_sha256=row["document_sha256"],
                                tokens=row["tokens"],
                                shard=row["shard"],
                                token_start=row["token_start"],
                                token_end=row["token_end"],
                                token_payload=token_payload,
                                signature=signature,
                            )
                            batch.append(document)
                            input_rows += 1
                            input_tokens += row["tokens"]
                            if len(batch) >= batch_documents:
                                process_batch()
                        process_batch()
                        if (
                            input_rows
                            != manifest.get("document_ledger", {}).get("rows")
                            or input_tokens != manifest.get("tokens")
                            or retained_rows + removed_rows != input_rows
                            or retained_tokens + removed_tokens != input_tokens
                        ):
                            raise NearDedupError(
                                f"near-dedup accounting differs: {spec.name}"
                            )
                        corpus_stats.append(
                            {
                                "name": spec.name,
                                "priority": corpus_index,
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
                                "documents": input_rows,
                                "tokens": input_tokens,
                                "near_duplicate_documents_dropped": (
                                    removed_rows
                                ),
                                "near_duplicate_tokens_dropped": (
                                    removed_tokens
                                ),
                                "residual_documents": retained_rows,
                                "residual_tokens": retained_tokens,
                                "verification": verification,
                            }
                        )
        connection.close()
        connection = None
        database_path.unlink()
        removals_receipt = {
            "path": removals_path.name,
            "bytes": removals_path.stat().st_size,
            "sha256": sha256_file(removals_path),
            "rows": global_removed_rows,
            "contains_document_text": False,
        }
        report = {
            "schema": REPORT_SCHEMA,
            "selection_code_sha256": sha256_file(selection_code),
            "tokenizer_sha256": tokenizer_sha256,
            "retention_policy": "first_corpus_then_source_order_wins",
            "algorithm": {
                "candidate_localization": (
                    "sixteen_bottom_splitmix64_five_token_shingles"
                ),
                "candidate_hashes": SIGNATURE_HASHES,
                "shingle_tokens": SHINGLE_TOKENS,
                "minimum_tokens": minimum_tokens,
                "jaccard_threshold": jaccard_threshold,
                "containment_threshold": containment_threshold,
                "minimum_length_ratio": minimum_length_ratio,
                "exact_confirmation_required": True,
            },
            "corpora": corpus_stats,
            "totals": {
                "input_documents": sum(
                    item["documents"] for item in corpus_stats
                ),
                "input_tokens": sum(item["tokens"] for item in corpus_stats),
                "near_duplicate_documents_dropped": global_removed_rows,
                "near_duplicate_tokens_dropped": global_removed_tokens,
                "residual_documents": global_kept_rows,
                "residual_tokens": sum(
                    item["residual_tokens"] for item in corpus_stats
                ),
            },
            "reason_counts": dict(reason_counts),
            "removals": removals_receipt,
            "external_inputs_verified": require_external_inputs,
            "exact_duplicate_status": (
                "duplicate_document_sha256_rejected_before_publication"
            ),
        }
        report["payload_sha256"] = canonical_payload_sha256(report)
        with report_path.open("x") as destination:
            json.dump(report, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.link(removals_path, output_dir / removals_path.name)
        os.link(report_path, output_dir / report_path.name)
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
            pass
        raise


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        action="append",
        type=lambda value: _parse_corpus(value),
        required=True,
    )
    parser.add_argument("--selection-code", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jaccard-threshold", type=float, default=DEFAULT_JACCARD)
    parser.add_argument(
        "--containment-threshold",
        type=float,
        default=DEFAULT_CONTAINMENT,
    )
    parser.add_argument(
        "--minimum-length-ratio",
        type=float,
        default=DEFAULT_LENGTH_RATIO,
    )
    parser.add_argument("--minimum-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    parser.add_argument(
        "--batch-documents",
        type=int,
        default=DEFAULT_BATCH_DOCUMENTS,
    )
    parser.add_argument(
        "--skip-external-input-verification",
        action="store_true",
    )
    arguments = parser.parse_args(argv)
    report = audit_near_duplicates(
        arguments.corpus,
        selection_code=arguments.selection_code,
        output_dir=arguments.output_dir,
        jaccard_threshold=arguments.jaccard_threshold,
        containment_threshold=arguments.containment_threshold,
        minimum_length_ratio=arguments.minimum_length_ratio,
        minimum_tokens=arguments.minimum_tokens,
        batch_documents=arguments.batch_documents,
        require_external_inputs=not arguments.skip_external_input_verification,
    )
    print(json.dumps(report, sort_keys=True))


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
        or not Path(raw_path).is_absolute()
        or (
            code_separator == "::"
            and (
                not raw_selection_code
                or not Path(raw_selection_code).is_absolute()
            )
        )
    ):
        raise argparse.ArgumentTypeError(
            "corpus must be lowercase-name=/absolute/path"
        )
    return CorpusSpec(
        name=name,
        path=Path(raw_path),
        selection_code=(
            Path(raw_selection_code)
            if code_separator == "::"
            else None
        ),
    )


if __name__ == "__main__":
    main()
