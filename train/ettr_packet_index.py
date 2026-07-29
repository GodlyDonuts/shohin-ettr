"""Disk-backed packet-sufficiency admission for large ETTR corpora.

The in-memory index is appropriate for qualification boards. Production v3
contains millions of expanded rows, so duplicating Python tuples and frozensets
in every distributed rank is wasteful. This module builds one immutable,
fixed-width index and verifies batches with binary search over read-only mmap
files while preserving the exact receipt semantics of
``ETTRPacketSufficiencyIndex``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import sqlite3
import stat
import struct
from typing import Iterable, Iterator, Mapping, Sequence

from endogenous_typed_theory_reactor import TheoryReactorError
from ettr_data_contract import (
    ETTRContinuationBatch,
    ETTRPacketSufficiencyReceipt,
    ETTR_PACKET_SUFFICIENCY_SCHEMA,
    continuation_batch_payload_sha256,
    terminal_packet_query_context,
)


DISK_INDEX_SCHEMA = "shohin-ettr-packet-sufficiency-disk-v1"
CONTEXT_RECORD_BYTES = 34
BATCH_RECORD_BYTES = 32
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_FILES = {
    "contexts": "contexts.bin",
    "train_batches": "train-batches.bin",
    "validation_batches": "validation-batches.bin",
}


class ETTRPacketIndexError(ValueError):
    """A disk-backed packet-sufficiency index violates its receipt."""


@dataclass(frozen=True)
class ETTRCompactPacketBatch:
    """Cryptographic packet-index projection of one continuation batch."""

    payload_digest: bytes
    rows: tuple[tuple[bytes, int], ...]

    def validate(self) -> None:
        if len(self.payload_digest) != 32 or not self.rows:
            raise ETTRPacketIndexError("compact packet batch geometry differs")
        for context, target in self.rows:
            if len(context) != 32 or not 0 <= target <= 0xFFFF:
                raise ETTRPacketIndexError("compact packet row differs")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _canonical_file_bytes(value: object) -> bytes:
    return _canonical_json_bytes(value) + b"\n"


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _identity(path: Path, label: str, *, immutable: bool) -> tuple[int, ...]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ETTRPacketIndexError(f"{label} cannot be inspected") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or (immutable and metadata.st_mode & 0o222)
    ):
        raise ETTRPacketIndexError(
            f"{label} is not an immutable single-link regular file"
        )
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _write_no_replace(path: Path, chunks: Iterable[bytes]) -> dict[str, object]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o400)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            for chunk in chunks:
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return {"bytes": size, "path": path.name, "sha256": digest.hexdigest()}


def _json_string_list_sha256(values: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    for value in values:
        if not first:
            digest.update(b",")
        first = False
        digest.update(b'"')
        digest.update(value.hex().encode("ascii"))
        digest.update(b'"')
    digest.update(b"]")
    return digest.hexdigest()


def _target_commitment(context: bytes, target: int) -> bytes:
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "context_sha256": context.hex(),
                "target": target,
            }
        )
    ).digest()


def compact_packet_batch(
    batch: ETTRContinuationBatch,
) -> ETTRCompactPacketBatch:
    """Project a batch to the exact commitments consumed by the index."""

    if not isinstance(batch, ETTRContinuationBatch):
        raise ETTRPacketIndexError("ETTR packet-index batch type differs")
    rows = int(batch.episodes.world.tokens.shape[0])
    if rows < 1:
        raise ETTRPacketIndexError("ETTR packet-index batch is empty")
    compact_rows = []
    for row in range(rows):
        context_value, target = terminal_packet_query_context(batch, row)
        if not 0 <= target <= 0xFFFF:
            raise ETTRPacketIndexError(
                "ETTR packet-index target exceeds uint16"
            )
        compact_rows.append(
            (
                hashlib.sha256(
                    _canonical_json_bytes(context_value)
                ).digest(),
                target,
            )
        )
    compact = ETTRCompactPacketBatch(
        payload_digest=bytes.fromhex(
            continuation_batch_payload_sha256(batch)
        ),
        rows=tuple(compact_rows),
    )
    compact.validate()
    return compact


def _compact_batches(
    batches: Iterable[ETTRContinuationBatch],
) -> Iterator[ETTRCompactPacketBatch]:
    for batch in batches:
        yield compact_packet_batch(batch)


def _insert_split(
    connection: sqlite3.Connection,
    batches: Iterable[ETTRCompactPacketBatch],
    *,
    split: int,
) -> tuple[int, int]:
    batch_table = "train_batches" if split == 0 else "validation_batches"
    batch_count = 0
    row_count = 0
    for batch in batches:
        if not isinstance(batch, ETTRCompactPacketBatch):
            raise ETTRPacketIndexError("compact packet batch type differs")
        batch.validate()
        payload_digest = batch.payload_digest
        try:
            connection.execute(
                f"INSERT INTO {batch_table}(digest) VALUES (?)",
                (payload_digest,),
            )
        except sqlite3.IntegrityError as exc:
            raise ETTRPacketIndexError(
                "ETTR packet index contains a duplicate batch payload"
            ) from exc
        other = (
            "validation_batches"
            if split == 0
            else "train_batches"
        )
        if connection.execute(
            f"SELECT 1 FROM {other} WHERE digest = ?",
            (payload_digest,),
        ).fetchone():
            raise ETTRPacketIndexError(
                "ETTR packet index train/validation batches overlap"
            )
        batch_count += 1
        for context, target in batch.rows:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO contexts(
                    digest, target, split, commitment
                ) VALUES (?, ?, ?, ?)
                """,
                (context, target, split, _target_commitment(context, target)),
            )
            if cursor.rowcount == 0:
                prior = connection.execute(
                    "SELECT target, split FROM contexts WHERE digest = ?",
                    (context,),
                ).fetchone()
                if prior is None or prior[0] != target:
                    raise ETTRPacketIndexError(
                        "ETTR packet/query context maps to multiple targets"
                    )
                if prior[1] != split:
                    raise ETTRPacketIndexError(
                        "ETTR packet-index train/validation contexts overlap"
                    )
            row_count += 1
    return batch_count, row_count


def _query_bytes(
    connection: sqlite3.Connection,
    query: str,
) -> Iterator[bytes]:
    for (value,) in connection.execute(query):
        if not isinstance(value, bytes):
            raise ETTRPacketIndexError("ETTR packet-index database differs")
        yield value


def _build_disk_packet_index_from_compact(
    output: Path,
    *,
    train_batches: Iterable[ETTRCompactPacketBatch],
    validation_batches: Iterable[ETTRCompactPacketBatch],
) -> dict[str, object]:
    """Build one immutable no-replace index from compact batch streams."""

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise ETTRPacketIndexError(
            f"refusing existing packet-index output: {output}"
        ) from exc
    database = output / "build.sqlite3"
    connection = sqlite3.connect(database)
    try:
        connection.executescript(
            """
            PRAGMA temp_store = FILE;
            PRAGMA synchronous = FULL;
            CREATE TABLE contexts(
                digest BLOB PRIMARY KEY,
                target INTEGER NOT NULL,
                split INTEGER NOT NULL,
                commitment BLOB NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE train_batches(
                digest BLOB PRIMARY KEY
            ) WITHOUT ROWID;
            CREATE TABLE validation_batches(
                digest BLOB PRIMARY KEY
            ) WITHOUT ROWID;
            """
        )
        train_count, train_rows = _insert_split(
            connection,
            train_batches,
            split=0,
        )
        validation_count, validation_rows = _insert_split(
            connection,
            validation_batches,
            split=1,
        )
        if train_count < 1 or train_rows < 1:
            raise ETTRPacketIndexError("ETTR packet index has no training data")
        connection.commit()

        context_count = int(
            connection.execute("SELECT COUNT(*) FROM contexts").fetchone()[0]
        )
        train_contexts = int(
            connection.execute(
                "SELECT COUNT(*) FROM contexts WHERE split = 0"
            ).fetchone()[0]
        )
        validation_contexts = int(
            connection.execute(
                "SELECT COUNT(*) FROM contexts WHERE split = 1"
            ).fetchone()[0]
        )
        context_sha256 = _json_string_list_sha256(
            _query_bytes(
                connection,
                "SELECT digest FROM contexts ORDER BY digest",
            )
        )
        target_bound_sha256 = _json_string_list_sha256(
            _query_bytes(
                connection,
                "SELECT commitment FROM contexts ORDER BY commitment",
            )
        )
        train_payload_sha256 = _json_string_list_sha256(
            _query_bytes(
                connection,
                "SELECT digest FROM train_batches ORDER BY digest",
            )
        )
        validation_payload_sha256 = _json_string_list_sha256(
            _query_bytes(
                connection,
                "SELECT digest FROM validation_batches ORDER BY digest",
            )
        )

        contexts_file = _write_no_replace(
            output / _FILES["contexts"],
            (
                bytes(digest) + struct.pack(">H", int(target))
                for digest, target in connection.execute(
                    "SELECT digest, target FROM contexts ORDER BY digest"
                )
            ),
        )
        train_file = _write_no_replace(
            output / _FILES["train_batches"],
            _query_bytes(
                connection,
                "SELECT digest FROM train_batches ORDER BY digest",
            ),
        )
        validation_file = _write_no_replace(
            output / _FILES["validation_batches"],
            _query_bytes(
                connection,
                "SELECT digest FROM validation_batches ORDER BY digest",
            ),
        )
    except BaseException:
        connection.close()
        raise
    else:
        connection.close()
    database.unlink()

    receipt = ETTRPacketSufficiencyReceipt(
        schema=ETTR_PACKET_SUFFICIENCY_SCHEMA,
        batches=train_count + validation_count,
        rows=train_rows + validation_rows,
        unique_contexts=context_count,
        context_sha256=context_sha256,
        target_bound_sha256=target_bound_sha256,
    )
    manifest: dict[str, object] = {
        "context_record_bytes": CONTEXT_RECORD_BYTES,
        "files": {
            "contexts": contexts_file,
            "train_batches": train_file,
            "validation_batches": validation_file,
        },
        "receipt": {
            "batches": receipt.batches,
            "context_sha256": receipt.context_sha256,
            "rows": receipt.rows,
            "schema": receipt.schema,
            "target_bound_sha256": receipt.target_bound_sha256,
            "unique_contexts": receipt.unique_contexts,
        },
        "schema": DISK_INDEX_SCHEMA,
        "train_batches": train_count,
        "train_contexts": train_contexts,
        "train_payload_sha256": train_payload_sha256,
        "train_rows": train_rows,
        "validation_batches": validation_count,
        "validation_contexts": validation_contexts,
        "validation_payload_sha256": validation_payload_sha256,
        "validation_rows": validation_rows,
    }
    manifest["manifest_payload_sha256"] = hashlib.sha256(
        _canonical_json_bytes(manifest)
    ).hexdigest()
    _write_no_replace(
        output / "manifest.json",
        (_canonical_file_bytes(manifest),),
    )
    for path in output.iterdir():
        path.chmod(0o400)
    output.chmod(0o500)
    return manifest


def build_disk_packet_index(
    output: Path,
    *,
    train_batches: Iterable[ETTRContinuationBatch],
    validation_batches: Iterable[ETTRContinuationBatch],
) -> dict[str, object]:
    """Build one immutable no-replace index from single-pass batch streams."""

    return _build_disk_packet_index_from_compact(
        output,
        train_batches=_compact_batches(train_batches),
        validation_batches=_compact_batches(validation_batches),
    )


def build_disk_packet_index_from_compact(
    output: Path,
    *,
    train_batches: Iterable[ETTRCompactPacketBatch],
    validation_batches: Iterable[ETTRCompactPacketBatch],
) -> dict[str, object]:
    """Build the same index from independently computed commitments."""

    return _build_disk_packet_index_from_compact(
        output,
        train_batches=train_batches,
        validation_batches=validation_batches,
    )


def _strict_manifest(path: Path) -> Mapping[str, object]:
    before = _identity(path, "packet-index manifest", immutable=True)
    payload = path.read_bytes()
    after = _identity(path, "packet-index manifest", immutable=True)
    if before != after:
        raise ETTRPacketIndexError(
            "packet-index manifest changed while being read"
        )
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ETTRPacketIndexError("packet-index manifest is malformed") from exc
    if (
        not isinstance(value, dict)
        or _canonical_file_bytes(value) != payload
        or value.get("schema") != DISK_INDEX_SCHEMA
    ):
        raise ETTRPacketIndexError(
            "packet-index manifest is not canonical"
        )
    claimed = value.get("manifest_payload_sha256")
    unsigned = dict(value)
    unsigned.pop("manifest_payload_sha256", None)
    if (
        not isinstance(claimed, str)
        or _HEX64.fullmatch(claimed) is None
        or hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
        != claimed
    ):
        raise ETTRPacketIndexError(
            "packet-index manifest self-hash differs"
        )
    return value


def _file_from_receipt(
    root: Path,
    value: object,
    *,
    expected_name: str,
) -> Path:
    if (
        not isinstance(value, Mapping)
        or value.get("path") != expected_name
        or not isinstance(value.get("bytes"), int)
        or value["bytes"] < 0
        or not isinstance(value.get("sha256"), str)
        or _HEX64.fullmatch(str(value["sha256"])) is None
    ):
        raise ETTRPacketIndexError(
            "packet-index file receipt differs"
        )
    path = root / expected_name
    before = _identity(path, expected_name, immutable=True)
    digest, size = _sha256_file(path)
    after = _identity(path, expected_name, immutable=True)
    if (
        before != after
        or size != value["bytes"]
        or digest != value["sha256"]
    ):
        raise ETTRPacketIndexError(
            f"packet-index file identity differs: {expected_name}"
        )
    return path


def _require_integer(value: object, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ETTRPacketIndexError(f"{label} differs")
    return value


class ETTRDiskPacketSufficiencyIndex:
    """Read-only mmap implementation of the train-step verifier protocol."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        value = _strict_manifest(self.root / "manifest.json")
        files = value.get("files")
        receipt_value = value.get("receipt")
        if not isinstance(files, Mapping) or not isinstance(
            receipt_value,
            Mapping,
        ):
            raise ETTRPacketIndexError("packet-index manifest fields differ")
        context_path = _file_from_receipt(
            self.root,
            files.get("contexts"),
            expected_name=_FILES["contexts"],
        )
        train_path = _file_from_receipt(
            self.root,
            files.get("train_batches"),
            expected_name=_FILES["train_batches"],
        )
        validation_path = _file_from_receipt(
            self.root,
            files.get("validation_batches"),
            expected_name=_FILES["validation_batches"],
        )
        if value.get("context_record_bytes") != CONTEXT_RECORD_BYTES:
            raise ETTRPacketIndexError(
                "packet-index context record width differs"
            )
        self.train_batches = _require_integer(
            value.get("train_batches"),
            "packet-index train batches",
            minimum=1,
        )
        self.validation_batches = _require_integer(
            value.get("validation_batches"),
            "packet-index validation batches",
        )
        self.train_rows = _require_integer(
            value.get("train_rows"),
            "packet-index train rows",
            minimum=1,
        )
        self.validation_rows = _require_integer(
            value.get("validation_rows"),
            "packet-index validation rows",
        )
        self._train_contexts = _require_integer(
            value.get("train_contexts"),
            "packet-index train contexts",
            minimum=1,
        )
        self._validation_contexts = _require_integer(
            value.get("validation_contexts"),
            "packet-index validation contexts",
        )
        self.train_payload_sha256 = str(
            value.get("train_payload_sha256")
        )
        self.validation_payload_sha256 = str(
            value.get("validation_payload_sha256")
        )
        if (
            _HEX64.fullmatch(self.train_payload_sha256) is None
            or _HEX64.fullmatch(self.validation_payload_sha256) is None
        ):
            raise ETTRPacketIndexError(
                "packet-index split payload hash differs"
            )
        try:
            self.receipt = ETTRPacketSufficiencyReceipt(
                schema=str(receipt_value["schema"]),
                batches=_require_integer(
                    receipt_value["batches"],
                    "packet-index receipt batches",
                    minimum=1,
                ),
                rows=_require_integer(
                    receipt_value["rows"],
                    "packet-index receipt rows",
                    minimum=1,
                ),
                unique_contexts=_require_integer(
                    receipt_value["unique_contexts"],
                    "packet-index receipt contexts",
                    minimum=1,
                ),
                context_sha256=str(receipt_value["context_sha256"]),
                target_bound_sha256=str(
                    receipt_value["target_bound_sha256"]
                ),
            )
        except (KeyError, TheoryReactorError) as exc:
            raise ETTRPacketIndexError(
                "packet-index receipt differs"
            ) from exc
        if (
            self.receipt.batches
            != self.train_batches + self.validation_batches
            or self.receipt.rows != self.train_rows + self.validation_rows
            or self.receipt.unique_contexts
            != self.train_contexts + self.validation_contexts
        ):
            raise ETTRPacketIndexError(
                "packet-index manifest counts do not reconcile"
            )

        self._handles = [
            context_path.open("rb"),
            train_path.open("rb"),
            validation_path.open("rb"),
        ]
        self._maps: list[mmap.mmap | bytes] = []
        for handle in self._handles:
            self._maps.append(
                mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
                if os.fstat(handle.fileno()).st_size
                else b""
            )
        self._contexts, self._train, self._validation = self._maps
        if (
            len(self._contexts) != self.receipt.unique_contexts * CONTEXT_RECORD_BYTES
            or len(self._train) != self.train_batches * BATCH_RECORD_BYTES
            or len(self._validation)
            != self.validation_batches * BATCH_RECORD_BYTES
        ):
            self.close()
            raise ETTRPacketIndexError(
                "packet-index fixed-width file size differs"
            )
        self._require_sorted_unique(
            self._contexts,
            CONTEXT_RECORD_BYTES,
            "contexts",
        )
        self._require_sorted_unique(
            self._train,
            BATCH_RECORD_BYTES,
            "train batches",
        )
        self._require_sorted_unique(
            self._validation,
            BATCH_RECORD_BYTES,
            "validation batches",
        )

    @property
    def train_contexts(self) -> int:
        return self._train_contexts

    @property
    def validation_contexts(self) -> int:
        return self._validation_contexts

    @staticmethod
    def _require_sorted_unique(
        data: mmap.mmap | bytes,
        width: int,
        label: str,
    ) -> None:
        prior: bytes | None = None
        for offset in range(0, len(data), width):
            current = bytes(data[offset : offset + 32])
            if prior is not None and current <= prior:
                raise ETTRPacketIndexError(
                    f"packet-index {label} are not sorted and unique"
                )
            prior = current

    @staticmethod
    def _find(
        data: mmap.mmap | bytes,
        width: int,
        key: bytes,
    ) -> int | None:
        lower = 0
        upper = len(data) // width
        while lower < upper:
            middle = (lower + upper) // 2
            offset = middle * width
            candidate = data[offset : offset + 32]
            if candidate < key:
                lower = middle + 1
            else:
                upper = middle
        offset = lower * width
        return (
            offset
            if lower < len(data) // width
            and data[offset : offset + 32] == key
            else None
        )

    def _verify(
        self,
        batches: Sequence[ETTRContinuationBatch],
        batch_index: mmap.mmap | bytes,
        split: str,
    ) -> None:
        frozen = tuple(batches)
        if not frozen or any(
            not isinstance(batch, ETTRContinuationBatch)
            for batch in frozen
        ):
            raise TheoryReactorError(
                "ETTR terminal-packet sufficiency sequence differs"
            )
        for batch in frozen:
            payload = bytes.fromhex(
                continuation_batch_payload_sha256(batch)
            )
            if self._find(batch_index, BATCH_RECORD_BYTES, payload) is None:
                raise TheoryReactorError(
                    f"ETTR batch is absent from the frozen {split} "
                    "payload index"
                )
            rows = int(batch.episodes.world.tokens.shape[0])
            for row in range(rows):
                context_value, target = terminal_packet_query_context(
                    batch,
                    row,
                )
                context = hashlib.sha256(
                    _canonical_json_bytes(context_value)
                ).digest()
                offset = self._find(
                    self._contexts,
                    CONTEXT_RECORD_BYTES,
                    context,
                )
                if (
                    offset is None
                    or struct.unpack(
                        ">H",
                        self._contexts[offset + 32 : offset + 34],
                    )[0]
                    != target
                ):
                    raise TheoryReactorError(
                        f"ETTR batch is absent from the frozen {split} "
                        "packet sufficiency index"
                    )

    def verify_train(
        self,
        batches: Sequence[ETTRContinuationBatch],
    ) -> None:
        self._verify(batches, self._train, "train")

    def verify_validation(
        self,
        batches: Sequence[ETTRContinuationBatch],
    ) -> None:
        self._verify(batches, self._validation, "validation")

    def close(self) -> None:
        for value in getattr(self, "_maps", ()):
            if isinstance(value, mmap.mmap):
                value.close()
        for handle in getattr(self, "_handles", ()):
            handle.close()
        self._maps = []
        self._handles = []

    def __enter__(self) -> "ETTRDiskPacketSufficiencyIndex":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "DISK_INDEX_SCHEMA",
    "ETTRDiskPacketSufficiencyIndex",
    "ETTRPacketIndexError",
    "build_disk_packet_index",
]
