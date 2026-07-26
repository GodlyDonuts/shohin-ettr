"""Canonical raw-stage tokenization receipts for ETTR qualification.

This module is deliberately standalone from execution custody.  It binds the
frozen qualification board's raw WORLD, COMMAND, and selected late QUERY
bytes to one exact immutable tokenizer JSON file and one explicit encoding
configuration.  Validation rebuilds the complete receipt from those primary
inputs instead of trusting caller-supplied token rows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Literal

from tokenizers import Tokenizer

from ettr_factorial_qualification_board import (
    ETTRFactorialQualificationBoard,
    QualificationRow,
    TOTAL_PACKETS,
)


TOKENIZATION_RECEIPT_SCHEMA = "ettr-factorial-tokenization-receipt-v2"
TOKENIZATION_ROW_SCHEMA = "ettr-factorial-tokenization-row-v1"
ENCODING_SCHEMA = "ettr-factorial-raw-stage-encoding-v1"
SELECTED_QUERY_SEMANTIC_INDEX = 0
SELECTED_QUERY_PARAPHRASE_INDEX = 0
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STAGES = ("world", "command", "query")


class ETTRFactorialTokenizationError(ValueError):
    """The raw-stage tokenization custody contract failed."""


def canonical_json_bytes(value: object) -> bytes:
    """Encode one deterministic, finite, ASCII JSON document."""

    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ETTRFactorialTokenizationError(
            "tokenization receipt is not canonicalizable"
        ) from exc


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(_read_immutable_bytes(path))


def immutable_regular(path: Path) -> None:
    """Reject writable files, links, and non-regular filesystem objects."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ETTRFactorialTokenizationError(
            f"tokenization input is unavailable: {path}"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o222
    ):
        raise ETTRFactorialTokenizationError(
            f"tokenization input is not immutable: {path}"
        )


def _read_immutable_bytes(path: Path) -> bytes:
    immutable_regular(path)
    before = path.lstat()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ETTRFactorialTokenizationError(
            f"tokenization input cannot be opened: {path}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_mode & 0o222
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise ETTRFactorialTokenizationError(
                f"tokenization input changed before read: {path}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        after.st_mode != opened.st_mode
        or after.st_size != opened.st_size
        or after.st_mtime_ns != opened.st_mtime_ns
        or after.st_ctime_ns != opened.st_ctime_ns
        or sum(len(chunk) for chunk in chunks) != opened.st_size
    ):
        raise ETTRFactorialTokenizationError(
            f"tokenization input changed during read: {path}"
        )
    return b"".join(chunks)


def _read_canonical_json(path: Path) -> dict[str, object]:
    payload = _read_immutable_bytes(path)
    try:
        value = json.loads(payload.decode("ascii", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ETTRFactorialTokenizationError(
            f"tokenization receipt is malformed: {path}"
        ) from exc
    if not isinstance(value, dict) or payload != canonical_json_bytes(value):
        raise ETTRFactorialTokenizationError(
            f"tokenization receipt is not canonical: {path}"
        )
    return value


def _write_once(path: Path, payload: bytes) -> str:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise ETTRFactorialTokenizationError(
            "tokenization receipt path already exists"
        ) from exc
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ETTRFactorialTokenizationError(
                    "tokenization receipt write was short"
                )
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    path.chmod(0o444)
    return sha256_bytes(payload)


@dataclass(frozen=True, slots=True)
class ETTRFactorialTokenizationRow:
    """One ordered packet-stage raw/token binding."""

    schema: str
    stage: Literal["world", "command", "query"]
    packet_index: int
    packet_factor_id: str
    world_factor_id: str
    command_factor_id: str
    source_row_id: str
    raw_sha256: str
    raw_hex: str
    unpadded_length: int
    token_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ETTRFactorialTokenizationReceipt:
    """Complete canonical raw-package-to-token transformation receipt."""

    schema: str
    encoding_schema: str
    board_sha256: str
    world_package_sha256: str
    command_package_sha256: str
    query_package_sha256: str
    tokenizer_sha256: str
    tokenizer_byte_count: int
    tokenizer_vocab_size: int
    text_encoding: str
    add_special_tokens: bool
    padding_side: str
    pad_token_id: int
    seq_len: int
    selected_query_semantic_index: int
    selected_query_paraphrase_index: int
    packet_factor_ids: tuple[str, ...]
    world_rows: tuple[ETTRFactorialTokenizationRow, ...]
    command_rows: tuple[ETTRFactorialTokenizationRow, ...]
    query_rows: tuple[ETTRFactorialTokenizationRow, ...]
    qualification_query_rows: tuple[ETTRFactorialTokenizationRow, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes())

    def stage_payload(
        self,
        stage: Literal["world", "command", "query"],
    ) -> dict[str, list[list[int]]]:
        """Return the exact JSON object consumed by the detached stage runners."""

        if stage == "world":
            rows = self.world_rows
        elif stage == "command":
            rows = self.command_rows
        elif stage == "query":
            rows = self.query_rows
        else:
            raise ETTRFactorialTokenizationError(
                f"unknown tokenization stage: {stage}"
            )
        return {
            "attention_mask": [list(row.attention_mask) for row in rows],
            "token_ids": [list(row.token_ids) for row in rows],
        }

    def world_stage_payload(self) -> dict[str, list[list[int]]]:
        return self.stage_payload("world")

    def command_stage_payload(self) -> dict[str, list[list[int]]]:
        return self.stage_payload("command")

    def query_stage_payload(self) -> dict[str, list[list[int]]]:
        return self.stage_payload("query")

    def stage_payload_bytes(
        self,
        stage: Literal["world", "command", "query"],
    ) -> bytes:
        return canonical_json_bytes(self.stage_payload(stage))

    def write_once(self, path: Path) -> str:
        return _write_once(path, self.canonical_bytes())

    def validate(
        self,
        board: ETTRFactorialQualificationBoard,
        tokenizer_path: Path,
        *,
        expected_receipt_sha256: str,
    ) -> None:
        """Recompute from primary inputs and reject any receipt divergence."""

        if (
            _SHA256.fullmatch(expected_receipt_sha256) is None
            or self.sha256() != expected_receipt_sha256
        ):
            raise ETTRFactorialTokenizationError(
                "tokenization receipt hash differs"
            )
        rebuilt = build_ettr_factorial_tokenization_receipt(
            board,
            tokenizer_path,
            seq_len=self.seq_len,
            pad_token_id=self.pad_token_id,
        )
        if self != rebuilt:
            raise ETTRFactorialTokenizationError(
                "tokenization receipt differs from raw stages"
            )

    @classmethod
    def from_dict(
        cls,
        value: dict[str, object],
    ) -> ETTRFactorialTokenizationReceipt:
        expected = {
            "schema",
            "encoding_schema",
            "board_sha256",
            "world_package_sha256",
            "command_package_sha256",
            "query_package_sha256",
            "tokenizer_sha256",
            "tokenizer_byte_count",
            "tokenizer_vocab_size",
            "text_encoding",
            "add_special_tokens",
            "padding_side",
            "pad_token_id",
            "seq_len",
            "selected_query_semantic_index",
            "selected_query_paraphrase_index",
            "packet_factor_ids",
            "world_rows",
            "command_rows",
            "query_rows",
            "qualification_query_rows",
        }
        if set(value) != expected:
            raise ETTRFactorialTokenizationError(
                "tokenization receipt keys differ"
            )
        converted = dict(value)
        try:
            converted["packet_factor_ids"] = tuple(value["packet_factor_ids"])
            for name in (
                "world_rows",
                "command_rows",
                "query_rows",
                "qualification_query_rows",
            ):
                raw_rows = value[name]
                if not isinstance(raw_rows, list):
                    raise TypeError
                rows = []
                for raw_row in raw_rows:
                    if not isinstance(raw_row, dict):
                        raise TypeError
                    row = dict(raw_row)
                    row["token_ids"] = tuple(row["token_ids"])
                    row["attention_mask"] = tuple(row["attention_mask"])
                    rows.append(ETTRFactorialTokenizationRow(**row))
                converted[name] = tuple(rows)
            return cls(**converted)
        except (KeyError, TypeError, ValueError) as exc:
            raise ETTRFactorialTokenizationError(
                "tokenization receipt values differ"
            ) from exc

    @classmethod
    def from_path(cls, path: Path) -> ETTRFactorialTokenizationReceipt:
        return cls.from_dict(_read_canonical_json(path))


def _validate_board_packages(
    board: ETTRFactorialQualificationBoard,
) -> None:
    receipt = board.receipt
    if (
        not receipt.all_contracts_pass
        or receipt.packet_count != TOTAL_PACKETS
        or len(board.packet_factor_ids) != TOTAL_PACKETS
        or len(set(board.packet_factor_ids)) != TOTAL_PACKETS
        or sha256_bytes(board.world_package_bytes())
        != receipt.world_package_sha256
        or sha256_bytes(board.command_package_bytes())
        != receipt.command_package_sha256
        or sha256_bytes(board.query_package_bytes())
        != receipt.query_package_sha256
    ):
        raise ETTRFactorialTokenizationError(
            "qualification board package custody differs"
        )


def _load_tokenizer(
    tokenizer_path: Path,
) -> tuple[Tokenizer, bytes, str]:
    tokenizer_bytes = _read_immutable_bytes(tokenizer_path)
    try:
        tokenizer_text = tokenizer_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ETTRFactorialTokenizationError(
            "tokenizer JSON is not strict UTF-8"
        ) from exc
    try:
        tokenizer = Tokenizer.from_str(tokenizer_text)
    except Exception as exc:  # noqa: BLE001 - tokenizers raises a generic exception.
        raise ETTRFactorialTokenizationError(
            "tokenizer JSON cannot be loaded"
        ) from exc
    return tokenizer, tokenizer_bytes, sha256_bytes(tokenizer_bytes)


def _validate_encoding_config(
    tokenizer: Tokenizer,
    *,
    seq_len: int,
    pad_token_id: int,
) -> int:
    if (
        not isinstance(seq_len, int)
        or isinstance(seq_len, bool)
        or seq_len <= 0
        or not isinstance(pad_token_id, int)
        or isinstance(pad_token_id, bool)
        or pad_token_id < 0
    ):
        raise ETTRFactorialTokenizationError(
            "tokenization encoding configuration differs"
        )
    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if (
        pad_token_id >= vocab_size
        or tokenizer.id_to_token(pad_token_id) is None
    ):
        raise ETTRFactorialTokenizationError(
            "tokenization pad token is outside the tokenizer vocabulary"
        )
    return vocab_size


def _encode_row(
    *,
    stage: Literal["world", "command", "query"],
    packet_index: int,
    row: QualificationRow,
    raw_bytes: bytes,
    tokenizer: Tokenizer,
    seq_len: int,
    pad_token_id: int,
) -> ETTRFactorialTokenizationRow:
    try:
        text = raw_bytes.decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ETTRFactorialTokenizationError(
            f"{stage} raw bytes are not strict ASCII"
        ) from exc
    encoded = tokenizer.encode(text, add_special_tokens=False)
    token_ids = tuple(encoded.ids)
    if (
        not token_ids
        or len(token_ids) > seq_len
        or any(
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            for token_id in token_ids
        )
    ):
        raise ETTRFactorialTokenizationError(
            f"{stage} tokenization does not fit the frozen sequence"
        )
    padding = seq_len - len(token_ids)
    padded_ids = (*token_ids, *((pad_token_id,) * padding))
    attention_mask = (*((1,) * len(token_ids)), *((0,) * padding))
    return ETTRFactorialTokenizationRow(
        schema=TOKENIZATION_ROW_SCHEMA,
        stage=stage,
        packet_index=packet_index,
        packet_factor_id=row.packet_factor_id,
        world_factor_id=row.world_factor_id,
        command_factor_id=row.command_factor_id,
        source_row_id=row.row_id,
        raw_sha256=sha256_bytes(raw_bytes),
        raw_hex=raw_bytes.hex(),
        unpadded_length=len(token_ids),
        token_ids=padded_ids,
        attention_mask=attention_mask,
    )


def build_ettr_factorial_tokenization_receipt(
    board: ETTRFactorialQualificationBoard,
    tokenizer_path: Path,
    *,
    seq_len: int,
    pad_token_id: int,
) -> ETTRFactorialTokenizationReceipt:
    """Tokenize the exact frozen stages in canonical packet order."""

    _validate_board_packages(board)
    tokenizer, tokenizer_bytes, tokenizer_sha256 = _load_tokenizer(
        tokenizer_path
    )
    tokenizer_vocab_size = _validate_encoding_config(
        tokenizer,
        seq_len=seq_len,
        pad_token_id=pad_token_id,
    )
    selected = tuple(
        row
        for row in board.rows
        if row.semantic_index == SELECTED_QUERY_SEMANTIC_INDEX
        and row.paraphrase_index == SELECTED_QUERY_PARAPHRASE_INDEX
    )
    packet_factor_ids = board.packet_factor_ids
    if (
        len(selected) != TOTAL_PACKETS
        or tuple(row.packet_factor_id for row in selected)
        != packet_factor_ids
    ):
        raise ETTRFactorialTokenizationError(
            "selected late query packet order differs"
        )

    world_rows = tuple(
        _encode_row(
            stage="world",
            packet_index=packet_index,
            row=row,
            raw_bytes=row.world_bytes,
            tokenizer=tokenizer,
            seq_len=seq_len,
            pad_token_id=pad_token_id,
        )
        for packet_index, row in enumerate(selected)
    )
    command_rows = tuple(
        _encode_row(
            stage="command",
            packet_index=packet_index,
            row=row,
            raw_bytes=row.command_bytes,
            tokenizer=tokenizer,
            seq_len=seq_len,
            pad_token_id=pad_token_id,
        )
        for packet_index, row in enumerate(selected)
    )
    query_rows = tuple(
        _encode_row(
            stage="query",
            packet_index=packet_index,
            row=row,
            raw_bytes=row.query_prefix_bytes,
            tokenizer=tokenizer,
            seq_len=seq_len,
            pad_token_id=pad_token_id,
        )
        for packet_index, row in enumerate(selected)
    )
    qualification_query_rows = tuple(
        _encode_row(
            stage="query",
            packet_index=row_index,
            row=row,
            raw_bytes=row.query_prefix_bytes,
            tokenizer=tokenizer,
            seq_len=seq_len,
            pad_token_id=pad_token_id,
        )
        for row_index, row in enumerate(board.rows)
    )
    return ETTRFactorialTokenizationReceipt(
        schema=TOKENIZATION_RECEIPT_SCHEMA,
        encoding_schema=ENCODING_SCHEMA,
        board_sha256=board.receipt.payload_sha256,
        world_package_sha256=board.receipt.world_package_sha256,
        command_package_sha256=board.receipt.command_package_sha256,
        query_package_sha256=board.receipt.query_package_sha256,
        tokenizer_sha256=tokenizer_sha256,
        tokenizer_byte_count=len(tokenizer_bytes),
        tokenizer_vocab_size=tokenizer_vocab_size,
        text_encoding="ascii-strict",
        add_special_tokens=False,
        padding_side="right",
        pad_token_id=pad_token_id,
        seq_len=seq_len,
        selected_query_semantic_index=SELECTED_QUERY_SEMANTIC_INDEX,
        selected_query_paraphrase_index=SELECTED_QUERY_PARAPHRASE_INDEX,
        packet_factor_ids=packet_factor_ids,
        world_rows=world_rows,
        command_rows=command_rows,
        query_rows=query_rows,
        qualification_query_rows=qualification_query_rows,
    )


__all__ = [
    "ENCODING_SCHEMA",
    "ETTRFactorialTokenizationError",
    "ETTRFactorialTokenizationReceipt",
    "ETTRFactorialTokenizationRow",
    "SELECTED_QUERY_PARAPHRASE_INDEX",
    "SELECTED_QUERY_SEMANTIC_INDEX",
    "TOKENIZATION_RECEIPT_SCHEMA",
    "TOKENIZATION_ROW_SCHEMA",
    "build_ettr_factorial_tokenization_receipt",
    "canonical_json_bytes",
    "immutable_regular",
    "sha256_bytes",
    "sha256_file",
]
