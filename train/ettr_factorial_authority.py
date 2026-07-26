"""Offline-root authority for claim-bearing ETTR qualification custody."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


AUTHORITY_SCHEMA = "ettr-factorial-custody-authority-record-v2"
AUTHORIZED_SEAL_SCHEMA = "ettr-factorial-custody-seal-v4"
AUTHORITY_SIGNATURE_DOMAIN = b"shohin-ettr-custody-authority-v2\x00"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEX_32_BYTES = re.compile(r"^[0-9a-f]{64}$")
_HEX_64_BYTES = re.compile(r"^[0-9a-f]{128}$")


class ETTRCustodyAuthorityError(ValueError):
    """The independently anchored authority chain differs."""


def _canonical_json_bytes(value: object) -> bytes:
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


def _read_immutable_single_link(path: Path, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ETTRCustodyAuthorityError(
            f"authority input cannot be opened: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_mode & 0o222
            or not 0 <= before.st_size <= max_bytes
        ):
            raise ETTRCustodyAuthorityError(
                f"authority input is not immutable single-link file: {path}"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise ETTRCustodyAuthorityError(
                    f"authority input was truncated: {path}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ETTRCustodyAuthorityError(f"authority input grew during read: {path}")
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise ETTRCustodyAuthorityError(
                f"authority input changed during read: {path}"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class ETTRCustodyRootTrust:
    public_key_bytes: bytes
    public_key_sha256: str


@dataclass(frozen=True, slots=True)
class ETTRCustodyAuthorityRecord:
    """One root-authorized signer bound to one board and execution manifest."""

    schema: str
    authorized_seal_schema: str
    root_public_key_sha256: str
    custody_public_key_hex: str
    launch_verifier_public_key_hex: str
    launch_verifier_public_key_fingerprint: str
    claim_runtime_verification_receipt_sha256: str
    board_sha256: str
    execution_manifest_sha256: str
    root_signature_hex: str

    def unsigned_payload(self) -> dict[str, object]:
        return asdict(replace(self, root_signature_hex=""))

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(asdict(self))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def load_ettr_custody_root(
    path: Path,
    *,
    pinned_public_key_sha256: str,
) -> ETTRCustodyRootTrust:
    """Load the verifier-owned root only under an external fingerprint pin."""

    public_key_bytes = _read_immutable_single_link(path, max_bytes=32)
    actual_sha256 = hashlib.sha256(public_key_bytes).hexdigest()
    if (
        len(public_key_bytes) != 32
        or _SHA256.fullmatch(pinned_public_key_sha256) is None
        or actual_sha256 != pinned_public_key_sha256
    ):
        raise ETTRCustodyAuthorityError("custody root trust differs")
    try:
        Ed25519PublicKey.from_public_bytes(public_key_bytes)
    except ValueError as exc:
        raise ETTRCustodyAuthorityError("custody root public key differs") from exc
    return ETTRCustodyRootTrust(
        public_key_bytes=public_key_bytes,
        public_key_sha256=actual_sha256,
    )


def make_root_signed_ettr_custody_authority(
    *,
    root_private_key: Ed25519PrivateKey,
    custody_public_key_hex: str,
    launch_verifier_public_key_hex: str,
    claim_runtime_verification_receipt_sha256: str,
    board_sha256: str,
    execution_manifest_sha256: str,
) -> ETTRCustodyAuthorityRecord:
    """Issue one domain-separated authority record with an offline root."""

    root_public_key = root_private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    if (
        _HEX_32_BYTES.fullmatch(custody_public_key_hex) is None
        or _HEX_32_BYTES.fullmatch(launch_verifier_public_key_hex) is None
        or _SHA256.fullmatch(claim_runtime_verification_receipt_sha256) is None
        or _SHA256.fullmatch(board_sha256) is None
        or _SHA256.fullmatch(execution_manifest_sha256) is None
    ):
        raise ETTRCustodyAuthorityError("custody authority input differs")
    custody_public_key = bytes.fromhex(custody_public_key_hex)
    launch_verifier_public_key = bytes.fromhex(launch_verifier_public_key_hex)
    if (
        custody_public_key == root_public_key
        or launch_verifier_public_key == root_public_key
        or launch_verifier_public_key == custody_public_key
    ):
        raise ETTRCustodyAuthorityError("custody authority key roles overlap")
    try:
        Ed25519PublicKey.from_public_bytes(custody_public_key)
        Ed25519PublicKey.from_public_bytes(launch_verifier_public_key)
    except ValueError as exc:
        raise ETTRCustodyAuthorityError("custody authority public key differs") from exc
    unsigned = ETTRCustodyAuthorityRecord(
        schema=AUTHORITY_SCHEMA,
        authorized_seal_schema=AUTHORIZED_SEAL_SCHEMA,
        root_public_key_sha256=hashlib.sha256(root_public_key).hexdigest(),
        custody_public_key_hex=custody_public_key_hex,
        launch_verifier_public_key_hex=launch_verifier_public_key_hex,
        launch_verifier_public_key_fingerprint=hashlib.sha256(
            launch_verifier_public_key
        ).hexdigest(),
        claim_runtime_verification_receipt_sha256=(
            claim_runtime_verification_receipt_sha256
        ),
        board_sha256=board_sha256,
        execution_manifest_sha256=execution_manifest_sha256,
        root_signature_hex="",
    )
    signature = root_private_key.sign(
        AUTHORITY_SIGNATURE_DOMAIN + _canonical_json_bytes(unsigned.unsigned_payload())
    ).hex()
    return replace(unsigned, root_signature_hex=signature)


def write_ettr_custody_authority_once(
    path: Path,
    record: ETTRCustodyAuthorityRecord,
) -> str:
    payload = record.canonical_bytes()
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
    except FileExistsError as exc:
        raise ETTRCustodyAuthorityError(
            "custody authority path already exists"
        ) from exc
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ETTRCustodyAuthorityError("custody authority write was short")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    path.chmod(0o444)
    return hashlib.sha256(payload).hexdigest()


def read_root_signed_ettr_custody_authority(
    path: Path,
    *,
    root_trust: ETTRCustodyRootTrust,
    expected_record_sha256: str,
    expected_board_sha256: str,
    expected_execution_manifest_sha256: str,
) -> ETTRCustodyAuthorityRecord:
    """Authenticate an immutable record under the verifier-owned root."""

    payload = _read_immutable_single_link(path, max_bytes=16 * 1024)
    try:
        value = json.loads(payload.decode("ascii"))
        record = ETTRCustodyAuthorityRecord(**value)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ETTRCustodyAuthorityError(
            "custody authority record is malformed"
        ) from exc
    if (
        payload != record.canonical_bytes()
        or record.schema != AUTHORITY_SCHEMA
        or record.authorized_seal_schema != AUTHORIZED_SEAL_SCHEMA
        or _SHA256.fullmatch(record.root_public_key_sha256) is None
        or _HEX_32_BYTES.fullmatch(record.custody_public_key_hex) is None
        or _HEX_32_BYTES.fullmatch(record.launch_verifier_public_key_hex) is None
        or _SHA256.fullmatch(record.launch_verifier_public_key_fingerprint) is None
        or _SHA256.fullmatch(record.claim_runtime_verification_receipt_sha256) is None
        or _SHA256.fullmatch(record.board_sha256) is None
        or _SHA256.fullmatch(record.execution_manifest_sha256) is None
        or _HEX_64_BYTES.fullmatch(record.root_signature_hex) is None
        or _SHA256.fullmatch(expected_record_sha256) is None
        or record.sha256() != expected_record_sha256
        or record.root_public_key_sha256 != root_trust.public_key_sha256
        or record.board_sha256 != expected_board_sha256
        or record.execution_manifest_sha256 != expected_execution_manifest_sha256
        or bytes.fromhex(record.custody_public_key_hex) == root_trust.public_key_bytes
        or bytes.fromhex(record.launch_verifier_public_key_hex)
        == root_trust.public_key_bytes
        or record.launch_verifier_public_key_hex == record.custody_public_key_hex
        or hashlib.sha256(
            bytes.fromhex(record.launch_verifier_public_key_hex)
        ).hexdigest()
        != record.launch_verifier_public_key_fingerprint
    ):
        raise ETTRCustodyAuthorityError("custody authority record differs")
    try:
        Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(record.launch_verifier_public_key_hex)
        )
        root_public_key = Ed25519PublicKey.from_public_bytes(
            root_trust.public_key_bytes
        )
        root_public_key.verify(
            bytes.fromhex(record.root_signature_hex),
            AUTHORITY_SIGNATURE_DOMAIN
            + _canonical_json_bytes(record.unsigned_payload()),
        )
    except (InvalidSignature, ValueError) as exc:
        raise ETTRCustodyAuthorityError(
            "custody authority root signature differs"
        ) from exc
    return record


__all__ = [
    "AUTHORIZED_SEAL_SCHEMA",
    "AUTHORITY_SCHEMA",
    "ETTRCustodyAuthorityError",
    "ETTRCustodyAuthorityRecord",
    "ETTRCustodyRootTrust",
    "load_ettr_custody_root",
    "make_root_signed_ettr_custody_authority",
    "read_root_signed_ettr_custody_authority",
    "write_ettr_custody_authority_once",
]
