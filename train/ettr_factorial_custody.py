"""Hash-anchored custody records for staged ETTR qualification execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Literal

from endogenous_typed_theory_reactor import TheoryReactorError
from ettr_factorial_qualification_board import (
    ETTRFactorialQualificationBoard,
    TOTAL_PACKETS,
)


EXECUTION_MANIFEST_SCHEMA = "ettr-factorial-execution-manifest-v1"
STAGE_RECEIPT_SCHEMA = "ettr-factorial-stage-execution-receipt-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: object) -> bytes:
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


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def immutable_regular(path: Path) -> None:
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o222
    ):
        raise TheoryReactorError(f"custody input is not immutable: {path}")


def read_canonical_json(path: Path) -> dict[str, object]:
    immutable_regular(path)
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TheoryReactorError(f"custody JSON is malformed: {path}") from exc
    if not isinstance(value, dict) or payload != canonical_json_bytes(value):
        raise TheoryReactorError(f"custody JSON is not canonical: {path}")
    return value


def write_json_once(path: Path, value: object) -> str:
    payload = canonical_json_bytes(value)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise TheoryReactorError("custody receipt path already exists") from exc
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise TheoryReactorError("custody receipt write was short")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    path.chmod(0o444)
    return sha256_bytes(payload)


@dataclass(frozen=True, slots=True)
class ETTRFactorialExecutionManifest:
    """Externally preregistered component and stage-input identity."""

    schema: str
    board_sha256: str
    model_sha256: str
    config_sha256: str
    checkpoint_sha256: str
    checkpoint_step: int
    compiler_sha256: str
    reactor_sha256: str
    world_package_sha256: str
    command_package_sha256: str
    world_tokens_sha256: str
    command_tokens_sha256: str
    row_count: int

    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(asdict(self)))

    def validate_hash(self, expected_manifest_sha256: str) -> None:
        hashes = (
            self.board_sha256,
            self.model_sha256,
            self.config_sha256,
            self.checkpoint_sha256,
            self.compiler_sha256,
            self.reactor_sha256,
            self.world_package_sha256,
            self.command_package_sha256,
            self.world_tokens_sha256,
            self.command_tokens_sha256,
            expected_manifest_sha256,
        )
        if (
            self.schema != EXECUTION_MANIFEST_SCHEMA
            or any(_SHA256.fullmatch(value) is None for value in hashes)
            or self.row_count != TOTAL_PACKETS
            or not isinstance(self.checkpoint_step, int)
            or isinstance(self.checkpoint_step, bool)
            or self.checkpoint_step < 0
            or self.sha256() != expected_manifest_sha256
        ):
            raise TheoryReactorError("factorial execution manifest differs")

    def validate(
        self,
        board: ETTRFactorialQualificationBoard,
        *,
        expected_model_sha256: str,
        expected_manifest_sha256: str,
    ) -> None:
        self.validate_hash(expected_manifest_sha256)
        if (
            _SHA256.fullmatch(expected_model_sha256) is None
            or self.board_sha256 != board.receipt.payload_sha256
            or self.model_sha256 != expected_model_sha256
            or self.world_package_sha256 != board.receipt.world_package_sha256
            or self.command_package_sha256 != board.receipt.command_package_sha256
        ):
            raise TheoryReactorError("factorial execution manifest differs")

    @classmethod
    def from_path(cls, path: Path) -> ETTRFactorialExecutionManifest:
        try:
            return cls(**read_canonical_json(path))
        except TypeError as exc:
            raise TheoryReactorError("execution manifest keys differ") from exc


@dataclass(frozen=True, slots=True)
class ETTRStageExecutionReceipt:
    """One immutable process result in the compiler-to-executor chain."""

    schema: str
    stage: Literal["world", "command"]
    manifest_sha256: str
    parent_receipt_sha256: str | None
    input_state_file_sha256: str | None
    input_state_tensor_sha256: str | None
    token_input_sha256: str
    component_sha256: str
    checkpoint_sha256: str
    output_state_file_sha256: str
    output_state_tensor_sha256: str
    row_count: int

    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(asdict(self)))

    def validate(
        self,
        manifest: ETTRFactorialExecutionManifest,
        *,
        expected_receipt_sha256: str,
    ) -> None:
        nullable_hashes = (
            self.parent_receipt_sha256,
            self.input_state_file_sha256,
            self.input_state_tensor_sha256,
        )
        if (
            self.schema != STAGE_RECEIPT_SCHEMA
            or self.stage not in ("world", "command")
            or _SHA256.fullmatch(self.manifest_sha256) is None
            or any(
                value is not None and _SHA256.fullmatch(value) is None
                for value in nullable_hashes
            )
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.token_input_sha256,
                    self.component_sha256,
                    self.checkpoint_sha256,
                    self.output_state_file_sha256,
                    self.output_state_tensor_sha256,
                    expected_receipt_sha256,
                )
            )
            or self.manifest_sha256 != manifest.sha256()
            or self.checkpoint_sha256 != manifest.checkpoint_sha256
            or self.row_count != manifest.row_count
            or self.sha256() != expected_receipt_sha256
        ):
            raise TheoryReactorError("factorial stage receipt differs")
        if self.stage == "world":
            valid_stage = (
                self.parent_receipt_sha256 is None
                and self.input_state_file_sha256 is None
                and self.input_state_tensor_sha256 is None
                and self.token_input_sha256 == manifest.world_tokens_sha256
                and self.component_sha256 == manifest.compiler_sha256
            )
        else:
            valid_stage = (
                self.parent_receipt_sha256 is not None
                and self.input_state_file_sha256 is not None
                and self.input_state_tensor_sha256 is not None
                and self.token_input_sha256 == manifest.command_tokens_sha256
                and self.component_sha256 == manifest.reactor_sha256
            )
        if not valid_stage:
            raise TheoryReactorError("factorial stage receipt provenance differs")

    @classmethod
    def from_path(cls, path: Path) -> ETTRStageExecutionReceipt:
        try:
            return cls(**read_canonical_json(path))
        except TypeError as exc:
            raise TheoryReactorError("stage receipt keys differ") from exc


__all__ = [
    "ETTRFactorialExecutionManifest",
    "ETTRStageExecutionReceipt",
    "EXECUTION_MANIFEST_SCHEMA",
    "STAGE_RECEIPT_SCHEMA",
    "canonical_json_bytes",
    "immutable_regular",
    "read_canonical_json",
    "sha256_bytes",
    "sha256_file",
    "write_json_once",
]
