"""Immutable non-pickle wire format for source-deleted ETTR state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import stat

from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors

from endogenous_typed_theory_reactor import (
    TheoryReactorConfig,
    TheoryReactorError,
    TypedTheoryState,
    validate_deployed_state,
)


STATE_SCHEMA = "shohin-ettr-source-deleted-state-v2"
STATE_TENSOR_NAMES = frozenset(
    {
        "active",
        "committed",
        "halted",
        "relations",
        "root",
        "type_probabilities",
        "value_probabilities",
    }
)
STATE_METADATA_NAMES = frozenset(
    {
        "config",
        "schema",
        "step",
    }
)


class ETTRStateIOError(ValueError):
    """A state wire or immutable-file invariant failed."""


@dataclass(frozen=True, slots=True)
class ETTRStateReceipt:
    path: str
    sha256: str
    bytes: int
    tensor_names: tuple[str, ...]
    source_bytes_absent: bool
    immutable: bool


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ETTRStateIOError(
            "state metadata is not canonical JSON"
        ) from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def state_bytes(
    state: TypedTheoryState,
    config: TheoryReactorConfig,
) -> bytes:
    validate_deployed_state(state, config)
    tensors = {
        name: getattr(state, name).detach().to("cpu").contiguous()
        for name in sorted(STATE_TENSOR_NAMES)
    }
    metadata = {
        "schema": STATE_SCHEMA,
        "config": _canonical_json(asdict(config)),
        "step": str(state.step),
    }
    if set(metadata) != STATE_METADATA_NAMES:
        raise ETTRStateIOError(
            "state metadata allowlist differs"
        )
    return save_safetensors(tensors, metadata=metadata)


def write_state_once(
    path: Path,
    state: TypedTheoryState,
    config: TheoryReactorConfig,
    *,
    forbidden_source: bytes | None = None,
) -> ETTRStateReceipt:
    payload = state_bytes(state, config)
    if forbidden_source:
        if len(forbidden_source) < 32:
            raise ETTRStateIOError(
                "source sentinel must be at least 32 bytes"
            )
        if forbidden_source in payload:
            raise ETTRStateIOError(
                "source bytes entered the deleted state"
            )
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as exc:
        raise ETTRStateIOError(
            "state path already exists"
        ) from exc
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ETTRStateIOError(
                    "short state write"
                )
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    path.chmod(0o444)
    return ETTRStateReceipt(
        path=str(path),
        sha256=_sha256(payload),
        bytes=len(payload),
        tensor_names=tuple(sorted(STATE_TENSOR_NAMES)),
        source_bytes_absent=(
            forbidden_source is None
            or forbidden_source not in payload
        ),
        immutable=True,
    )


def read_state(
    path: Path,
    config: TheoryReactorConfig,
) -> TypedTheoryState:
    path = path.resolve()
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ETTRStateIOError(
            "state file is absent"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise ETTRStateIOError(
            "state path is not a regular file"
        )
    if metadata.st_mode & 0o222:
        raise ETTRStateIOError(
            "state file is mutable"
        )
    payload = path.read_bytes()
    try:
        tensors = load_safetensors(payload)
    except Exception as exc:
        raise ETTRStateIOError(
            "state safetensors payload differs"
        ) from exc
    if set(tensors) != STATE_TENSOR_NAMES:
        raise ETTRStateIOError(
            "state tensor allowlist differs"
        )
    metadata_payload = _read_safetensors_metadata(payload)
    if set(metadata_payload) != STATE_METADATA_NAMES:
        raise ETTRStateIOError(
            "state metadata keys differ"
        )
    if metadata_payload["schema"] != STATE_SCHEMA:
        raise ETTRStateIOError(
            "state schema differs"
        )
    if metadata_payload["config"] != _canonical_json(asdict(config)):
        raise ETTRStateIOError(
            "state configuration differs"
        )
    try:
        step = int(metadata_payload["step"])
    except ValueError as exc:
        raise ETTRStateIOError(
            "state step differs"
        ) from exc
    state = TypedTheoryState(
        value_probabilities=tensors["value_probabilities"],
        type_probabilities=tensors["type_probabilities"],
        relations=tensors["relations"],
        active=tensors["active"],
        root=tensors["root"],
        committed=tensors["committed"],
        halted=tensors["halted"],
        step=step,
    )
    try:
        validate_deployed_state(state, config)
    except TheoryReactorError as exc:
        raise ETTRStateIOError(
            "state tensor geometry differs"
        ) from exc
    return state


def verify_state_receipt(
    path: Path,
    receipt: ETTRStateReceipt,
    *,
    forbidden_source: bytes | None = None,
) -> None:
    payload = path.resolve().read_bytes()
    if (
        receipt.path != str(path.resolve())
        or receipt.sha256 != _sha256(payload)
        or receipt.bytes != len(payload)
        or receipt.tensor_names != tuple(sorted(STATE_TENSOR_NAMES))
        or path.stat().st_mode & 0o222
    ):
        raise ETTRStateIOError(
            "state receipt differs"
        )
    if forbidden_source and forbidden_source in payload:
        raise ETTRStateIOError(
            "source bytes entered the received state"
        )


def _read_safetensors_metadata(payload: bytes) -> dict[str, str]:
    if len(payload) < 8:
        raise ETTRStateIOError(
            "state header is truncated"
        )
    header_length = int.from_bytes(payload[:8], "little")
    header_end = 8 + header_length
    if header_end > len(payload):
        raise ETTRStateIOError(
            "state header length differs"
        )
    try:
        header = json.loads(payload[8:header_end])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ETTRStateIOError(
            "state header is malformed"
        ) from exc
    metadata = header.get("__metadata__")
    if (
        not isinstance(metadata, dict)
        or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            for key, value in metadata.items()
        )
    ):
        raise ETTRStateIOError(
            "state metadata differs"
        )
    return metadata


__all__ = [
    "ETTRStateIOError",
    "ETTRStateReceipt",
    "STATE_METADATA_NAMES",
    "STATE_SCHEMA",
    "STATE_TENSOR_NAMES",
    "read_state",
    "state_bytes",
    "verify_state_receipt",
    "write_state_once",
]
