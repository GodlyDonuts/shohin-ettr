"""Standalone, recomputable identity receipt for a complete ETTR model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from io import BytesIO
import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Mapping

from safetensors.torch import load as load_safetensors
import torch
from torch import nn

from endogenous_typed_theory_reactor import (
    EndogenousTypedTheoryReactorGPT,
    SYSTEM_PARAMETER_CAP,
    TheoryReactorConfig,
)
from ettr_qualification import _model_sha256
from model import GPT, GPTConfig


ETTR_MODEL_ASSEMBLY_SCHEMA = "shohin-ettr-complete-model-assembly-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ETTRModelAssemblyError(ValueError):
    """A complete-model assembly input or receipt failed validation."""


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _immutable_regular(path: Path) -> stat.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ETTRModelAssemblyError(
            f"assembly input cannot be inspected: {path}"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_mode & 0o222
        or metadata.st_nlink != 1
    ):
        raise ETTRModelAssemblyError(
            f"assembly input is not an immutable regular file: {path}"
        )
    return metadata


def _read_immutable_bytes(path: Path) -> bytes:
    before = _immutable_regular(path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ETTRModelAssemblyError(
            f"assembly input cannot be read: {path}"
        ) from exc
    after = _immutable_regular(path)
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
    if identity_before != identity_after or len(payload) != before.st_size:
        raise ETTRModelAssemblyError(
            f"assembly input changed while being read: {path}"
        )
    return payload


def _read_canonical_config(
    path: Path,
) -> tuple[TheoryReactorConfig, bytes]:
    payload = _read_immutable_bytes(path)
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ETTRModelAssemblyError(
            "TheoryReactor configuration is malformed"
        ) from exc
    if not isinstance(value, dict) or payload != _canonical_json_bytes(value):
        raise ETTRModelAssemblyError(
            "TheoryReactor configuration is not canonical"
        )
    expected_keys = {item.name for item in fields(TheoryReactorConfig)}
    float_keys = {
        "execution_trace_read_scale",
        "open_state_read_floor",
    }
    integer_keys = expected_keys - float_keys
    if (
        set(value) != expected_keys
        or any(
            not isinstance(value[name], int) or isinstance(value[name], bool)
            for name in integer_keys
        )
        or any(not isinstance(value[name], float) for name in float_keys)
    ):
        raise ETTRModelAssemblyError(
            "TheoryReactor configuration fields differ"
        )
    try:
        config = TheoryReactorConfig(**value)
        config.validate()
    except (TypeError, ValueError) as exc:
        raise ETTRModelAssemblyError(
            "TheoryReactor configuration is invalid"
        ) from exc
    if asdict(config) != value:
        raise ETTRModelAssemblyError(
            "TheoryReactor configuration values differ"
        )
    return config, payload


def _load_base_checkpoint(
    payload: bytes,
    *,
    expected_step: int,
) -> GPT:
    try:
        checkpoint = torch.load(
            BytesIO(payload),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ETTRModelAssemblyError(
            "protected base checkpoint cannot be loaded safely"
        ) from exc
    if (
        not isinstance(checkpoint, dict)
        or not isinstance(expected_step, int)
        or isinstance(expected_step, bool)
        or expected_step < 0
        or checkpoint.get("step") != expected_step
        or not isinstance(checkpoint.get("cfg"), dict)
        or not isinstance(checkpoint.get("model"), Mapping)
    ):
        raise ETTRModelAssemblyError(
            "protected base checkpoint contract differs"
        )
    config_payload = checkpoint["cfg"]
    defaults = asdict(GPTConfig())
    if set(config_payload) != set(defaults):
        raise ETTRModelAssemblyError(
            "protected base GPT configuration fields differ"
        )
    for name, default in defaults.items():
        value = config_payload[name]
        if type(value) is not type(default):
            raise ETTRModelAssemblyError(
                "protected base GPT configuration values differ"
            )
    try:
        config = GPTConfig(**config_payload)
        base = GPT(config).eval()
        incompatibility = base.load_state_dict(
            checkpoint["model"],
            strict=True,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ETTRModelAssemblyError(
            "protected base checkpoint strict load differs"
        ) from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ETTRModelAssemblyError(
            "protected base checkpoint strict load differs"
        )
    return base


def _load_component(
    module: nn.Module,
    payload: bytes,
    *,
    name: str,
) -> None:
    try:
        state = load_safetensors(payload)
        incompatibility = module.load_state_dict(state, strict=True)
    except Exception as exc:
        raise ETTRModelAssemblyError(
            f"{name} safetensors strict load differs"
        ) from exc
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise ETTRModelAssemblyError(
            f"{name} safetensors strict load differs"
        )


@dataclass(frozen=True, slots=True)
class ETTRModelAssemblyReceipt:
    """Identity and parameter ledger reconstructed from exact model inputs."""

    schema: str
    config_sha256: str
    checkpoint_sha256: str
    checkpoint_step: int
    compiler_sha256: str
    reactor_sha256: str
    query_reader_sha256: str
    complete_model_sha256: str
    base_parameters: int
    architecture_parameters: int
    total_parameters: int
    parameter_cap: int
    remaining_under_cap: int

    @classmethod
    def build(
        cls,
        *,
        config_path: Path,
        checkpoint_path: Path,
        checkpoint_step: int,
        compiler_path: Path,
        reactor_path: Path,
        query_reader_path: Path,
    ) -> ETTRModelAssemblyReceipt:
        """Build a receipt solely from the complete immutable assembly."""

        return cls.recompute(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            checkpoint_step=checkpoint_step,
            compiler_path=compiler_path,
            reactor_path=reactor_path,
            query_reader_path=query_reader_path,
        )

    @classmethod
    def recompute(
        cls,
        *,
        config_path: Path,
        checkpoint_path: Path,
        checkpoint_step: int,
        compiler_path: Path,
        reactor_path: Path,
        query_reader_path: Path,
    ) -> ETTRModelAssemblyReceipt:
        """Strictly reconstruct the complete model and recompute its identity."""

        config, config_bytes = _read_canonical_config(config_path)
        checkpoint_bytes = _read_immutable_bytes(checkpoint_path)
        compiler_bytes = _read_immutable_bytes(compiler_path)
        reactor_bytes = _read_immutable_bytes(reactor_path)
        query_reader_bytes = _read_immutable_bytes(query_reader_path)

        base = _load_base_checkpoint(
            checkpoint_bytes,
            expected_step=checkpoint_step,
        )
        try:
            model = EndogenousTypedTheoryReactorGPT(base, config)
        except (TypeError, ValueError) as exc:
            raise ETTRModelAssemblyError(
                "complete ETTR model geometry differs"
            ) from exc
        _load_component(
            model.compiler,
            compiler_bytes,
            name="compiler",
        )
        _load_component(
            model.reactor,
            reactor_bytes,
            name="reactor",
        )
        _load_component(
            model.query_reader,
            query_reader_bytes,
            name="query reader",
        )
        model.eval()
        try:
            parameters = model.parameter_receipt()
        except ValueError as exc:
            raise ETTRModelAssemblyError(
                "complete ETTR parameter ledger differs"
            ) from exc
        if (
            parameters.parameter_cap != config.parameter_cap
            or parameters.parameter_cap > SYSTEM_PARAMETER_CAP
            or parameters.complete_system_parameters
            != parameters.base_parameters + parameters.architecture_parameters
            or parameters.complete_system_parameters > SYSTEM_PARAMETER_CAP
            or parameters.complete_system_parameters > parameters.parameter_cap
            or parameters.remaining_under_cap
            != parameters.parameter_cap - parameters.complete_system_parameters
            or sum(value.numel() for value in model.parameters())
            != parameters.complete_system_parameters
        ):
            raise ETTRModelAssemblyError(
                "complete ETTR parameter cap validation failed"
            )
        return cls(
            schema=ETTR_MODEL_ASSEMBLY_SCHEMA,
            config_sha256=_sha256_bytes(config_bytes),
            checkpoint_sha256=_sha256_bytes(checkpoint_bytes),
            checkpoint_step=checkpoint_step,
            compiler_sha256=_sha256_bytes(compiler_bytes),
            reactor_sha256=_sha256_bytes(reactor_bytes),
            query_reader_sha256=_sha256_bytes(query_reader_bytes),
            complete_model_sha256=_model_sha256(model),
            base_parameters=parameters.base_parameters,
            architecture_parameters=parameters.architecture_parameters,
            total_parameters=parameters.complete_system_parameters,
            parameter_cap=parameters.parameter_cap,
            remaining_under_cap=parameters.remaining_under_cap,
        )

    @classmethod
    def from_path(cls, path: Path) -> ETTRModelAssemblyReceipt:
        """Read an immutable canonical receipt without trusting its claims."""

        payload = _read_immutable_bytes(path)
        try:
            value = json.loads(payload.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ETTRModelAssemblyError(
                "model assembly receipt is malformed"
            ) from exc
        if not isinstance(value, dict) or payload != _canonical_json_bytes(value):
            raise ETTRModelAssemblyError(
                "model assembly receipt is not canonical"
            )
        try:
            receipt = cls(**value)
        except TypeError as exc:
            raise ETTRModelAssemblyError(
                "model assembly receipt fields differ"
            ) from exc
        receipt._validate_fields()
        return receipt

    def canonical_bytes(self) -> bytes:
        """Return the unique ASCII JSON encoding used by the receipt hash."""

        return _canonical_json_bytes(asdict(self))

    def sha(self) -> str:
        """Return the SHA-256 of the canonical receipt bytes."""

        return _sha256_bytes(self.canonical_bytes())

    def sha256(self) -> str:
        """Compatibility alias for callers that spell out the hash algorithm."""

        return self.sha()

    def validate(
        self,
        *,
        expected_receipt_sha256: str,
        config_path: Path,
        checkpoint_path: Path,
        compiler_path: Path,
        reactor_path: Path,
        query_reader_path: Path,
    ) -> None:
        """Fail closed unless every claim matches a fresh strict assembly."""

        self._validate_fields()
        if (
            _SHA256.fullmatch(expected_receipt_sha256) is None
            or self.sha() != expected_receipt_sha256
        ):
            raise ETTRModelAssemblyError(
                "model assembly receipt hash differs"
            )
        recomputed = type(self).recompute(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            checkpoint_step=self.checkpoint_step,
            compiler_path=compiler_path,
            reactor_path=reactor_path,
            query_reader_path=query_reader_path,
        )
        if recomputed != self:
            raise ETTRModelAssemblyError(
                "complete model assembly differs from receipt"
            )

    def _validate_fields(self) -> None:
        hashes = (
            self.config_sha256,
            self.checkpoint_sha256,
            self.compiler_sha256,
            self.reactor_sha256,
            self.query_reader_sha256,
            self.complete_model_sha256,
        )
        counts = (
            self.base_parameters,
            self.architecture_parameters,
            self.total_parameters,
            self.parameter_cap,
            self.remaining_under_cap,
        )
        if (
            self.schema != ETTR_MODEL_ASSEMBLY_SCHEMA
            or any(
                not isinstance(value, str) or _SHA256.fullmatch(value) is None
                for value in hashes
            )
            or not isinstance(self.checkpoint_step, int)
            or isinstance(self.checkpoint_step, bool)
            or self.checkpoint_step < 0
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in counts
            )
            or self.base_parameters <= 0
            or self.architecture_parameters <= 0
            or self.parameter_cap > SYSTEM_PARAMETER_CAP
            or self.total_parameters
            != self.base_parameters + self.architecture_parameters
            or self.total_parameters > self.parameter_cap
            or self.total_parameters > SYSTEM_PARAMETER_CAP
            or self.remaining_under_cap
            != self.parameter_cap - self.total_parameters
        ):
            raise ETTRModelAssemblyError(
                "model assembly receipt fields are invalid"
            )


__all__ = [
    "ETTR_MODEL_ASSEMBLY_SCHEMA",
    "ETTRModelAssemblyError",
    "ETTRModelAssemblyReceipt",
]
