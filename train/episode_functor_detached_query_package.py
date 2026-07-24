"""Hash-bound, pickle-free deployment package for the detached query parser."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Mapping

import torch
from safetensors.torch import load, save_file

from episode_functor_query_parser import NeuralOpaqueQueryParser


PACKAGE_SCHEMA = "shohin.efc.detached-query-parser.v1"
AUTHORIZATION_SCHEMA = "shohin.efc.detached-execution.v1"
PROTECTED_SHOHIN_PARAMETERS = 125_081_664
PROTECTED_SHOHIN_SHA256 = (
    "211d6b2cddf0c2cf8b12cb0b2d73f9c4440d85f6f531018080c8afd35b2f66a6"
)
SYSTEM_PARAMETER_LIMIT = 200_000_000
MAX_MANIFEST_BYTES = 1_048_576
MAX_PARSER_PARAMETERS = 5_000_000
MAX_WEIGHTS_BYTES = 25_000_000
_CONFIG_KEYS = frozenset(
    {
        "external_feature_width",
        "feedforward",
        "heads",
        "layers",
        "max_steps",
        "width",
    }
)


class DetachedQueryPackageError(ValueError):
    """The detached parser package or its cryptographic binding is invalid."""


@dataclass(frozen=True, slots=True)
class DetachedQueryParserReceipt:
    schema: str
    manifest_sha256: str
    state_sha256: str
    weights_sha256: str
    weights_bytes: int
    parameter_count: int
    architecture: dict[str, int]


@dataclass(frozen=True, slots=True)
class DetachedExecutionAuthorization:
    schema: str
    machine_sha256: str
    parser_manifest_sha256: str
    parser_state_sha256: str
    parser_parameter_count: int
    source_compiler_parameter_count: int
    source_compiler_state_sha256: str
    protected_shohin_parameters: int
    protected_shohin_sha256: str
    complete_parameter_count: int


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular_bytes(path: Path, *, maximum: int, label: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DetachedQueryPackageError(
            f"deployment {label} cannot be opened safely"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size <= 0
            or metadata.st_size > maximum
        ):
            raise DetachedQueryPackageError(
                f"deployment {label} size or file type differs"
            )
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise DetachedQueryPackageError(
                    f"deployment {label} was truncated"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise DetachedQueryPackageError(
                f"deployment {label} grew during read"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _exclusive_publish(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise DetachedQueryPackageError(
            f"deployment output already exists: {destination.name}"
        ) from exc
    finally:
        source.unlink(missing_ok=True)
    directory = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _state_schema(
    state: Mapping[str, torch.Tensor],
) -> list[dict[str, object]]:
    return [
        {
            "dtype": str(tensor.dtype),
            "name": name,
            "shape": list(tensor.shape),
        }
        for name, tensor in sorted(state.items())
    ]


def module_state_sha256(module: torch.nn.Module) -> str:
    """Hash exact named module state independently of file encoding."""

    if not isinstance(module, torch.nn.Module):
        raise DetachedQueryPackageError("deployment module type differs")
    digest = sha256()
    for name, tensor in sorted(module.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        dtype_bytes = str(contiguous.dtype).encode("ascii")
        digest.update(len(name_bytes).to_bytes(4, "little"))
        digest.update(name_bytes)
        digest.update(len(dtype_bytes).to_bytes(4, "little"))
        digest.update(dtype_bytes)
        digest.update(contiguous.ndim.to_bytes(4, "little"))
        for dimension in contiguous.shape:
            digest.update(int(dimension).to_bytes(8, "little"))
        raw = contiguous.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def detached_query_parser_state_sha256(
    parser: NeuralOpaqueQueryParser,
) -> str:
    """Hash exact named parser state independently of file encoding."""

    if not isinstance(parser, NeuralOpaqueQueryParser):
        raise DetachedQueryPackageError("deployment parser type differs")
    return module_state_sha256(parser)


def export_detached_query_parser_package(
    parser: NeuralOpaqueQueryParser,
    *,
    weights_path: Path,
    manifest_path: Path,
) -> DetachedQueryParserReceipt:
    """Publish exact parser weights and a canonical hash-bound manifest."""

    if not isinstance(parser, NeuralOpaqueQueryParser):
        raise DetachedQueryPackageError("deployment parser type differs")
    if parser.external_feature_width != 0:
        raise DetachedQueryPackageError(
            "deployment parser retains an external feature dependency"
        )
    weights_path = Path(weights_path)
    manifest_path = Path(manifest_path)
    if (
        not weights_path.is_absolute()
        or not manifest_path.is_absolute()
        or weights_path.parent != manifest_path.parent
        or weights_path == manifest_path
        or weights_path.exists()
        or manifest_path.exists()
    ):
        raise DetachedQueryPackageError(
            "deployment paths must be new absolute siblings"
        )
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in parser.state_dict().items()
    }
    temporary = weights_path.with_name(
        f".{weights_path.name}.{secrets.token_hex(16)}.tmp"
    )
    try:
        save_file(state, temporary)
        os.chmod(temporary, 0o600)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _exclusive_publish(temporary, weights_path)
    finally:
        temporary.unlink(missing_ok=True)
    weights_sha256 = _file_sha256(weights_path)
    architecture = parser.architecture_config()
    manifest = {
        "architecture": architecture,
        "parameter_count": parser.parameter_count(),
        "schema": PACKAGE_SCHEMA,
        "state_sha256": detached_query_parser_state_sha256(parser),
        "state_schema": _state_schema(state),
        "weights_bytes": weights_path.stat().st_size,
        "weights_sha256": weights_sha256,
    }
    manifest_payload = _canonical_json(manifest)
    try:
        _write_exclusive(manifest_path, manifest_payload)
    except BaseException:
        weights_path.unlink(missing_ok=True)
        raise
    return DetachedQueryParserReceipt(
        schema=PACKAGE_SCHEMA,
        manifest_sha256=sha256(manifest_payload).hexdigest(),
        state_sha256=manifest["state_sha256"],
        weights_sha256=weights_sha256,
        weights_bytes=weights_path.stat().st_size,
        parameter_count=parser.parameter_count(),
        architecture=architecture,
    )


def load_detached_query_parser_package(
    *,
    weights_path: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> tuple[NeuralOpaqueQueryParser, DetachedQueryParserReceipt]:
    """Verify and load one exact parser package without pickle execution."""

    weights_path = Path(weights_path)
    manifest_path = Path(manifest_path)
    if (
        len(expected_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_manifest_sha256
        )
    ):
        raise DetachedQueryPackageError(
            "deployment package inputs or expected hash are invalid"
        )
    manifest_payload = _read_regular_bytes(
        manifest_path,
        maximum=MAX_MANIFEST_BYTES,
        label="manifest",
    )
    if sha256(manifest_payload).hexdigest() != expected_manifest_sha256:
        raise DetachedQueryPackageError("deployment manifest hash differs")
    try:
        manifest = json.loads(manifest_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DetachedQueryPackageError(
            "deployment manifest JSON is invalid"
        ) from exc
    if (
        not isinstance(manifest, dict)
        or set(manifest)
        != {
            "architecture",
            "parameter_count",
            "schema",
            "state_sha256",
            "state_schema",
            "weights_bytes",
            "weights_sha256",
        }
        or _canonical_json(manifest) != manifest_payload
        or manifest["schema"] != PACKAGE_SCHEMA
        or not isinstance(manifest["architecture"], dict)
        or set(manifest["architecture"]) != _CONFIG_KEYS
        or any(
            type(value) is not int
            for value in manifest["architecture"].values()
        )
        or manifest["architecture"]["external_feature_width"] != 0
        or not 32 <= manifest["architecture"]["width"] <= 320
        or not 1 <= manifest["architecture"]["layers"] <= 8
        or not 1 <= manifest["architecture"]["heads"] <= 16
        or not (
            manifest["architecture"]["width"]
            <= manifest["architecture"]["feedforward"]
            <= 2_048
        )
        or not 1 <= manifest["architecture"]["max_steps"] <= 32
        or type(manifest["parameter_count"]) is not int
        or not 1 <= manifest["parameter_count"] < MAX_PARSER_PARAMETERS
        or type(manifest["weights_bytes"]) is not int
        or not 1 <= manifest["weights_bytes"] <= MAX_WEIGHTS_BYTES
        or not isinstance(manifest["weights_sha256"], str)
        or not isinstance(manifest["state_sha256"], str)
        or len(manifest["state_sha256"]) != 64
        or not isinstance(manifest["state_schema"], list)
    ):
        raise DetachedQueryPackageError(
            "deployment manifest schema or values differ"
        )
    weights_payload = _read_regular_bytes(
        weights_path,
        maximum=MAX_WEIGHTS_BYTES,
        label="weights",
    )
    if (
        len(weights_payload) != manifest["weights_bytes"]
        or sha256(weights_payload).hexdigest()
        != manifest["weights_sha256"]
    ):
        raise DetachedQueryPackageError("deployment parser weights differ")
    try:
        parser = NeuralOpaqueQueryParser(**manifest["architecture"])
        state = load(weights_payload)
    except (OSError, RuntimeError, ValueError) as exc:
        raise DetachedQueryPackageError(
            "deployment parser construction or tensor load failed"
        ) from exc
    if (
        parser.parameter_count() != manifest["parameter_count"]
        or _state_schema(state) != manifest["state_schema"]
        or any(
            tensor.is_floating_point()
            and not bool(torch.isfinite(tensor).all())
            for tensor in state.values()
        )
    ):
        raise DetachedQueryPackageError(
            "deployment parser parameter or tensor schema differs"
        )
    try:
        parser.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise DetachedQueryPackageError(
            "deployment parser state does not load exactly"
        ) from exc
    if (
        detached_query_parser_state_sha256(parser)
        != manifest["state_sha256"]
    ):
        raise DetachedQueryPackageError(
            "deployment parser canonical state hash differs"
        )
    parser.eval()
    receipt = DetachedQueryParserReceipt(
        schema=PACKAGE_SCHEMA,
        manifest_sha256=expected_manifest_sha256,
        state_sha256=manifest["state_sha256"],
        weights_sha256=manifest["weights_sha256"],
        weights_bytes=manifest["weights_bytes"],
        parameter_count=manifest["parameter_count"],
        architecture=dict(manifest["architecture"]),
    )
    return parser, receipt


def receipt_payload(receipt: DetachedQueryParserReceipt) -> dict[str, object]:
    """Return a JSON-ready immutable receipt."""

    return asdict(receipt)


def build_detached_execution_authorization(
    *,
    machine_sha256: str,
    parser_receipt: DetachedQueryParserReceipt,
    source_compiler_parameter_count: int,
    source_compiler_state_sha256: str,
) -> DetachedExecutionAuthorization:
    """Bind one wire and preregistered parser under the aggregate cap."""

    complete = (
        PROTECTED_SHOHIN_PARAMETERS
        + source_compiler_parameter_count
        + parser_receipt.parameter_count
    )
    if (
        len(machine_sha256) != 64
        or any(character not in "0123456789abcdef" for character in machine_sha256)
        or type(source_compiler_parameter_count) is not int
        or source_compiler_parameter_count <= 0
        or len(source_compiler_state_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in source_compiler_state_sha256
        )
        or complete >= SYSTEM_PARAMETER_LIMIT
    ):
        raise DetachedQueryPackageError(
            "detached execution authorization leaves custody or cap"
        )
    return DetachedExecutionAuthorization(
        schema=AUTHORIZATION_SCHEMA,
        machine_sha256=machine_sha256,
        parser_manifest_sha256=parser_receipt.manifest_sha256,
        parser_state_sha256=parser_receipt.state_sha256,
        parser_parameter_count=parser_receipt.parameter_count,
        source_compiler_parameter_count=source_compiler_parameter_count,
        source_compiler_state_sha256=source_compiler_state_sha256,
        protected_shohin_parameters=PROTECTED_SHOHIN_PARAMETERS,
        protected_shohin_sha256=PROTECTED_SHOHIN_SHA256,
        complete_parameter_count=complete,
    )


def export_detached_execution_authorization(
    authorization: DetachedExecutionAuthorization,
    *,
    path: Path,
) -> str:
    """Publish one canonical no-clobber authorization and return its hash."""

    if not isinstance(authorization, DetachedExecutionAuthorization):
        raise DetachedQueryPackageError(
            "detached execution authorization type differs"
        )
    payload = _canonical_json(asdict(authorization))
    _write_exclusive(Path(path), payload)
    return sha256(payload).hexdigest()


def load_detached_execution_authorization(
    *,
    path: Path,
    expected_sha256: str,
) -> DetachedExecutionAuthorization:
    """Load exactly the externally committed authorization bytes."""

    payload = _read_regular_bytes(
        Path(path),
        maximum=MAX_MANIFEST_BYTES,
        label="authorization",
    )
    if sha256(payload).hexdigest() != expected_sha256:
        raise DetachedQueryPackageError(
            "detached execution authorization hash differs"
        )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DetachedQueryPackageError(
            "detached execution authorization JSON is invalid"
        ) from exc
    fields = {
        "complete_parameter_count",
        "machine_sha256",
        "parser_manifest_sha256",
        "parser_parameter_count",
        "parser_state_sha256",
        "protected_shohin_parameters",
        "protected_shohin_sha256",
        "schema",
        "source_compiler_parameter_count",
        "source_compiler_state_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or _canonical_json(value) != payload
        or value["schema"] != AUTHORIZATION_SCHEMA
        or any(
            type(value[name]) is not int
            for name in (
                "complete_parameter_count",
                "parser_parameter_count",
                "protected_shohin_parameters",
                "source_compiler_parameter_count",
            )
        )
        or value["protected_shohin_parameters"]
        != PROTECTED_SHOHIN_PARAMETERS
        or value["protected_shohin_sha256"] != PROTECTED_SHOHIN_SHA256
        or value["complete_parameter_count"]
        != (
            PROTECTED_SHOHIN_PARAMETERS
            + value["parser_parameter_count"]
            + value["source_compiler_parameter_count"]
        )
        or value["complete_parameter_count"] >= SYSTEM_PARAMETER_LIMIT
        or any(
            not isinstance(value[name], str)
            or len(value[name]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in value[name]
            )
            for name in (
                "machine_sha256",
                "parser_manifest_sha256",
                "parser_state_sha256",
                "protected_shohin_sha256",
                "source_compiler_state_sha256",
            )
        )
    ):
        raise DetachedQueryPackageError(
            "detached execution authorization schema differs"
        )
    return DetachedExecutionAuthorization(**value)


__all__ = [
    "AUTHORIZATION_SCHEMA",
    "DetachedExecutionAuthorization",
    "DetachedQueryPackageError",
    "DetachedQueryParserReceipt",
    "PACKAGE_SCHEMA",
    "build_detached_execution_authorization",
    "detached_query_parser_state_sha256",
    "export_detached_execution_authorization",
    "export_detached_query_parser_package",
    "load_detached_execution_authorization",
    "load_detached_query_parser_package",
    "module_state_sha256",
    "receipt_payload",
]
