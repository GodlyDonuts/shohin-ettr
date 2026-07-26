"""Canonical source bundle for pre-import ETTR candidate execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import shutil
import stat
import sys


RUNTIME_BUNDLE_SCHEMA = "ettr-runtime-bundle-v1"
RUNTIME_SOURCE_FILES = (
    "endogenous_typed_theory_reactor.py",
    "ettr_factorial_custody.py",
    "ettr_state_io.py",
    "model.py",
    "run_ettr_late_query.py",
    "run_ettr_state_executor.py",
    "run_ettr_world_compiler.py",
)
RUNTIME_DISTRIBUTIONS = ("safetensors", "torch")


class ETTRRuntimeBundleError(ValueError):
    """A source bundle or runtime identity differs from its frozen receipt."""


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_receipt(name: str) -> tuple[str, str, str, str, str]:
    try:
        version = metadata.version(name)
        spec = util.find_spec(name)
    except (ImportError, metadata.PackageNotFoundError) as exc:
        raise ETTRRuntimeBundleError(
            f"runtime distribution is unavailable: {name}"
        ) from exc
    if spec is None or spec.origin is None:
        raise ETTRRuntimeBundleError(
            f"runtime distribution origin is unavailable: {name}"
        )
    origin = Path(spec.origin).resolve()
    if not origin.is_file():
        raise ETTRRuntimeBundleError(
            f"runtime distribution origin is not a file: {name}"
        )
    distribution_root = Path(metadata.distribution(name).locate_file(".")).resolve()
    return (
        name,
        version,
        str(origin),
        _sha256_file(origin),
        str(distribution_root),
    )


def _immutable_regular_sha256(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ETTRRuntimeBundleError(
            f"runtime bundle file cannot be opened: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_mode & 0o222
        ):
            raise ETTRRuntimeBundleError(
                f"runtime bundle file is not immutable: {path}"
            )
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ETTRRuntimeBundleError(
                f"runtime bundle file changed during read: {path}"
            )
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def _validate_bundle_root(bundle_root: Path) -> None:
    metadata_value = bundle_root.lstat()
    if (
        stat.S_ISLNK(metadata_value.st_mode)
        or not stat.S_ISDIR(metadata_value.st_mode)
        or metadata_value.st_mode & 0o222
    ):
        raise ETTRRuntimeBundleError("runtime bundle root is not immutable")


@dataclass(frozen=True, slots=True)
class ETTRRuntimeBundleReceipt:
    """Exact local source and interpreter identity admitted before imports."""

    schema: str
    python_implementation: str
    python_version: str
    python_executable_sha256: str
    source_files: tuple[tuple[str, str], ...]
    distributions: tuple[tuple[str, str, str, str, str], ...]

    @classmethod
    def build(cls, source_root: Path) -> ETTRRuntimeBundleReceipt:
        source_root = source_root.resolve()
        source_files = tuple(
            (name, _sha256_file(source_root / name))
            for name in RUNTIME_SOURCE_FILES
        )
        return cls(
            schema=RUNTIME_BUNDLE_SCHEMA,
            python_implementation=sys.implementation.name,
            python_version=sys.version,
            python_executable_sha256=_sha256_file(
                Path(sys.executable).resolve()
            ),
            source_files=source_files,
            distributions=tuple(
                _distribution_receipt(name)
                for name in RUNTIME_DISTRIBUTIONS
            ),
        )

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(asdict(self))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def validate(self, bundle_root: Path) -> None:
        _validate_bundle_root(bundle_root)
        if (
            self.schema != RUNTIME_BUNDLE_SCHEMA
            or self.python_implementation != sys.implementation.name
            or self.python_version != sys.version
            or self.python_executable_sha256
            != _sha256_file(Path(sys.executable).resolve())
            or tuple(name for name, _ in self.source_files)
            != RUNTIME_SOURCE_FILES
            or tuple(name for name, _, _, _, _ in self.distributions)
            != RUNTIME_DISTRIBUTIONS
            or self.distributions
            != tuple(
                _distribution_receipt(name)
                for name in RUNTIME_DISTRIBUTIONS
            )
        ):
            raise ETTRRuntimeBundleError("runtime bundle identity differs")
        actual_names = tuple(
            sorted(
                path.name
                for path in bundle_root.iterdir()
            )
        )
        if actual_names != tuple(sorted(RUNTIME_SOURCE_FILES)):
            raise ETTRRuntimeBundleError("runtime bundle inventory differs")
        for name, expected_sha256 in self.source_files:
            path = bundle_root / name
            if _immutable_regular_sha256(path) != expected_sha256:
                raise ETTRRuntimeBundleError(
                    f"runtime bundle source differs: {name}"
                )

    @classmethod
    def from_path(cls, path: Path) -> ETTRRuntimeBundleReceipt:
        expected_sha256 = _immutable_regular_sha256(path)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ETTRRuntimeBundleError("runtime receipt changed during read")
        try:
            value = json.loads(payload.decode("ascii"))
            receipt = cls(
                schema=value["schema"],
                python_implementation=value["python_implementation"],
                python_version=value["python_version"],
                python_executable_sha256=value["python_executable_sha256"],
                source_files=tuple(
                    tuple(item) for item in value["source_files"]
                ),
                distributions=tuple(
                    tuple(item) for item in value["distributions"]
                ),
            )
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ETTRRuntimeBundleError(
                "runtime receipt is malformed"
            ) from exc
        if payload != receipt.canonical_bytes():
            raise ETTRRuntimeBundleError("runtime receipt is not canonical")
        return receipt


def materialize_runtime_bundle(
    source_root: Path,
    bundle_root: Path,
) -> ETTRRuntimeBundleReceipt:
    """Copy the exact allowlist into a fresh read-only runtime directory."""

    receipt = ETTRRuntimeBundleReceipt.build(source_root)
    try:
        bundle_root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise ETTRRuntimeBundleError(
            "runtime bundle destination already exists"
        ) from exc
    for name, _ in receipt.source_files:
        destination = bundle_root / name
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with (
                (source_root / name).open("rb") as source,
                os.fdopen(descriptor, "wb") as target,
            ):
                descriptor = -1
                shutil.copyfileobj(source, target)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        destination.chmod(0o444)
    bundle_root.chmod(0o555)
    receipt.validate(bundle_root)
    return receipt


__all__ = [
    "ETTRRuntimeBundleError",
    "ETTRRuntimeBundleReceipt",
    "RUNTIME_BUNDLE_SCHEMA",
    "RUNTIME_DISTRIBUTIONS",
    "RUNTIME_SOURCE_FILES",
    "materialize_runtime_bundle",
]
