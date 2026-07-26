"""Canonical source bundle for pre-import ETTR candidate execution."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Sequence


RUNTIME_BUNDLE_SCHEMA = "ettr-runtime-bundle-v3"
COMMON_RUNTIME_SOURCE_FILES = (
    "endogenous_typed_theory_reactor.py",
    "ettr_factorial_custody.py",
    "ettr_state_io.py",
    "model.py",
)
STAGE_RUNNERS = {
    "world": "run_ettr_world_compiler.py",
    "command": "run_ettr_state_executor.py",
    "query": "run_ettr_late_query.py",
}
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
    runtime_prefix = Path(sys.prefix).resolve()
    try:
        origin_relative = origin.relative_to(runtime_prefix)
        root_relative = distribution_root.relative_to(runtime_prefix)
    except ValueError as exc:
        raise ETTRRuntimeBundleError(
            f"runtime distribution escapes interpreter prefix: {name}"
        ) from exc
    return (
        name,
        version,
        origin_relative.as_posix(),
        _sha256_file(origin),
        root_relative.as_posix(),
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
    stage: str
    python_implementation: str
    python_version: str
    python_executable_sha256: str
    source_files: tuple[tuple[str, str], ...]
    distributions: tuple[tuple[str, str, str, str, str], ...]

    @classmethod
    def build(
        cls,
        source_root: Path,
        *,
        stage: str,
    ) -> ETTRRuntimeBundleReceipt:
        source_root = source_root.resolve()
        try:
            source_names = (
                *COMMON_RUNTIME_SOURCE_FILES,
                STAGE_RUNNERS[stage],
            )
        except KeyError as exc:
            raise ETTRRuntimeBundleError(
                "runtime bundle stage differs"
            ) from exc
        source_files = tuple(
            (name, _sha256_file(source_root / name))
            for name in source_names
        )
        return cls(
            schema=RUNTIME_BUNDLE_SCHEMA,
            stage=stage,
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
            or self.stage not in STAGE_RUNNERS
            or self.python_implementation != sys.implementation.name
            or self.python_version != sys.version
            or self.python_executable_sha256
            != _sha256_file(Path(sys.executable).resolve())
            or tuple(name for name, _ in self.source_files)
            != (
                *COMMON_RUNTIME_SOURCE_FILES,
                STAGE_RUNNERS[self.stage],
            )
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
        if actual_names != tuple(
            sorted(
                (
                    *COMMON_RUNTIME_SOURCE_FILES,
                    STAGE_RUNNERS[self.stage],
                )
            )
        ):
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
                stage=value["stage"],
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
    *,
    stage: str,
) -> ETTRRuntimeBundleReceipt:
    """Copy the exact allowlist into a fresh read-only runtime directory."""

    receipt = ETTRRuntimeBundleReceipt.build(source_root, stage=stage)
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


def write_runtime_bundle_receipt_once(
    source_root: Path,
    receipt_path: Path,
    *,
    stage: str,
) -> ETTRRuntimeBundleReceipt:
    """Write one portable source/runtime receipt without copying sources."""

    receipt = ETTRRuntimeBundleReceipt.build(
        source_root.resolve(strict=True),
        stage=stage,
    )
    payload = receipt.canonical_bytes()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(receipt_path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise ETTRRuntimeBundleError(
                    "runtime receipt write made no progress"
                )
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    receipt_path.chmod(0o444)
    return receipt


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_RUNNERS),
        required=True,
    )
    arguments = parser.parse_args(argv)
    receipt = write_runtime_bundle_receipt_once(
        arguments.source_root,
        arguments.receipt,
        stage=arguments.stage,
    )
    print(receipt.sha256())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "COMMON_RUNTIME_SOURCE_FILES",
    "ETTRRuntimeBundleError",
    "ETTRRuntimeBundleReceipt",
    "RUNTIME_BUNDLE_SCHEMA",
    "RUNTIME_DISTRIBUTIONS",
    "STAGE_RUNNERS",
    "materialize_runtime_bundle",
    "write_runtime_bundle_receipt_once",
]
