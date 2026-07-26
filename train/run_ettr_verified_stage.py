#!/usr/bin/env python3
"""Stdlib-only verifier that admits ETTR stage imports after source hashing."""

from __future__ import annotations

import argparse
import errno
import hashlib
from importlib import abc, machinery
import json
import os
from pathlib import Path
import stat
import sys


RUNTIME_BUNDLE_SCHEMA = "ettr-runtime-bundle-v3"
EXECUTION_MANIFEST_SCHEMA = "ettr-factorial-execution-manifest-v4"
COMMON_RUNTIME_SOURCE_FILES = (
    "endogenous_typed_theory_reactor.py",
    "ettr_factorial_custody.py",
    "ettr_state_io.py",
    "model.py",
)
RUNTIME_DISTRIBUTIONS = ("safetensors", "torch")
STAGE_RUNNERS = {
    "world": "run_ettr_world_compiler.py",
    "command": "run_ettr_state_executor.py",
    "query": "run_ettr_late_query.py",
}


class VerifiedStageError(ValueError):
    """The pre-import stage bundle differs from its preregistered identity."""


def _reject_inherited_descriptors() -> None:
    """Candidate code receives no persistent descriptor above stderr."""

    descriptor_root = next(
        (
            Path(candidate)
            for candidate in ("/proc/self/fd", "/dev/fd")
            if Path(candidate).is_dir()
        ),
        None,
    )
    if descriptor_root is None:
        raise VerifiedStageError("bootstrap descriptor inventory is unavailable")
    try:
        names = os.listdir(descriptor_root)
    except OSError as exc:
        raise VerifiedStageError(
            "bootstrap descriptor inventory is unavailable"
        ) from exc
    for name in names:
        if not name.isdigit() or int(name) <= 2:
            continue
        try:
            os.fstat(int(name))
        except FileNotFoundError:
            # os.listdir may briefly expose its own closed directory descriptor.
            continue
        except OSError as exc:
            if exc.errno == errno.EBADF:
                continue
            raise VerifiedStageError("bootstrap descriptor inventory differs") from exc
        raise VerifiedStageError("bootstrap inherited descriptor differs")


class _VerifiedSourceFinder(abc.MetaPathFinder, abc.Loader):
    """Load admitted first-party modules only from retained verified bytes."""

    def __init__(self, sources: dict[str, tuple[str, bytes]]) -> None:
        self._sources = sources

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: object = None,
    ) -> machinery.ModuleSpec | None:
        del path, target
        if fullname not in self._sources:
            return None
        origin, _ = self._sources[fullname]
        return machinery.ModuleSpec(fullname, self, origin=origin)

    def create_module(self, spec: machinery.ModuleSpec) -> None:
        del spec
        return None

    def exec_module(self, module: object) -> None:
        name = getattr(module, "__name__", None)
        if not isinstance(name, str) or name not in self._sources:
            raise VerifiedStageError("verified module identity differs")
        origin, payload = self._sources[name]
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict):
            raise VerifiedStageError("verified module namespace differs")
        exec(compile(payload, origin, "exec"), namespace)


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


def _read_immutable_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise VerifiedStageError(
            f"bootstrap input cannot be opened: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_mode & 0o222
        ):
            raise VerifiedStageError(
                f"bootstrap input is not immutable: {path}"
            )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 * 1024 * 1024):
            chunks.append(chunk)
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
            raise VerifiedStageError(
                f"bootstrap input changed during read: {path}"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_immutable_canonical(path: Path) -> tuple[dict[str, object], bytes]:
    payload = _read_immutable_bytes(path)
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifiedStageError(f"bootstrap JSON is malformed: {path}") from exc
    if not isinstance(value, dict) or payload != _canonical_json_bytes(value):
        raise VerifiedStageError(f"bootstrap JSON is not canonical: {path}")
    return value, payload


def _verify(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    receipt_path: Path,
    bundle_root: Path,
    stage: str,
) -> tuple[
    Path,
    bytes,
    dict[str, tuple[str, bytes]],
    tuple[str, ...],
]:
    manifest, manifest_bytes = _read_immutable_canonical(manifest_path)
    receipt, receipt_bytes = _read_immutable_canonical(receipt_path)
    own_sha256 = _sha256_file(Path(__file__).resolve())
    claim_runtime_sha256 = os.environ.get(
        "SHOHIN_ETTR_CLAIM_RUNTIME_SHA256"
    )
    claim_runtime_inventory_sha256 = os.environ.get(
        "SHOHIN_ETTR_CLAIM_RUNTIME_INVENTORY_SHA256"
    )
    external_launcher_sha256 = os.environ.get(
        "SHOHIN_ETTR_EXTERNAL_LAUNCHER_SHA256"
    )
    bwrap_sha256 = os.environ.get("SHOHIN_ETTR_BWRAP_SHA256")
    stage_policy_sha256 = os.environ.get(
        "SHOHIN_ETTR_STAGE_POLICY_SHA256"
    )
    expected_stage_policy_sha256 = manifest.get(
        {
            "world": "world_stage_policy_sha256",
            "command": "command_stage_policy_sha256",
            "query": "query_stage_policy_sha256",
        }[stage]
    )
    network_isolated = os.environ.get(
        "SHOHIN_ETTR_NETWORK_NAMESPACE_ISOLATED"
    )
    if (
        manifest.get("schema") != EXECUTION_MANIFEST_SCHEMA
        or hashlib.sha256(manifest_bytes).hexdigest()
        != expected_manifest_sha256
        or manifest.get("bootstrap_sha256") != own_sha256
        or manifest.get(
            {
                "world": "world_runtime_bundle_sha256",
                "command": "command_runtime_bundle_sha256",
                "query": "query_runtime_bundle_sha256",
            }[stage]
        )
        != hashlib.sha256(receipt_bytes).hexdigest()
        or manifest.get("claim_runtime_archive_sha256")
        != claim_runtime_sha256
        or manifest.get("claim_runtime_inventory_sha256")
        != claim_runtime_inventory_sha256
        or manifest.get("external_launcher_sha256")
        != external_launcher_sha256
        or manifest.get("bwrap_sha256") != bwrap_sha256
        or expected_stage_policy_sha256 != stage_policy_sha256
        or manifest.get("network_namespace_required") is not True
        or network_isolated != "1"
        or receipt.get("schema") != RUNTIME_BUNDLE_SCHEMA
        or receipt.get("python_implementation") != sys.implementation.name
        or receipt.get("python_version") != sys.version
        or receipt.get("python_executable_sha256")
        != _sha256_file(Path(sys.executable).resolve())
    ):
        raise VerifiedStageError("bootstrap identity differs")
    try:
        source_files = tuple(tuple(item) for item in receipt["source_files"])
        distributions = tuple(tuple(item) for item in receipt["distributions"])
    except (KeyError, TypeError) as exc:
        raise VerifiedStageError("runtime receipt geometry differs") from exc
    if (
        receipt.get("stage") != stage
        or tuple(name for name, _ in source_files)
        != (
            *COMMON_RUNTIME_SOURCE_FILES,
            STAGE_RUNNERS[stage],
        )
        or tuple(name for name, _, _, _, _ in distributions)
        != RUNTIME_DISTRIBUTIONS
    ):
        raise VerifiedStageError("runtime receipt identity differs")
    runtime_prefix = Path(sys.prefix).resolve()
    for name, version, origin_text, expected_sha256, root_text in distributions:
        try:
            origin = (runtime_prefix / origin_text).resolve(strict=True)
            root = (runtime_prefix / root_text).resolve(strict=True)
            origin.relative_to(runtime_prefix)
            root.relative_to(runtime_prefix)
        except (OSError, RuntimeError, ValueError) as exc:
            raise VerifiedStageError("external runtime TCB differs") from exc
        if (
            not name
            or not version
            or _sha256_file(origin) != expected_sha256
            or not root.is_dir()
            or origin.parent.parent != root
        ):
            raise VerifiedStageError("external runtime TCB differs")
    bundle_metadata = bundle_root.lstat()
    if (
        stat.S_ISLNK(bundle_metadata.st_mode)
        or not stat.S_ISDIR(bundle_metadata.st_mode)
        or bundle_metadata.st_mode & 0o222
    ):
        raise VerifiedStageError("runtime source root differs")
    actual_names = tuple(
        sorted(path.name for path in bundle_root.iterdir())
    )
    runtime_source_files = (
        *COMMON_RUNTIME_SOURCE_FILES,
        STAGE_RUNNERS[stage],
    )
    if actual_names != tuple(sorted(runtime_source_files)):
        raise VerifiedStageError("runtime source inventory differs")
    expected_files = dict(source_files)
    verified_sources: dict[str, tuple[str, bytes]] = {}
    for name in runtime_source_files:
        path = bundle_root / name
        payload = _read_immutable_bytes(path)
        if hashlib.sha256(payload).hexdigest() != expected_files[name]:
            raise VerifiedStageError(f"runtime source differs: {name}")
        verified_sources[Path(name).stem] = (str(path), payload)
    runner = STAGE_RUNNERS[stage]
    runner_field = {
        "world": "compiler_runner_sha256",
        "command": "executor_runner_sha256",
        "query": "query_runner_sha256",
    }[stage]
    if manifest.get(runner_field) != expected_files[runner]:
        raise VerifiedStageError("stage runner identity differs")
    runner_path = bundle_root / runner
    runner_bytes = verified_sources[runner_path.stem][1]
    runtime_roots = tuple(
        dict.fromkeys(
            str((runtime_prefix / root).resolve(strict=True))
            for _, _, _, _, root in distributions
        )
    )
    return runner_path, runner_bytes, verified_sources, runtime_roots


def main() -> None:
    if not (
        sys.flags.isolated
        and sys.flags.safe_path
        and sys.flags.no_site
        and sys.flags.ignore_environment
        and sys.flags.dont_write_bytecode
    ):
        raise VerifiedStageError("bootstrap requires python -I -S -B")
    _reject_inherited_descriptors()
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGE_RUNNERS), required=True)
    parser.add_argument("runner_arguments", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    runner_arguments = arguments.runner_arguments
    if runner_arguments[:1] == ["--"]:
        runner_arguments = runner_arguments[1:]
    runner, runner_bytes, verified_sources, runtime_roots = _verify(
        manifest_path=arguments.manifest,
        expected_manifest_sha256=arguments.manifest_sha256,
        receipt_path=arguments.runtime_receipt,
        bundle_root=arguments.bundle_root,
        stage=arguments.stage,
    )
    os.environ.pop("PYTHONPATH", None)
    os.environ.pop("PYTHONHOME", None)
    sys.dont_write_bytecode = True
    forbidden_manifest_options = {
        "--execution-manifest",
        "--execution-manifest-sha256",
    }
    if any(
        argument.split("=", 1)[0] in forbidden_manifest_options
        for argument in runner_arguments
    ):
        raise VerifiedStageError("runner manifest override is forbidden")
    stdlib_paths = tuple(path for path in sys.path if path)
    sys.path[:] = [
        *runtime_roots,
        *stdlib_paths,
    ]
    sys.path_importer_cache.clear()
    sys.meta_path.insert(0, _VerifiedSourceFinder(verified_sources))
    sys.argv = [
        str(runner),
        *runner_arguments,
        "--execution-manifest",
        str(arguments.manifest),
        "--execution-manifest-sha256",
        arguments.manifest_sha256,
    ]
    globals_value = {
        "__builtins__": __builtins__,
        "__file__": str(runner),
        "__name__": "__main__",
        "__package__": None,
        "__spec__": None,
    }
    exec(compile(runner_bytes, str(runner), "exec"), globals_value)


if __name__ == "__main__":
    main()
