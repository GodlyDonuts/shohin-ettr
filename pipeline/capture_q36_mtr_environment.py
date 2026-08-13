#!/usr/bin/env python3
"""Capture the exact base environment and Q36 NF4/fast-kernel overlays."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any

from capture_pcf1_environment import (
    CUDA_BUILD,
    CUDA_COMPILED_VERSION,
    ENVIRONMENT_MANIFEST_SHA256,
    ENVIRONMENT_PIP_FREEZE_SHA256,
    ENVIRONMENT_ROOT,
    ENVIRONMENT_RUNTIME_SHA256,
    PACKAGE_ORIGINS,
    PACKAGE_VERSIONS,
    PYTHON_BASE_PREFIX,
    PYTHON_ENTRYPOINT,
    PYTHON_EXECUTABLE,
    PYTHON_SHA256,
    PYTHON_SITE_PACKAGES,
    PYTHON_VERSION,
    TORCH_VERSION,
    TRANSFORMERS_VERSION,
    _environment_tree,
    _verify_manifest,
    sha256_file,
)
from q36_mtr_roles import MODEL_CONFIG_SHA256, MODEL_ID, MODEL_REVISION

SCHEMA = "shohin-q36-mtr-environment-v1"
BNB_ROOT = Path("/lustre/fs1/home/sa305415/shohin/env_targets/bitsandbytes-0.50.0-r1")
BNB_MANIFEST_SHA256 = "2201774754fb2e0fdd2208b78d34b803b910d8e34c79a43de49b29d7df3a8355"
BNB_VERSION = "0.50.0"
FAST_KERNEL_ROOT = Path(
    "/lustre/fs1/home/sa305415/shohin/env_targets/qwen36-fastkernels-0.4.2-r5"
)
FAST_KERNEL_MANIFEST_SHA256 = (
    "dde2adf539302a321afd7322ded3f2f729ac5f96368113a8af82f64efc0b9e8b"
)
FAST_KERNEL_PACKAGES = {
    "causal-conv1d": "1.6.2.post1",
    "flash-linear-attention": "0.4.2",
}
OVERLAY_ORIGINS = {
    "bitsandbytes": BNB_ROOT / "bitsandbytes/__init__.py",
    "causal-conv1d": FAST_KERNEL_ROOT
    / "causal_conv1d_cuda.cpython-313-x86_64-linux-gnu.so",
    "flash-linear-attention": FAST_KERNEL_ROOT / "fla/__init__.py",
}
OVERLAY_IMPORTS = {
    "bitsandbytes": "bitsandbytes",
    "causal-conv1d": "causal_conv1d_cuda",
    "flash-linear-attention": "fla",
}


class Q36MTREnvironmentError(RuntimeError):
    """The prospective Q36 environment differs from the qualified overlays."""


def _canonical_member(relative: str) -> str:
    if relative.startswith("./"):
        relative = relative[2:]
    pure = PurePosixPath(relative)
    if (
        not relative
        or not pure.parts
        or pure.is_absolute()
        or "." in pure.parts
        or ".." in pure.parts
        or pure.as_posix() != relative
    ):
        raise Q36MTREnvironmentError("Q36 overlay manifest member differs")
    return relative


def _verify_overlay(root: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    manifest = root / "SHA256SUMS"
    if (
        root.is_symlink()
        or not root.is_dir()
        or manifest.is_symlink()
        or not manifest.is_file()
        or sha256_file(manifest) != expected_manifest_sha256
    ):
        raise Q36MTREnvironmentError("Q36 overlay root or manifest differs")
    declared: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        canonical = _canonical_member(relative)
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or canonical in declared
        ):
            raise Q36MTREnvironmentError("Q36 overlay manifest geometry differs")
        declared.append(canonical)
        path = root / canonical
        if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
            raise Q36MTREnvironmentError("Q36 overlay member differs")
    if declared != sorted(declared):
        raise Q36MTREnvironmentError("Q36 overlay manifest order differs")
    actual_files: set[str] = set()
    directories: list[str] = []
    file_bytes = 0
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            actual_files.add(relative)
            file_bytes += path.stat().st_size
        elif stat.S_ISDIR(mode):
            directories.append(relative)
        else:
            raise Q36MTREnvironmentError(
                "Q36 overlay contains a link or special member"
            )
    if actual_files != {*declared, "SHA256SUMS"}:
        raise Q36MTREnvironmentError("Q36 overlay exact membership differs")
    geometry = {
        "manifest_sha256": expected_manifest_sha256,
        "manifest_entries": len(declared),
        "files": len(actual_files),
        "directories": len(directories),
        "file_bytes": file_bytes,
        "directory_sha256": hashlib.sha256(
            ("\n".join(directories) + "\n").encode()
        ).hexdigest(),
    }
    return geometry


def environment_payload(
    runtime_root: Path, runtime_manifest_sha256: str
) -> dict[str, Any]:
    import platform
    import sys

    import torch
    import tokenizers
    import transformers

    runtime = runtime_root.resolve(strict=True)
    runtime_manifest = runtime / "SHA256SUMS"
    if (
        runtime.is_symlink()
        or not runtime.is_dir()
        or runtime_manifest.is_symlink()
        or not runtime_manifest.is_file()
        or sha256_file(runtime_manifest) != runtime_manifest_sha256
    ):
        raise Q36MTREnvironmentError("Q36 packaged runtime differs")
    invoked_python = Path(sys.executable)
    resolved_python = invoked_python.resolve()
    if (
        invoked_python != PYTHON_ENTRYPOINT
        or resolved_python != PYTHON_EXECUTABLE
        or Path(sys.prefix).resolve() != ENVIRONMENT_ROOT
        or Path(sys.base_prefix).resolve() != PYTHON_BASE_PREFIX
        or sha256_file(resolved_python) != PYTHON_SHA256
        or platform.python_implementation() != "CPython"
        or platform.python_version() != PYTHON_VERSION
    ):
        raise Q36MTREnvironmentError("Q36 Python identity differs")
    compiled_cuda = int(torch._C._cuda_getCompiledVersion())
    if (
        torch.__version__ != TORCH_VERSION
        or transformers.__version__ != TRANSFORMERS_VERSION
        or torch.version.cuda != CUDA_BUILD
        or compiled_cuda != CUDA_COMPILED_VERSION
    ):
        raise Q36MTREnvironmentError("Q36 ML/CUDA runtime differs")
    base_manifest_entries = _verify_manifest(
        ENVIRONMENT_ROOT,
        ENVIRONMENT_ROOT / "SHA256SUMS",
        ENVIRONMENT_MANIFEST_SHA256,
    )
    base_tree = _environment_tree(ENVIRONMENT_ROOT)
    bnb = _verify_overlay(BNB_ROOT, BNB_MANIFEST_SHA256)
    fast = _verify_overlay(FAST_KERNEL_ROOT, FAST_KERNEL_MANIFEST_SHA256)
    versions = {**PACKAGE_VERSIONS, "bitsandbytes": BNB_VERSION, **FAST_KERNEL_PACKAGES}
    origins: dict[str, str] = {
        name: str(path) for name, path in PACKAGE_ORIGINS.items()
    }
    for name, expected in versions.items():
        if importlib.metadata.version(name) != expected:
            raise Q36MTREnvironmentError(f"Q36 package version differs: {name}")
    for name, module in OVERLAY_IMPORTS.items():
        specification = importlib.util.find_spec(module)
        if specification is None or specification.origin is None:
            raise Q36MTREnvironmentError(f"Q36 overlay module is absent: {module}")
        origin = Path(specification.origin).resolve()
        if origin != OVERLAY_ORIGINS[name]:
            raise Q36MTREnvironmentError(f"Q36 overlay module origin differs: {module}")
        origins[name] = str(origin)
    runtime_payload = json.loads((runtime / "runtime.json").read_text(encoding="utf-8"))
    allowlisted = runtime_payload.get("allowlisted_files")
    if (
        runtime_payload.get("schema") != "shohin-q36-mtr-runtime-v1"
        or runtime_payload.get("status") != "complete"
        or not isinstance(allowlisted, list)
        or not allowlisted
    ):
        raise Q36MTREnvironmentError("Q36 runtime receipt differs")
    sources = {}
    for relative in allowlisted:
        if not isinstance(relative, str):
            raise Q36MTREnvironmentError("Q36 runtime source name differs")
        path = runtime / relative
        if path.is_symlink() or not path.is_file():
            raise Q36MTREnvironmentError("Q36 runtime source is absent")
        sources[relative] = sha256_file(path)
    return {
        "schema": SCHEMA,
        "status": "pass",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "python": {
            "implementation": "CPython",
            "version": PYTHON_VERSION,
            "entrypoint": str(PYTHON_ENTRYPOINT),
            "resolved_executable": str(resolved_python),
            "executable_sha256": PYTHON_SHA256,
            "prefix": str(ENVIRONMENT_ROOT),
            "base_prefix": str(PYTHON_BASE_PREFIX),
            "site_packages": str(PYTHON_SITE_PACKAGES),
        },
        "packages": versions,
        "module_origins": origins,
        "torch_version": TORCH_VERSION,
        "transformers_version": TRANSFORMERS_VERSION,
        "tokenizers_version": tokenizers.__version__,
        "cuda_build_version": CUDA_BUILD,
        "cuda_compiled_version": compiled_cuda,
        "environment_root": str(ENVIRONMENT_ROOT),
        "environment_manifest_sha256": ENVIRONMENT_MANIFEST_SHA256,
        "environment_manifest_entries": base_manifest_entries,
        "environment_tree": base_tree,
        "environment_tree_sha256": base_tree["sha256"],
        "environment_runtime_sha256": ENVIRONMENT_RUNTIME_SHA256,
        "pip_freeze_sha256": ENVIRONMENT_PIP_FREEZE_SHA256,
        "bitsandbytes_root": str(BNB_ROOT),
        "bitsandbytes_overlay": bnb,
        "fast_kernel_root": str(FAST_KERNEL_ROOT),
        "fast_kernel_overlay": fast,
        "runtime_root": str(runtime),
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "runtime_source_sha256s": sources,
        "offline_required": True,
        "bytecode_writes_permitted": False,
        "scientific_rows_read": 0,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTREnvironmentError(
            f"refusing existing Q36 environment receipt: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = environment_payload(args.runtime_root, args.runtime_manifest_sha256)
    atomic_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "schema": payload["schema"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
