#!/usr/bin/env python3
"""Capture or verify the exact allocated-node PCF1 software environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import subprocess
import sys
from typing import Any

SCHEMA = "shohin-pcf1-environment-receipt-v1"
ENVIRONMENT_ROOT = Path(
    "/lustre/fs1/home/sa305415/shohin/envs/product-reasoning-b3a3603-r2"
)
ENVIRONMENT_MANIFEST_SHA256 = (
    "7eec9e1e94da3480820458912cb96f4ee13c3427543b3addb5ea31953b5d1971"
)
ENVIRONMENT_RUNTIME_SHA256 = (
    "277b97fbd6b18760c9789cf3f3372bdb6b40ca87bf84a1df4b41ee3194c4e9dd"
)
ENVIRONMENT_PIP_FREEZE_SHA256 = (
    "1d4dfd4a1dc11af9788b0bab072d262278db1814d3fca49465d4df5931b3b87a"
)
PYTHON_EXECUTABLE = Path("/lustre/fs1/home/sa305415/shohin/miniforge3/bin/python3.13")
PYTHON_ENTRYPOINT = ENVIRONMENT_ROOT / "bin/python"
PYTHON_BASE_PREFIX = Path("/lustre/fs1/home/sa305415/shohin/miniforge3")
PYTHON_SITE_PACKAGES = ENVIRONMENT_ROOT / "lib/python3.13/site-packages"
PYTHON_SHA256 = "051a031d827eab9778e982571db754662809164c8a3ec01e9beea1e1088123e0"
PYTHON_VERSION = "3.13.13"
TORCH_VERSION = "2.6.0+cu124"
TRANSFORMERS_VERSION = "5.15.0.dev0"
CUDA_BUILD = "12.4"
CUDA_COMPILED_VERSION = 12040
PACKAGE_VERSIONS = {
    "accelerate": "1.14.0",
    "huggingface-hub": "1.22.0",
    "peft": "0.20.0",
    "safetensors": "0.8.0",
    "sentencepiece": "0.2.2",
    "tokenizers": "0.22.2",
    "torch": TORCH_VERSION,
    "transformers": TRANSFORMERS_VERSION,
    "triton": "3.2.0",
}
PACKAGE_IMPORTS = {
    "accelerate": "accelerate",
    "huggingface-hub": "huggingface_hub",
    "peft": "peft",
    "safetensors": "safetensors",
    "sentencepiece": "sentencepiece",
    "tokenizers": "tokenizers",
    "torch": "torch",
    "transformers": "transformers",
    "triton": "triton",
}
PACKAGE_ORIGINS = {
    "accelerate": ENVIRONMENT_ROOT
    / "lib/python3.13/site-packages/accelerate/__init__.py",
    "huggingface-hub": PYTHON_BASE_PREFIX
    / "lib/python3.13/site-packages/huggingface_hub/__init__.py",
    "peft": ENVIRONMENT_ROOT / "lib/python3.13/site-packages/peft/__init__.py",
    "safetensors": ENVIRONMENT_ROOT
    / "lib/python3.13/site-packages/safetensors/__init__.py",
    "sentencepiece": ENVIRONMENT_ROOT
    / "lib/python3.13/site-packages/sentencepiece/__init__.py",
    "tokenizers": ENVIRONMENT_ROOT
    / "lib/python3.13/site-packages/tokenizers/__init__.py",
    "torch": PYTHON_BASE_PREFIX / "lib/python3.13/site-packages/torch/__init__.py",
    "transformers": ENVIRONMENT_ROOT
    / "lib/python3.13/site-packages/transformers/__init__.py",
    "triton": PYTHON_BASE_PREFIX / "lib/python3.13/site-packages/triton/__init__.py",
}
ENVIRONMENT_TREE_SHA256 = (
    "6c3311032bc4efb065222378e053e1cc15266b37bd868aee2bc05aa94f8ebf9c"
)
ENVIRONMENT_TREE_ENTRIES = 9_416
ENVIRONMENT_TREE_FILES = 8_022
ENVIRONMENT_TREE_DIRECTORIES = 1_390
ENVIRONMENT_TREE_SYMLINKS = 4
ENVIRONMENT_TREE_BYTES = 170_408_093
SOURCE_DEPENDENCIES = (
    "pipeline/build_pcf1_custody.py",
    "pipeline/capture_pcf1_environment.py",
    "pipeline/normalize_pcf1_reports.py",
    "pipeline/package_pcf1_runtime.py",
    "pipeline/score_pcf1_commit.py",
    "train/hf_pcf1_apply_commit.py",
    "train/hf_pcf1_evaluate.py",
    "train/hf_pcf1_generate_drafts.py",
    "train/hf_pcf1_mechanics.py",
    "train/hf_pcf1_train_commit.py",
    "train/hf_product_reasoning_eval.py",
    "train/hf_product_reasoning_train.py",
    "train/pcf1_code_sandbox.py",
    "train/pcf1_environment.py",
    "train/jobs/pcf1_common.sh",
)


class PCF1EnvironmentError(RuntimeError):
    """The PCF1 runtime environment differs from its frozen identity."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_manifest(root: Path, manifest: Path, expected_sha256: str) -> int:
    if not root.is_dir() or not manifest.is_file():
        raise PCF1EnvironmentError("PCF1 environment manifest is missing")
    if sha256_file(manifest) != expected_sha256:
        raise PCF1EnvironmentError("PCF1 environment manifest hash differs")
    entries: list[tuple[str, str]] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not relative
        ):
            raise PCF1EnvironmentError("PCF1 environment manifest entry differs")
        entries.append((digest, relative))
    expected_entries = [
        (ENVIRONMENT_RUNTIME_SHA256, str(root / "runtime.json")),
        (ENVIRONMENT_PIP_FREEZE_SHA256, str(root / "pip-freeze.txt")),
    ]
    if entries != expected_entries or any(
        not PurePosixPath(relative).is_absolute() for _, relative in entries
    ):
        raise PCF1EnvironmentError("PCF1 environment manifest geometry differs")
    subprocess.run(
        ["sha256sum", "-c", str(manifest)],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.DEVNULL,
    )
    return len(entries)


def _environment_tree(root: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    files = directories = symlinks = file_bytes = 0
    paths = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    for path in paths:
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            size = path.stat().st_size
            row: dict[str, Any] = {
                "path": relative,
                "sha256": sha256_file(path),
                "size": size,
                "type": "file",
            }
            files += 1
            file_bytes += size
        elif stat.S_ISDIR(mode):
            row = {"path": relative, "type": "directory"}
            directories += 1
        elif stat.S_ISLNK(mode):
            row = {"path": relative, "target": os.readlink(path), "type": "symlink"}
            symlinks += 1
        else:
            raise PCF1EnvironmentError(
                f"PCF1 environment has special member: {relative}"
            )
        digest.update(
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
    result: dict[str, int | str] = {
        "sha256": digest.hexdigest(),
        "entries": len(paths),
        "files": files,
        "directories": directories,
        "symlinks": symlinks,
        "file_bytes": file_bytes,
    }
    expected = {
        "sha256": ENVIRONMENT_TREE_SHA256,
        "entries": ENVIRONMENT_TREE_ENTRIES,
        "files": ENVIRONMENT_TREE_FILES,
        "directories": ENVIRONMENT_TREE_DIRECTORIES,
        "symlinks": ENVIRONMENT_TREE_SYMLINKS,
        "file_bytes": ENVIRONMENT_TREE_BYTES,
    }
    if result != expected:
        raise PCF1EnvironmentError("PCF1 full environment tree differs")
    return result


def environment_payload(
    runtime_root: Path, runtime_manifest_sha256: str
) -> dict[str, Any]:
    import torch
    import tokenizers
    import transformers
    from transformers import AutoModelForMultimodalLM, AutoTokenizer

    runtime = runtime_root.resolve()
    runtime_manifest = runtime / "SHA256SUMS"
    if (
        not runtime.is_dir()
        or not runtime_manifest.is_file()
        or sha256_file(runtime_manifest) != runtime_manifest_sha256
    ):
        raise PCF1EnvironmentError("PCF1 packaged runtime differs")
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
        raise PCF1EnvironmentError("PCF1 Python identity differs")
    compiled_cuda = int(torch._C._cuda_getCompiledVersion())
    if (
        torch.__version__ != TORCH_VERSION
        or transformers.__version__ != TRANSFORMERS_VERSION
        or torch.version.cuda != CUDA_BUILD
        or compiled_cuda != CUDA_COMPILED_VERSION
    ):
        raise PCF1EnvironmentError("PCF1 ML/CUDA runtime differs")
    environment_manifest = ENVIRONMENT_ROOT / "SHA256SUMS"
    environment_entries = _verify_manifest(
        ENVIRONMENT_ROOT, environment_manifest, ENVIRONMENT_MANIFEST_SHA256
    )
    environment_tree = _environment_tree(ENVIRONMENT_ROOT)
    environment_runtime = ENVIRONMENT_ROOT / "runtime.json"
    pip_freeze = ENVIRONMENT_ROOT / "pip-freeze.txt"
    if (
        sha256_file(environment_runtime) != ENVIRONMENT_RUNTIME_SHA256
        or sha256_file(pip_freeze) != ENVIRONMENT_PIP_FREEZE_SHA256
    ):
        raise PCF1EnvironmentError("PCF1 environment receipt files differ")
    sources = {}
    for relative in SOURCE_DEPENDENCIES:
        path = runtime / relative
        if not path.is_file() or path.is_symlink():
            raise PCF1EnvironmentError(f"missing PCF1 runtime dependency: {relative}")
        sources[relative] = sha256_file(path)
    packages = {}
    module_origins = {}
    for name, expected_version in PACKAGE_VERSIONS.items():
        try:
            observed_version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise PCF1EnvironmentError(f"missing PCF1 package: {name}") from error
        if observed_version != expected_version:
            raise PCF1EnvironmentError(f"PCF1 package version differs: {name}")
        packages[name] = observed_version
        module_name = PACKAGE_IMPORTS[name]
        specification = importlib.util.find_spec(module_name)
        if specification is None or specification.origin is None:
            raise PCF1EnvironmentError(f"missing PCF1 module origin: {module_name}")
        origin = Path(specification.origin).resolve()
        if origin != PACKAGE_ORIGINS[name]:
            raise PCF1EnvironmentError(f"PCF1 module origin differs: {module_name}")
        module_origins[name] = str(origin)
    return {
        "schema": SCHEMA,
        "status": "complete",
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
        "packages": packages,
        "module_origins": module_origins,
        "torch_version": TORCH_VERSION,
        "transformers_version": TRANSFORMERS_VERSION,
        "tokenizers_version": tokenizers.__version__,
        "cuda_build_version": CUDA_BUILD,
        "cuda_compiled_version": compiled_cuda,
        "auto_tokenizer_class": f"{AutoTokenizer.__module__}.{AutoTokenizer.__name__}",
        "multimodal_auto_class": (
            f"{AutoModelForMultimodalLM.__module__}.{AutoModelForMultimodalLM.__name__}"
        ),
        "environment_root": str(ENVIRONMENT_ROOT),
        "environment_manifest_sha256": ENVIRONMENT_MANIFEST_SHA256,
        "environment_manifest_entries": environment_entries,
        "environment_tree": environment_tree,
        "environment_runtime_sha256": ENVIRONMENT_RUNTIME_SHA256,
        "pip_freeze_sha256": ENVIRONMENT_PIP_FREEZE_SHA256,
        "runtime_root": str(runtime),
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "runtime_source_sha256s": sources,
        "offline_required": True,
        "bytecode_writes_permitted": False,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise PCF1EnvironmentError(f"refusing existing environment receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise PCF1EnvironmentError("PCF1 environment publication race") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    capture = modes.add_parser("capture")
    verify = modes.add_parser("verify")
    for mode in (capture, verify):
        mode.add_argument("--runtime-root", type=Path, required=True)
        mode.add_argument("--runtime-manifest-sha256", required=True)
    capture.add_argument("--output", type=Path, required=True)
    verify.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    payload = environment_payload(args.runtime_root, args.runtime_manifest_sha256)
    if args.mode == "capture":
        atomic_json(args.output, payload)
    else:
        observed = json.loads(args.receipt.read_text(encoding="utf-8"))
        if observed != payload:
            raise PCF1EnvironmentError("PCF1 environment receipt replay differs")
    print(json.dumps({"status": "complete"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
