#!/usr/bin/env python3
"""Build and seal the pinned GPT-OSS MXFP4 execution overlay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
from typing import Any

SCHEMA = "shohin-gpt-oss-mxfp4-overlay-v1"
PACKAGES = {
    "kernels": "0.16.0",
    "kernels-data": "0.16.0",
    "triton": "3.4.0",
}
KERNEL_REPOSITORY = "kernels-community/gpt-oss-triton-kernels"
KERNEL_REVISION = "9655fcf7d0f638bec4a82f6f1a70014f0aa8cfb0"


class GptOssOverlayError(RuntimeError):
    """The pinned GPT-OSS MXFP4 overlay contract failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise GptOssOverlayError(f"refusing existing overlay output: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def pip_command(python: Path, target: Path) -> list[str]:
    return [
        str(python),
        "-m",
        "pip",
        "install",
        "--isolated",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--no-deps",
        "--target",
        str(target),
        *(f"{name}=={version}" for name, version in sorted(PACKAGES.items())),
    ]


def installed_versions(root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for distribution in importlib.metadata.Distribution.discover(path=[str(root)]):
        name = distribution.metadata.get("Name", "").lower()
        if name in PACKAGES:
            if name in observed:
                raise GptOssOverlayError("duplicate overlay distribution")
            observed[name] = distribution.version
    if observed != PACKAGES:
        raise GptOssOverlayError(
            f"overlay package versions differ: expected={PACKAGES!r} observed={observed!r}"
        )
    return observed


def manifest_tree(root: Path) -> tuple[str, dict[str, Any]]:
    rows: list[tuple[str, str]] = []
    covered_bytes = 0
    for candidate in sorted(root.rglob("*")):
        relative = candidate.relative_to(root).as_posix()
        mode = candidate.lstat().st_mode
        if stat.S_ISDIR(mode) and not candidate.is_symlink():
            continue
        if not stat.S_ISREG(mode) or candidate.is_symlink():
            raise GptOssOverlayError("overlay tree contains a link or special file")
        if relative == "SHA256SUMS":
            continue
        rows.append((relative, sha256_file(candidate)))
        covered_bytes += candidate.stat().st_size
    if not rows:
        raise GptOssOverlayError("overlay tree is empty")
    text = "".join(f"{digest}  {relative}\n" for relative, digest in rows)
    return text, {
        "manifest_entries": len(rows),
        "covered_bytes": covered_bytes,
        "exact_regular_files": True,
    }


def _remove_generated_caches(root: Path) -> None:
    for candidate in sorted(root.rglob("__pycache__"), reverse=True):
        if candidate.is_symlink() or not candidate.is_dir():
            raise GptOssOverlayError("overlay bytecode cache differs")
        shutil.rmtree(candidate)
    local_cache = root / "kernel-repo" / ".cache"
    if local_cache.exists():
        if local_cache.is_symlink() or not local_cache.is_dir():
            raise GptOssOverlayError("kernel download cache differs")
        shutil.rmtree(local_cache)


def _validate_imports(python: Path, root: Path) -> dict[str, Any]:
    program = """
import importlib.metadata as metadata
from pathlib import Path
import kernels
import triton
module = kernels.get_local_kernel(Path(__import__('os').environ['GPT_OSS_KERNEL_ROOT']), backend='cpu')
assert module.__file__ is not None
print(__import__('json').dumps({
    'kernels': metadata.version('kernels'),
    'kernels-data': metadata.version('kernels-data'),
    'triton': metadata.version('triton'),
    'kernel_module': str(Path(module.__file__).resolve()),
}, sort_keys=True))
"""
    environment = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(root),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "GPT_OSS_KERNEL_ROOT": str(root / "kernel-repo"),
        "LOCAL_KERNELS": f"{KERNEL_REPOSITORY}={root / 'kernel-repo'}",
        "HF_HUB_OFFLINE": "1",
    }
    completed = subprocess.run(
        [str(python), "-P", "-s", "-B", "-c", program],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise GptOssOverlayError("overlay import receipt is unreadable") from error
    if {key: payload.get(key) for key in PACKAGES} != PACKAGES:
        raise GptOssOverlayError("overlay import package versions differ")
    kernel_module = Path(str(payload.get("kernel_module"))).resolve(strict=True)
    if not kernel_module.is_relative_to((root / "kernel-repo").resolve(strict=True)):
        raise GptOssOverlayError("overlay kernel module origin differs")
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve(strict=False)
    report = args.report.resolve(strict=False)
    python = args.python.resolve(strict=True)
    if (
        output_root.exists()
        or output_root.is_symlink()
        or report.exists()
        or report.is_symlink()
    ):
        raise GptOssOverlayError("final overlay output exists")
    stage = output_root.with_name(f".{output_root.name}.partial")
    if stage.exists() or stage.is_symlink():
        raise GptOssOverlayError("overlay staging output exists")
    stage.mkdir(parents=True)
    subprocess.run(pip_command(python, stage), check=True)

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=KERNEL_REPOSITORY,
        repo_type="kernel",
        revision=KERNEL_REVISION,
        local_dir=stage / "kernel-repo",
        max_workers=args.workers,
    )
    _remove_generated_caches(stage)
    versions = installed_versions(stage)
    imports = _validate_imports(python, stage)
    _remove_generated_caches(stage)
    sums, tree = manifest_tree(stage)
    _atomic_text(stage / "SHA256SUMS", sums)
    manifest_sha256 = hashlib.sha256(sums.encode()).hexdigest()
    os.replace(stage, output_root)
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "packages": versions,
        "kernel_repository": KERNEL_REPOSITORY,
        "kernel_revision": KERNEL_REVISION,
        "kernel_module_relative": str(
            Path(imports["kernel_module"]).relative_to(stage)
        ),
        "manifest_sha256": manifest_sha256,
        **tree,
        "python": str(python),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    _atomic_json(report, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        raise GptOssOverlayError("overlay worker count differs")
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
