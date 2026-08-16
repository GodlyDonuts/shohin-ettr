#!/usr/bin/env python3
"""Package the deterministic allowlisted PCF1 runtime and SHA256SUMS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess

PROHIBITED = ("ndr1", "kcr1", "vte1", "microcode", "q35", "olmoe")
PROTECTED_PATHS = ("holdout", "product", "public")
QUALIFIED_SHARED = {
    "train/hf_product_reasoning_eval.py",
    "train/hf_product_reasoning_train.py",
    "train/hf_aqc1_train_commit.py",
    "train/hf_cvg1_completion_verifier.py",
    "train/integrated_reasoning_workspace.py",
}


class PCF1RuntimeError(RuntimeError):
    """The publication runtime allowlist or output differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_allowlist(path: Path) -> list[str]:
    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if entries != sorted(entries) or len(entries) != len(set(entries)):
        raise PCF1RuntimeError("PCF1 runtime allowlist must be sorted and unique")
    for entry in entries:
        pure = PurePosixPath(entry)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or not pure.parts
            or any(term in entry.casefold() for term in PROHIBITED)
        ):
            raise PCF1RuntimeError(f"prohibited PCF1 runtime member: {entry}")
    missing_shared = QUALIFIED_SHARED - set(entries)
    if missing_shared:
        raise PCF1RuntimeError(
            f"missing qualified shared runtime modules: {sorted(missing_shared)}"
        )
    return entries


def package(source_root: Path, allowlist: Path, output: Path) -> str:
    for path in (source_root, allowlist, output):
        rendered = f"{path}\n{path.resolve(strict=False)}".casefold()
        if any(term in rendered for term in PROTECTED_PATHS):
            raise PCF1RuntimeError(f"protected PCF1 runtime path: {path}")
    if output.exists() or output.is_symlink():
        raise PCF1RuntimeError(f"refusing existing PCF1 runtime: {output}")
    entries = load_allowlist(allowlist)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=source_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if status:
        raise PCF1RuntimeError("PCF1 runtime source repository is dirty")
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise PCF1RuntimeError("PCF1 runtime source commit differs")
    for entry in entries:
        source = source_root / entry
        if not source.is_file() or source.is_symlink():
            raise PCF1RuntimeError(f"missing or linked PCF1 runtime source: {entry}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise PCF1RuntimeError("PCF1 runtime temporary output exists")
    temporary.mkdir()
    try:
        for entry in entries:
            source = source_root / entry
            destination = temporary / entry
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as reader, destination.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1 << 20)
                writer.flush()
                os.fsync(writer.fileno())
            destination.chmod(source.stat().st_mode & 0o777)
        runtime_payload = {
            "schema": "shohin-pcf1-runtime-v1",
            "status": "complete",
            "source_commit": source_commit,
            "allowlist_sha256": sha256_file(allowlist),
            "allowlisted_files": entries,
            "extra_files_permitted": False,
        }
        runtime_path = temporary / "runtime.json"
        runtime_encoded = (
            json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n"
        ).encode()
        with runtime_path.open("xb") as handle:
            handle.write(runtime_encoded)
            handle.flush()
            os.fsync(handle.fileno())
        manifest = temporary / "SHA256SUMS"
        with manifest.open("xb") as handle:
            for entry in sorted((*entries, "runtime.json")):
                handle.write(f"{sha256_file(temporary / entry)}  {entry}\n".encode())
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(temporary, output)
        parent_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode & ~0o222)
    for path in sorted(
        (path for path in output.rglob("*") if path.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    ):
        path.chmod(path.stat().st_mode & ~0o222)
    output.chmod(output.stat().st_mode & ~0o222)
    return sha256_file(output / "SHA256SUMS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    digest = package(args.source_root, args.allowlist, args.output)
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
