#!/usr/bin/env python3
"""Package the exact, single-dispatch Q36-MTR runtime from a clean commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess

SCHEMA = "shohin-q36-mtr-runtime-v1"
PROHIBITED = (
    "ndr1",
    "kcr1",
    "vte1",
    "microcode",
    "q35",
    "olmoe",
    "edit_selector",
    "edit-selector",
    "dispatch_pcf1",
)
REQUIRED = {
    "pipeline/authorize_q36_mtr_phase.py",
    "pipeline/build_q36_mtr_commit_pairs.py",
    "pipeline/build_q36_mtr_custody.py",
    "pipeline/build_q36_mtr_data.py",
    "pipeline/capture_q36_mtr_accounting.py",
    "pipeline/capture_q36_mtr_cluster_preflight.py",
    "pipeline/capture_q36_mtr_environment.py",
    "pipeline/compare_q36_mtr.py",
    "pipeline/compile_q36_mtr_plan.py",
    "pipeline/dispatch_q36_mtr.py",
    "pipeline/jobs/q36_mtr_live_preflight.sbatch",
    "pipeline/jobs/q36_mtr_prepare_phase.sh",
    "pipeline/merge_q36_mtr_drafts.py",
    "pipeline/merge_q36_mtr_evaluations.py",
    "pipeline/mirror_q36_mtr_evidence.py",
    "pipeline/normalize_q36_mtr_score.py",
    "pipeline/q36_mtr_evidence.py",
    "pipeline/render_q36_mtr_publication_figure.py",
    "pipeline/score_q36_mtr.py",
    "pipeline/seal_q36_mtr_terminal_evidence.py",
    "pipeline/validate_q36_mtr_commit_application.py",
    "pipeline/validate_q36_mtr_live_preflight.py",
    "train/hf_q36_mtr_evaluate.py",
    "train/hf_q36_mtr_generate_drafts.py",
    "train/hf_q36_mtr_mechanics.py",
    "train/hf_q36_mtr_train_commit.py",
    "train/hf_q36_mtr_train_role.py",
    "train/jobs/q36_mtr_common.sh",
    "train/pcf1_code_sandbox.py",
    "train/q36_mtr_roles.py",
    "train/shared_post_mlp_revision.py",
}


class Q36MTRRuntimeError(RuntimeError):
    """The Q36-MTR runtime closure or repository identity differs."""


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
        raise Q36MTRRuntimeError("Q36-MTR allowlist must be sorted and unique")
    for entry in entries:
        pure = PurePosixPath(entry)
        if (
            pure.is_absolute()
            or not pure.parts
            or "." in pure.parts
            or ".." in pure.parts
            or any(term in entry.casefold() for term in PROHIBITED)
        ):
            raise Q36MTRRuntimeError(f"prohibited Q36-MTR runtime member: {entry}")
    missing = REQUIRED - set(entries)
    if missing:
        raise Q36MTRRuntimeError(
            f"missing Q36-MTR runtime dependencies: {sorted(missing)}"
        )
    return entries


def package(source_root: Path, allowlist: Path, output: Path) -> str:
    if output.exists() or output.is_symlink():
        raise Q36MTRRuntimeError(f"refusing existing Q36-MTR runtime: {output}")
    entries = load_allowlist(allowlist)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=source_root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if status:
        raise Q36MTRRuntimeError("Q36-MTR runtime source repository is dirty")
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
        raise Q36MTRRuntimeError("Q36-MTR source commit differs")
    for entry in entries:
        source = source_root / entry
        if not source.is_file() or source.is_symlink():
            raise Q36MTRRuntimeError(f"missing Q36-MTR runtime source: {entry}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise Q36MTRRuntimeError("Q36-MTR runtime temporary output exists")
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
        payload = {
            "schema": SCHEMA,
            "status": "complete",
            "source_commit": source_commit,
            "allowlist_sha256": sha256_file(allowlist),
            "allowlisted_files": entries,
            "scientific_submit_capability": True,
            "submission_count": 1,
            "model_acquisition_capability": False,
            "extra_files_permitted": False,
        }
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        with (temporary / "runtime.json").open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        with (temporary / "SHA256SUMS").open("xb") as handle:
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
    print(package(args.source_root, args.allowlist, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
