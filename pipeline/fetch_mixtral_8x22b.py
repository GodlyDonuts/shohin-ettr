#!/usr/bin/env python3
"""Acquire and seal the exact Mixtral-8x22B snapshot for upward MoE scaling."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any

MODEL_ID = "mistralai/Mixtral-8x22B-Instruct-v0.1"
MODEL_REVISION = "cc88a6cc19fbd17d9f1c0ee0b0d70a748dce698d"
MODEL_CONFIG_SHA256 = "9c4a6138d84029ab666943613e3d5844d2ea8fd6149f44f77188c62e2915e0f5"
EXPECTED_SIBLINGS = 69
EXPECTED_WEIGHT_SHARDS = 59
EXPECTED_WEIGHT_BYTES = 281_260_367_720
EXPECTED_LFS_BYTES = 281_261_542_528
SCHEMA = "shohin-mixtral-8x22b-acquisition-v1"

SUPPORT_MEMBERS = {
    ".gitattributes",
    "README.md",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer.model.v3",
    "tokenizer_config.json",
}
WEIGHT_MEMBERS = {
    f"model-{index:05d}-of-{EXPECTED_WEIGHT_SHARDS:05d}.safetensors"
    for index in range(1, EXPECTED_WEIGHT_SHARDS + 1)
}
EXPECTED_MEMBERS = SUPPORT_MEMBERS | WEIGHT_MEMBERS


class MixtralAcquisitionError(RuntimeError):
    """The immutable Mixtral acquisition contract differed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise MixtralAcquisitionError("refusing existing acquisition report")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sibling_receipt(info: Any) -> dict[str, tuple[str | None, int | None]]:
    if getattr(info, "sha", None) != MODEL_REVISION:
        raise MixtralAcquisitionError("resolved model revision differs")
    siblings = getattr(info, "siblings", None)
    if not isinstance(siblings, list) or len(siblings) != EXPECTED_SIBLINGS:
        raise MixtralAcquisitionError("repository sibling count differs")
    rows: dict[str, tuple[str | None, int | None]] = {}
    for sibling in siblings:
        name = getattr(sibling, "rfilename", None)
        lfs = getattr(sibling, "lfs", None)
        digest = getattr(lfs, "sha256", None) if lfs is not None else None
        size = getattr(lfs, "size", None) if lfs is not None else None
        if (
            not isinstance(name, str)
            or name in rows
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or Path(name).as_posix() != name
            or (
                digest is not None
                and (not isinstance(digest, str) or len(digest) != 64)
            )
            or (size is not None and (not isinstance(size, int) or size <= 0))
        ):
            raise MixtralAcquisitionError("repository sibling metadata differs")
        rows[name] = (digest, size)
    if set(rows) != EXPECTED_MEMBERS:
        raise MixtralAcquisitionError("repository membership differs")
    weight_bytes = sum(int(rows[name][1] or 0) for name in WEIGHT_MEMBERS)
    lfs_bytes = sum(int(size or 0) for _, size in rows.values())
    if (
        weight_bytes != EXPECTED_WEIGHT_BYTES
        or lfs_bytes != EXPECTED_LFS_BYTES
        or any(rows[name][0] is None for name in WEIGHT_MEMBERS)
    ):
        raise MixtralAcquisitionError("repository LFS geometry differs")
    return rows


def _regular_members(root: Path) -> set[str]:
    members: set[str] = set()
    for candidate in root.rglob("*"):
        mode = candidate.lstat().st_mode
        if stat.S_ISDIR(mode) and not candidate.is_symlink():
            continue
        if not stat.S_ISREG(mode) or candidate.is_symlink():
            raise MixtralAcquisitionError(
                "downloaded tree contains a nonregular member"
            )
        members.add(candidate.relative_to(root).as_posix())
    return members


def seal_snapshot(
    stage: Path,
    output: Path,
    rows: dict[str, tuple[str | None, int | None]],
) -> dict[str, Any]:
    cache = stage / ".cache"
    if cache.exists():
        if cache.is_symlink() or not cache.is_dir():
            raise MixtralAcquisitionError("snapshot cache geometry differs")
        shutil.rmtree(cache)
    if _regular_members(stage) != EXPECTED_MEMBERS:
        raise MixtralAcquisitionError("downloaded snapshot membership differs")
    for name, (expected_digest, expected_size) in rows.items():
        candidate = stage / name
        if expected_size is not None and candidate.stat().st_size != expected_size:
            raise MixtralAcquisitionError("downloaded LFS size differs")
        if expected_digest is not None and sha256_file(candidate) != expected_digest:
            raise MixtralAcquisitionError("downloaded LFS digest differs")
    if sha256_file(stage / "config.json") != MODEL_CONFIG_SHA256:
        raise MixtralAcquisitionError("downloaded config hash differs")

    source_revision = stage / "SOURCE_REVISION"
    source_revision.write_text(MODEL_REVISION + "\n", encoding="utf-8")
    with source_revision.open("r+", encoding="utf-8") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    members = sorted(EXPECTED_MEMBERS | {"SOURCE_REVISION"})
    manifest = stage / "SHA256SUMS"
    with manifest.open("x", encoding="utf-8") as handle:
        for name in members:
            handle.write(f"{sha256_file(stage / name)}  {name}\n")
        handle.flush()
        os.fsync(handle.fileno())
    if _regular_members(stage) != set(members) | {"SHA256SUMS"}:
        raise MixtralAcquisitionError("sealed snapshot membership differs")
    covered_bytes = sum((stage / name).stat().st_size for name in members)
    manifest_sha256 = sha256_file(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, output)
    return {
        "model_root": str(output.resolve()),
        "model_manifest": str((output / "SHA256SUMS").resolve()),
        "model_manifest_sha256": manifest_sha256,
        "manifest_entries": len(members),
        "covered_bytes": covered_bytes,
        "exact_membership": True,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    output = args.output.resolve(strict=False)
    report = args.report.resolve(strict=False)
    if (
        output.exists()
        or output.is_symlink()
        or report.exists()
        or report.is_symlink()
        or output.parent != report.parent
    ):
        raise MixtralAcquisitionError("acquisition output geometry differs")
    job_id = os.environ.get("SLURM_JOB_ID")
    if not isinstance(job_id, str) or not job_id.isdecimal():
        raise MixtralAcquisitionError("SLURM job identity is required")
    stage = output.with_name(f".{output.name}.partial.{job_id}")
    if stage.exists() or stage.is_symlink():
        raise MixtralAcquisitionError("acquisition staging path exists")

    info = HfApi().model_info(
        MODEL_ID,
        revision=MODEL_REVISION,
        files_metadata=True,
    )
    rows = _sibling_receipt(info)
    resolved = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=stage,
            max_workers=args.workers,
        )
    ).resolve(strict=True)
    if resolved != stage.resolve(strict=True):
        raise MixtralAcquisitionError("snapshot download root differs")
    model_receipt = seal_snapshot(stage, output, rows)
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "official_siblings": EXPECTED_SIBLINGS,
        "weight_shards": EXPECTED_WEIGHT_SHARDS,
        "official_weight_bytes": EXPECTED_WEIGHT_BYTES,
        "official_lfs_bytes": EXPECTED_LFS_BYTES,
        "config_sha256": MODEL_CONFIG_SHA256,
        "workers": args.workers,
        **model_receipt,
    }
    _atomic_json(report, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8, choices=range(1, 17))
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), sort_keys=True))
