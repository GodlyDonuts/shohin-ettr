#!/usr/bin/env python3
"""Restore one exact dense base model and publish a complete immutable manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

SCHEMA = "shohin-dense-model-restoration-v1"


class DenseModelRestorationError(RuntimeError):
    """The requested revision or restored tree differs from the frozen model."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(root: Path) -> tuple[list[dict[str, object]], str, int]:
    rows: list[dict[str, object]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not (
            stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
        ):
            raise DenseModelRestorationError(
                f"restored model contains a link or special file: {relative}"
            )
        if stat.S_ISREG(info.st_mode):
            rows.append(
                {"path": relative, "bytes": info.st_size, "sha256": sha256_file(path)}
            )
            total_bytes += info.st_size
    if not rows or not any(row["path"] == "config.json" for row in rows):
        raise DenseModelRestorationError("restored model is empty or lacks config.json")
    encoded = b"".join((f"{row['sha256']}  {row['path']}\n".encode() for row in rows))
    return rows, hashlib.sha256(encoded).hexdigest(), total_bytes


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise DenseModelRestorationError(f"refusing to replace {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, object]:
    from huggingface_hub import snapshot_download

    if args.model_root.exists() or args.receipt.exists():
        raise DenseModelRestorationError(
            "model root or restoration receipt already exists"
        )
    stage = args.model_root.with_name(f".{args.model_root.name}.stage.{os.getpid()}")
    if stage.exists():
        raise DenseModelRestorationError("restoration staging path already exists")
    stage.mkdir(parents=True, mode=0o700)
    try:
        snapshot_download(
            repo_id=args.repository,
            revision=args.revision,
            local_dir=stage,
            token=os.environ.get("HF_TOKEN"),
        )
        metadata = stage / ".cache"
        if metadata.exists():
            shutil.rmtree(metadata)
        rows, tree_sha256, total_bytes = tree_manifest(stage)
        config_sha256 = sha256_file(stage / "config.json")
        if config_sha256 != args.config_sha256:
            raise DenseModelRestorationError("restored model config hash differs")
        manifest = stage.with_name(f".{stage.name}.manifest.json")
        atomic_json(
            manifest,
            {
                "schema": "shohin-dense-model-manifest-v1",
                "repository": args.repository,
                "revision": args.revision,
                "files": rows,
                "tree_sha256": tree_sha256,
                "bytes": total_bytes,
            },
        )
        for path in stage.rglob("*"):
            path.chmod(0o555 if path.is_dir() else 0o444)
        stage.chmod(0o555)
        os.replace(stage, args.model_root)
        final_manifest = args.model_root.with_name(
            f"{args.model_root.name}.manifest.json"
        )
        os.replace(manifest, final_manifest)
        final_manifest.chmod(0o444)
        receipt = {
            "schema": SCHEMA,
            "status": "complete",
            "repository": args.repository,
            "model_revision": args.revision,
            "model_root": str(args.model_root.resolve()),
            "config_sha256": config_sha256,
            "manifest": str(final_manifest.resolve()),
            "manifest_sha256": sha256_file(final_manifest),
            "manifest_verified": True,
            "tree_sha256": tree_sha256,
            "files": len(rows),
            "bytes": total_bytes,
            "symlinks": 0,
            "special_files": 0,
        }
        atomic_json(args.receipt, receipt)
        args.receipt.chmod(0o444)
        return receipt
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    receipt = run(parse_args())
    print(json.dumps({key: receipt[key] for key in ("model_root", "files", "bytes")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
