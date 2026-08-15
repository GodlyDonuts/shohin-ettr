#!/usr/bin/env python3
"""Verify and remove one redownloadable dense base after its evidence closes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import stat

from restore_dense_public_model import sha256_file, tree_manifest


class DenseModelReclaimError(RuntimeError):
    """The exact restored model is not safe to reclaim."""


def run(args: argparse.Namespace) -> dict[str, object]:
    if (
        args.output.exists()
        or not args.model_root.is_dir()
        or args.model_root.is_symlink()
    ):
        raise DenseModelReclaimError("reclaim target or output state differs")
    receipt = json.loads(args.model_receipt.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != "shohin-dense-model-restoration-v1"
        or receipt.get("status") != "complete"
        or Path(receipt.get("model_root", "")).resolve() != args.model_root.resolve()
    ):
        raise DenseModelReclaimError("restoration receipt differs")
    rows, tree_sha256, total_bytes = tree_manifest(args.model_root)
    if (
        tree_sha256 != receipt.get("tree_sha256")
        or len(rows) != receipt.get("files")
        or total_bytes != receipt.get("bytes")
    ):
        raise DenseModelReclaimError("model tree changed before reclaim")
    quarantine = args.model_root.with_name(
        f".{args.model_root.name}.dense-public-reclaim.{os.getpid()}"
    )
    if quarantine.exists():
        raise DenseModelReclaimError("reclaim quarantine already exists")
    os.replace(args.model_root, quarantine)
    for path in sorted(quarantine.rglob("*"), reverse=True):
        path.chmod(0o700 if stat.S_ISDIR(path.lstat().st_mode) else 0o600)
    quarantine.chmod(0o700)
    shutil.rmtree(quarantine)
    payload = {
        "schema": "shohin-dense-model-reclaim-v1",
        "status": "complete",
        "model_root": str(args.model_root.resolve()),
        "model_receipt": str(args.model_receipt.resolve()),
        "model_receipt_sha256": sha256_file(args.model_receipt),
        "tree_sha256": tree_sha256,
        "files_removed": len(rows),
        "bytes_removed": total_bytes,
        "redownloadable_repository": receipt["repository"],
        "redownloadable_revision": receipt["model_revision"],
        "locally_recoverable": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    payload = run(parse_args())
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
