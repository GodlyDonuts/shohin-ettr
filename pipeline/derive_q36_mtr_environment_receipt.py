#!/usr/bin/env python3
"""Derive a zero-read Q36 environment receipt for an immutable runtime package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

SCHEMA = "shohin-q36-mtr-environment-v1"
DERIVATION = "runtime_rebind_no_scientific_reads_v1"


class Q36MTREnvironmentDerivationError(RuntimeError):
    """The base receipt or target runtime cannot support a derived receipt."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def derive(
    base_receipt: Path,
    runtime: Path,
    runtime_manifest_sha256: str,
    output: Path,
) -> dict:
    base = base_receipt.resolve(strict=True)
    target = runtime.resolve(strict=True)
    manifest = target / "SHA256SUMS"
    if (
        base.is_symlink()
        or target.is_symlink()
        or not base.is_file()
        or not target.is_dir()
        or manifest.is_symlink()
        or not manifest.is_file()
        or len(runtime_manifest_sha256) != 64
        or sha256_file(manifest) != runtime_manifest_sha256
    ):
        raise Q36MTREnvironmentDerivationError("runtime receipt input differs")
    payload = json.loads(base.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "pass"
        or payload.get("scientific_rows_read") != 0
        or payload.get("offline_required") is not True
        or payload.get("bytecode_writes_permitted") is not False
    ):
        raise Q36MTREnvironmentDerivationError("base environment receipt differs")
    payload["runtime_root"] = str(target)
    payload["runtime_manifest_sha256"] = runtime_manifest_sha256
    payload["derived_from_receipt_sha256"] = sha256_file(base)
    payload["derivation"] = DERIVATION
    if output.exists() or output.is_symlink():
        raise Q36MTREnvironmentDerivationError("derived environment output exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            derive(
                args.base_receipt,
                args.runtime,
                args.runtime_manifest_sha256,
                args.output,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
