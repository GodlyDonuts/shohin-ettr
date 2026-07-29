"""Independently verify one immutable ETTR-IL-v3 training release."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Sequence


_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "train", _ROOT / "pipeline"):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

from ettr_il_v3_protocol import canonical_json_bytes  # noqa: E402
from ettr_packet_index import (  # noqa: E402
    ETTRDiskPacketSufficiencyIndex,
)
from ettr_v3_streaming import ETTRV3StreamingRelease  # noqa: E402


SCHEMA = "r12-ettr-il-v3-training-release-verification-v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ETTRV3ReleaseVerificationError(ValueError):
    """The independently observed release differs from its frozen contract."""


def _hex(value: str, label: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ETTRV3ReleaseVerificationError(f"{label} differs")
    return value


def _identity(path: Path, label: str, *, immutable: bool) -> tuple[int, ...]:
    try:
        value = path.lstat()
    except OSError as exc:
        raise ETTRV3ReleaseVerificationError(
            f"{label} cannot be inspected"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(value.st_mode)
        or value.st_nlink != 1
        or (immutable and value.st_mode & 0o222)
    ):
        raise ETTRV3ReleaseVerificationError(
            f"{label} is not an immutable single-link regular file"
        )
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _stable_sha256(
    path: Path,
    label: str,
    *,
    immutable: bool,
) -> tuple[str, int]:
    before = _identity(path, label, immutable=immutable)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    after = _identity(path, label, immutable=immutable)
    if before != after:
        raise ETTRV3ReleaseVerificationError(
            f"{label} changed while being measured"
        )
    return digest.hexdigest(), before[4]


def _write_no_replace(path: Path, value: dict[str, object]) -> None:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o400)
    with os.fdopen(descriptor, "wb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())


def _first_batch(
    stream: ETTRV3StreamingRelease,
    split: str,
):
    iterator = stream.iter_batches(
        split,
        rank=0,
        world_size=1,
        epoch=0,
        seed=0,
    )
    try:
        return next(iterator)
    except StopIteration as exc:
        raise ETTRV3ReleaseVerificationError(
            f"{split} stream is empty"
        ) from exc
    finally:
        iterator.close()


def verify_training_release(
    *,
    release_root: Path,
    data_root: Path,
    tokenizer_path: Path,
    release_source_root: Path,
    expected_release_source_commit: str,
    verifier_source_commit: str,
    expected_protected_checkpoint_sha256: str,
    receipt_path: Path,
) -> dict[str, object]:
    """Verify release, source shards, packet index, and sampled streaming."""

    expected_release_source_commit = _hex(
        expected_release_source_commit,
        "expected release source commit",
        _HEX40,
    )
    verifier_source_commit = _hex(
        verifier_source_commit,
        "verifier source commit",
        _HEX40,
    )
    expected_protected_checkpoint_sha256 = _hex(
        expected_protected_checkpoint_sha256,
        "expected protected checkpoint SHA-256",
        _HEX64,
    )
    release_root = release_root.resolve()
    data_root = data_root.resolve()
    tokenizer_path = tokenizer_path.resolve()
    release_source_root = release_source_root.resolve()
    if (
        not release_root.is_dir()
        or release_root.is_symlink()
        or release_root.stat().st_mode & 0o222
        or not data_root.is_dir()
        or data_root.is_symlink()
        or not release_source_root.is_dir()
        or release_source_root.is_symlink()
    ):
        raise ETTRV3ReleaseVerificationError(
            "release, data, or source root differs"
        )

    release_sha256, release_bytes = _stable_sha256(
        release_root / "release.json",
        "training release",
        immutable=True,
    )
    stream = ETTRV3StreamingRelease(
        release_root,
        expected_release_sha256=release_sha256,
        data_root=data_root,
        tokenizer_path=tokenizer_path,
    )
    if (
        stream.release.get("source_commit")
        != expected_release_source_commit
        or stream.release.get("protected_checkpoint_sha256")
        != expected_protected_checkpoint_sha256
    ):
        raise ETTRV3ReleaseVerificationError(
            "release source or protected checkpoint differs"
        )

    builder_receipt = stream.release.get("release_builder")
    builder_path = (
        release_source_root
        / "pipeline"
        / "build_ettr_il_v3_training_release.py"
    )
    builder_sha256, builder_bytes = _stable_sha256(
        builder_path,
        "release builder source",
        immutable=False,
    )
    if (
        not isinstance(builder_receipt, dict)
        or builder_receipt.get("sha256") != builder_sha256
        or builder_receipt.get("bytes") != builder_bytes
    ):
        raise ETTRV3ReleaseVerificationError(
            "release builder source differs"
        )

    source_verification = stream.verify_source_shards()
    train_batch = _first_batch(stream, "train")
    development_batch = _first_batch(stream, "development")
    with ETTRDiskPacketSufficiencyIndex(
        release_root / "packet-index"
    ) as packet_index:
        if (
            packet_index.receipt
            != stream.manifest.packet_sufficiency_receipt()
            or packet_index.train_rows != stream.manifest.train_rows
            or packet_index.validation_rows
            != stream.manifest.validation_rows
        ):
            raise ETTRV3ReleaseVerificationError(
                "packet-index and continuation manifest differ"
            )
        packet_index.verify_train((train_batch,))
        packet_index.verify_validation((development_batch,))
        packet_verification = {
            "receipt": asdict(packet_index.receipt),
            "train_batches": packet_index.train_batches,
            "train_contexts": packet_index.train_contexts,
            "train_rows": packet_index.train_rows,
            "validation_batches": packet_index.validation_batches,
            "validation_contexts": packet_index.validation_contexts,
            "validation_rows": packet_index.validation_rows,
        }

    verifier_sha256, verifier_bytes = _stable_sha256(
        Path(__file__).resolve(),
        "training-release verifier",
        immutable=False,
    )
    receipt: dict[str, object] = {
        "packet_index": packet_verification,
        "protected_checkpoint_sha256": (
            expected_protected_checkpoint_sha256
        ),
        "release_bytes": release_bytes,
        "release_file_sha256": release_sha256,
        "release_payload_sha256": stream.release[
            "release_payload_sha256"
        ],
        "release_source_commit": expected_release_source_commit,
        "schema": SCHEMA,
        "source_verification": source_verification,
        "status": "pass",
        "verifier": {
            "bytes": verifier_bytes,
            "path": "pipeline/verify_ettr_il_v3_training_release.py",
            "sha256": verifier_sha256,
            "source_commit": verifier_source_commit,
        },
    }
    receipt["payload_sha256"] = hashlib.sha256(
        canonical_json_bytes(receipt)
    ).hexdigest()
    _write_no_replace(receipt_path, receipt)
    return receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--release-source-root", type=Path, required=True)
    parser.add_argument("--expected-release-source-commit", required=True)
    parser.add_argument("--verifier-source-commit", required=True)
    parser.add_argument("--expected-protected-checkpoint-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    receipt = verify_training_release(
        release_root=arguments.release_root,
        data_root=arguments.data_root,
        tokenizer_path=arguments.tokenizer,
        release_source_root=arguments.release_source_root,
        expected_release_source_commit=(
            arguments.expected_release_source_commit
        ),
        verifier_source_commit=arguments.verifier_source_commit,
        expected_protected_checkpoint_sha256=(
            arguments.expected_protected_checkpoint_sha256
        ),
        receipt_path=arguments.receipt,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
