#!/usr/bin/env python3
"""Build the hash-bound ETTR-IL-v3 release consumed by training.

This is the bridge between audited semantic-core shards and the optimizer
contract. It fully reopens every train/development core, reconstructs its exact
tensor batch, builds a disk-backed global packet-sufficiency index, and emits a
continuation manifest plus a location index. Reserve and sealed confirmation
splits remain outside the training stream.
"""

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
from typing import Iterator, Mapping, Sequence

import torch


_ROOT = Path(__file__).resolve().parents[1]
_TRAIN = _ROOT / "train"
for _path in (_ROOT / "pipeline", _TRAIN):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ettr_data_contract import (  # noqa: E402
    ETTR_CONTINUATION_SCHEMA,
    ETTRContinuationBatch,
    ETTRContinuationManifest,
    continuation_batch_payload_sha256,
    select_continuation_rows,
)
from endogenous_typed_theory_reactor import TheoryReactorConfig  # noqa: E402
from ettr_objectives import ETTRObjectiveConfig  # noqa: E402
from ettr_il_v2_token_native_surface import (  # noqa: E402
    TokenNativeSurfaceCodec,
)
from ettr_il_v3_materialize import rematerialize_record  # noqa: E402
from ettr_il_v3_protocol import (  # noqa: E402
    PROTOCOL,
    ROWS_PER_CORE,
    SPLIT_CORES,
    canonical_json_bytes,
)
from ettr_packet_index import build_disk_packet_index  # noqa: E402
from materialize_ettr_il_v3_corpus import (  # noqa: E402
    AUDIT_SCHEMA,
    SEPARATION_SCHEMA,
    _canonical_file,
    _iter_records,
    _relative,
    _sha256_file,
    _verify_self_hash,
)


RELEASE_SCHEMA = "r12-ettr-il-v3-training-release-v2"
STREAM_RECORD_SCHEMA = "r12-ettr-il-v3-training-stream-record-v1"
TRAINING_ROWS_PER_BATCH = 16
TRAINING_BATCHES_PER_CORE = ROWS_PER_CORE // TRAINING_ROWS_PER_BATCH
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_TRAINING_SPLITS = ("train", "development")
_MAIN_SPLITS = (
    "train",
    "development",
    "train_reserve",
    "development_reserve",
)


class ETTRV3ReleaseError(ValueError):
    """An audited v3 corpus cannot be released to the trainer."""


def _hex(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ETTRV3ReleaseError(f"{label} differs")
    return value


def _commit(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        raise ETTRV3ReleaseError(f"{label} differs")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ETTRV3ReleaseError(f"{label} differs")
    return value


def _regular_identity(
    path: Path,
    label: str,
    *,
    require_immutable: bool = False,
) -> tuple[int, ...]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ETTRV3ReleaseError(f"{label} cannot be inspected") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or (require_immutable and metadata.st_mode & 0o222)
    ):
        raise ETTRV3ReleaseError(
            f"{label} is not an immutable single-link regular file"
        )
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_nlink,
    )


def _stable_file_sha256(
    path: Path,
    label: str,
    *,
    require_immutable: bool = False,
) -> tuple[str, int]:
    before = _regular_identity(
        path,
        label,
        require_immutable=require_immutable,
    )
    digest, size = _sha256_file(path)
    after = _regular_identity(
        path,
        label,
        require_immutable=require_immutable,
    )
    if before != after or size != before[2]:
        raise ETTRV3ReleaseError(f"{label} changed while being measured")
    return digest, size


def _write_no_replace(path: Path, payload: bytes) -> dict[str, object]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o400)
    try:
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return {
        "bytes": len(payload),
        "path": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _load_main_audit(path: Path) -> tuple[dict[str, object], str]:
    audit = _canonical_file(path, "main materialization audit")
    if (
        audit.get("schema") != AUDIT_SCHEMA
        or audit.get("protocol") != PROTOCOL
        or audit.get("status") != "pass"
        or audit.get("role") != "main"
    ):
        raise ETTRV3ReleaseError("main materialization audit differs")
    return audit, _verify_self_hash(
        audit,
        field="audit_sha256",
        label="main materialization audit",
    )


def _load_separation(
    path: Path,
    *,
    main_audit_sha256: str,
) -> tuple[dict[str, object], str]:
    report = _canonical_file(path, "main/confirmation separation report")
    if (
        report.get("schema") != SEPARATION_SCHEMA
        or report.get("protocol") != PROTOCOL
        or report.get("status") != "pass"
        or report.get("main_audit_sha256") != main_audit_sha256
        or not isinstance(report.get("overlap_counts"), dict)
        or any(report["overlap_counts"].values())
    ):
        raise ETTRV3ReleaseError(
            "main/confirmation separation report differs"
        )
    return report, _verify_self_hash(
        report,
        field="separation_sha256",
        label="main/confirmation separation report",
    )


def _validate_split_counts(
    audit: Mapping[str, object],
    expected: Mapping[str, int],
) -> None:
    counts = audit.get("split_counts")
    if not isinstance(counts, dict) or set(counts) != set(expected):
        raise ETTRV3ReleaseError("main audit split inventory differs")
    for split, rows in expected.items():
        if counts.get(split) != rows:
            raise ETTRV3ReleaseError(
                f"main audit {split} core count differs"
            )
    if audit.get("core_rows") != sum(expected.values()):
        raise ETTRV3ReleaseError("main audit core total differs")


def _training_shards(
    audit: Mapping[str, object],
    data_root: Path,
) -> dict[str, tuple[dict[str, object], ...]]:
    values = audit.get("shards")
    if not isinstance(values, list) or not values:
        raise ETTRV3ReleaseError("main audit shard inventory differs")
    result: dict[str, list[dict[str, object]]] = {
        split: [] for split in _TRAINING_SPLITS
    }
    observed: set[str] = set()
    for raw in values:
        if (
            not isinstance(raw, dict)
            or set(raw)
            != {"bytes", "path", "report_sha256", "rows", "sha256", "split"}
        ):
            raise ETTRV3ReleaseError("main audit shard descriptor differs")
        split = raw["split"]
        path_value = _relative(raw["path"], "main audit shard path")
        if path_value in observed:
            raise ETTRV3ReleaseError("main audit shard path repeats")
        observed.add(path_value)
        path = data_root / path_value
        digest, size = _stable_file_sha256(
            path,
            "materialized shard",
            require_immutable=True,
        )
        if (
            digest != raw["sha256"]
            or size != raw["bytes"]
            or split not in _MAIN_SPLITS
        ):
            raise ETTRV3ReleaseError(
                "materialized shard identity or split differs"
            )
        if split in result:
            result[str(split)].append(dict(raw))
    return {
        split: tuple(sorted(records, key=lambda item: str(item["path"])))
        for split, records in result.items()
    }


class _StreamWriter:
    def __init__(self, path: Path):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        self._descriptor = os.open(path, flags, 0o400)
        self._path = path
        self._digest = hashlib.sha256()
        self._bytes = 0
        self.rows = 0

    def write(self, value: object) -> None:
        payload = canonical_json_bytes(value)
        view = memoryview(payload)
        while view:
            written = os.write(self._descriptor, view)
            if written <= 0:
                raise ETTRV3ReleaseError(
                    "stream-index write made no progress"
                )
            view = view[written:]
        self._digest.update(payload)
        self._bytes += len(payload)
        self.rows += 1

    def close(self) -> dict[str, object]:
        if self._descriptor < 0:
            raise ETTRV3ReleaseError("stream writer is already closed")
        os.fsync(self._descriptor)
        os.close(self._descriptor)
        self._descriptor = -1
        return {
            "bytes": self._bytes,
            "path": self._path.name,
            "rows": self.rows,
            "sha256": self._digest.hexdigest(),
        }

    def abort(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1
        self._path.unlink(missing_ok=True)


def _training_batch_row_indices(
    core_batch: ETTRContinuationBatch,
) -> tuple[torch.Tensor, ...]:
    """Pack complete causal rectangles without splitting equivariance pairs."""

    rectangles = core_batch.causal_rectangles.rows
    if (
        rectangles.ndim != 3
        or rectangles.shape[1:] != (2, 2)
        or rectangles.numel() != ROWS_PER_CORE
        or core_batch.equivariance is None
    ):
        raise ETTRV3ReleaseError("training rectangle geometry differs")
    rectangle_count = rectangles.shape[0]
    row_to_rectangle = [-1] * ROWS_PER_CORE
    for rectangle_index, row in enumerate(rectangles.reshape(rectangle_count, 4)):
        for value in row.tolist():
            if (
                not isinstance(value, int)
                or not 0 <= value < ROWS_PER_CORE
                or row_to_rectangle[value] != -1
            ):
                raise ETTRV3ReleaseError("training rectangle row ledger differs")
            row_to_rectangle[value] = rectangle_index
    if any(value < 0 for value in row_to_rectangle):
        raise ETTRV3ReleaseError("training rectangle coverage differs")

    parent = list(range(rectangle_count))

    def root(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root = root(left)
        right_root = root(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left, right in zip(
        core_batch.equivariance.left_index.tolist(),
        core_batch.equivariance.right_index.tolist(),
        strict=True,
    ):
        union(row_to_rectangle[left], row_to_rectangle[right])

    components: dict[int, list[int]] = {}
    for rectangle_index in range(rectangle_count):
        components.setdefault(root(rectangle_index), []).append(rectangle_index)
    ordered_components = sorted(
        components.values(),
        key=lambda value: (-len(value), value[0]),
    )
    rectangles_per_batch = TRAINING_ROWS_PER_BATCH // 4
    bins: list[list[int]] = [[] for _ in range(TRAINING_BATCHES_PER_CORE)]
    for component in ordered_components:
        destination = next(
            (
                value
                for value in bins
                if len(value) + len(component) <= rectangles_per_batch
            ),
            None,
        )
        if destination is None:
            raise ETTRV3ReleaseError(
                "equivariance components cannot fit training batches"
            )
        destination.extend(component)
    if any(len(value) != rectangles_per_batch for value in bins):
        raise ETTRV3ReleaseError("training rectangle batch fill differs")

    batches = []
    flattened = rectangles.reshape(rectangle_count, 4)
    for rectangle_indices in bins:
        selected = sorted(
            row
            for rectangle_index in rectangle_indices
            for row in flattened[rectangle_index].tolist()
        )
        if len(selected) != TRAINING_ROWS_PER_BATCH:
            raise ETTRV3ReleaseError("training row batch size differs")
        batches.append(torch.tensor(selected, dtype=torch.long))
    if sorted(value for batch in batches for value in batch.tolist()) != list(
        range(ROWS_PER_CORE)
    ):
        raise ETTRV3ReleaseError("training batch partition coverage differs")
    return tuple(batches)


def _batch_stream(
    *,
    split: str,
    descriptors: Sequence[Mapping[str, object]],
    data_root: Path,
    codec: TokenNativeSurfaceCodec,
    writer: _StreamWriter,
) -> Iterator[ETTRContinuationBatch]:
    ordinal = 0
    for descriptor in descriptors:
        path_value = _relative(
            descriptor["path"],
            "training shard path",
        )
        path = data_root / path_value
        observed_rows = 0
        for row_index, (payload, record) in enumerate(_iter_records(path)):
            if (
                record.identity.split != split
                or record.canonical_bytes() != payload
            ):
                raise ETTRV3ReleaseError(
                    "training stream record split or canonical form differs"
                )
            core_batch = rematerialize_record(record, codec)
            if (
                not isinstance(core_batch, ETTRContinuationBatch)
                or core_batch.episodes.world.tokens.shape[0] != ROWS_PER_CORE
                or ROWS_PER_CORE % TRAINING_ROWS_PER_BATCH
            ):
                raise ETTRV3ReleaseError(
                    "training stream tensor geometry differs"
                )
            for batch_index, indices in enumerate(
                _training_batch_row_indices(core_batch)
            ):
                batch = select_continuation_rows(core_batch, indices)
                batch.validate(
                    TheoryReactorConfig(),
                    ETTRObjectiveConfig(
                        vocab_size=codec.tokenizer.get_vocab_size(),
                    ),
                )
                batch_sha256 = continuation_batch_payload_sha256(batch)
                writer.write(
                    {
                        "batch_index": batch_index,
                        "batch_payload_sha256": batch_sha256,
                        "core_id": record.identity.core_id,
                        "core_sha256": record.core_sha256(),
                        "ordinal": ordinal,
                        "row_index": row_index,
                        "schema": STREAM_RECORD_SCHEMA,
                        "shard_path": path_value,
                        "split": split,
                    }
                )
                ordinal += 1
                yield batch
            observed_rows += 1
        if observed_rows != descriptor["rows"]:
            raise ETTRV3ReleaseError(
                "training shard row count differs during release"
            )


def build_training_release(
    *,
    main_audit_path: Path,
    separation_path: Path,
    data_root: Path,
    tokenizer_path: Path,
    protected_checkpoint_sha256: str,
    source_commit: str,
    output: Path,
    expected_split_counts: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Fully revalidate and publish one immutable v3 training release."""

    protected_checkpoint_sha256 = _hex(
        protected_checkpoint_sha256,
        "protected checkpoint SHA-256",
    )
    source_commit = _commit(source_commit, "release source commit")
    release_builder_sha256, release_builder_bytes = _stable_file_sha256(
        Path(__file__).resolve(),
        "training-release builder",
    )
    expected = (
        {split: SPLIT_CORES[split] for split in _MAIN_SPLITS}
        if expected_split_counts is None
        else dict(expected_split_counts)
    )
    if set(expected) != set(_MAIN_SPLITS) or any(
        not isinstance(value, int) or value < 0 for value in expected.values()
    ):
        raise ETTRV3ReleaseError("expected split counts differ")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise ETTRV3ReleaseError(
            f"refusing existing release output: {output}"
        ) from exc

    audit, audit_sha256 = _load_main_audit(main_audit_path)
    separation, separation_sha256 = _load_separation(
        separation_path,
        main_audit_sha256=audit_sha256,
    )
    _validate_split_counts(audit, expected)
    codec = TokenNativeSurfaceCodec(tokenizer_path)
    if codec.tokenizer_sha256 != audit.get("tokenizer_sha256"):
        raise ETTRV3ReleaseError("training tokenizer identity differs")
    tokenizer_digest, tokenizer_bytes = _stable_file_sha256(
        tokenizer_path,
        "training tokenizer",
    )
    if tokenizer_digest != codec.tokenizer_sha256:
        raise ETTRV3ReleaseError("training tokenizer byte hash differs")

    descriptors = _training_shards(audit, data_root)
    stream_writer = _StreamWriter(output / "stream-index.jsonl")
    try:
        packet_manifest = build_disk_packet_index(
            output / "packet-index",
            train_batches=_batch_stream(
                split="train",
                descriptors=descriptors["train"],
                data_root=data_root,
                codec=codec,
                writer=stream_writer,
            ),
            validation_batches=_batch_stream(
                split="development",
                descriptors=descriptors["development"],
                data_root=data_root,
                codec=codec,
                writer=stream_writer,
            ),
        )
        stream_receipt = stream_writer.close()
    except BaseException:
        stream_writer.abort()
        raise

    packet_receipt = packet_manifest["receipt"]
    if not isinstance(packet_receipt, dict):
        raise ETTRV3ReleaseError("packet-index receipt differs")
    train_rows = expected["train"] * ROWS_PER_CORE
    validation_rows = expected["development"] * ROWS_PER_CORE
    if (
        packet_manifest["train_batches"]
        != expected["train"] * TRAINING_BATCHES_PER_CORE
        or packet_manifest["validation_batches"]
        != expected["development"] * TRAINING_BATCHES_PER_CORE
        or packet_manifest["train_rows"] != train_rows
        or packet_manifest["validation_rows"] != validation_rows
        or stream_receipt["rows"]
        != (
            expected["train"] + expected["development"]
        )
        * TRAINING_BATCHES_PER_CORE
    ):
        raise ETTRV3ReleaseError(
            "training release batch or row count differs"
        )

    manifest = ETTRContinuationManifest(
        schema=ETTR_CONTINUATION_SCHEMA,
        protected_checkpoint_sha256=protected_checkpoint_sha256,
        tokenizer_sha256=codec.tokenizer_sha256,
        qualification_payload_sha256=_hex(
            audit.get("qualification_freeze_sha256"),
            "qualification freeze SHA-256",
        ),
        hybrid_payload_sha256=separation_sha256,
        train_rows=train_rows,
        validation_rows=validation_rows,
        train_payload_sha256=str(packet_manifest["train_payload_sha256"]),
        validation_payload_sha256=str(
            packet_manifest["validation_payload_sha256"]
        ),
        dataset_sha256=ETTRContinuationManifest.combined_dataset_sha256(
            str(packet_manifest["train_payload_sha256"]),
            str(packet_manifest["validation_payload_sha256"]),
        ),
        packet_sufficiency_train_batches=_integer(
            packet_manifest["train_batches"],
            "packet-index train batches",
            1,
        ),
        packet_sufficiency_validation_batches=_integer(
            packet_manifest["validation_batches"],
            "packet-index validation batches",
            1,
        ),
        packet_sufficiency_rows=_integer(
            packet_receipt.get("rows"),
            "packet-index rows",
            1,
        ),
        packet_sufficiency_unique_contexts=_integer(
            packet_receipt.get("unique_contexts"),
            "packet-index unique contexts",
            1,
        ),
        packet_sufficiency_train_contexts=_integer(
            packet_manifest["train_contexts"],
            "packet-index train contexts",
            1,
        ),
        packet_sufficiency_validation_contexts=_integer(
            packet_manifest["validation_contexts"],
            "packet-index validation contexts",
            1,
        ),
        packet_sufficiency_context_sha256=_hex(
            packet_receipt.get("context_sha256"),
            "packet-index context SHA-256",
        ),
        packet_sufficiency_target_bound_sha256=_hex(
            packet_receipt.get("target_bound_sha256"),
            "packet-index target-bound SHA-256",
        ),
        source_deleted=True,
        immutable_snapshot=True,
        live_writer_input=False,
        family_label_fields=(),
    )
    manifest.validate()
    continuation_payload = json.dumps(
        asdict(manifest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    continuation_receipt = _write_no_replace(
        output / "continuation-manifest.json",
        continuation_payload,
    )
    if continuation_receipt["sha256"] != manifest.sha256():
        raise ETTRV3ReleaseError(
            "continuation manifest artifact hash differs"
        )
    packet_manifest_path = output / "packet-index" / "manifest.json"
    packet_manifest_sha256, packet_manifest_bytes = _stable_file_sha256(
        packet_manifest_path,
        "packet-index manifest",
    )
    training_shards = [
        dict(record)
        for split in _TRAINING_SPLITS
        for record in descriptors[split]
    ]
    release: dict[str, object] = {
        "continuation_manifest": continuation_receipt,
        "continuation_manifest_sha256": manifest.sha256(),
        "data_protocol": PROTOCOL,
        "main_audit_sha256": audit_sha256,
        "packet_index_manifest": {
            "bytes": packet_manifest_bytes,
            "path": "packet-index/manifest.json",
            "payload_sha256": packet_manifest["manifest_payload_sha256"],
            "sha256": packet_manifest_sha256,
        },
        "protected_checkpoint_sha256": protected_checkpoint_sha256,
        "release_builder": {
            "bytes": release_builder_bytes,
            "path": "pipeline/build_ettr_il_v3_training_release.py",
            "sha256": release_builder_sha256,
        },
        "schema": RELEASE_SCHEMA,
        "separation_sha256": separation_sha256,
        "source_commit": source_commit,
        "status": "pass",
        "stream_index": stream_receipt,
        "tokenizer": {
            "bytes": tokenizer_bytes,
            "sha256": codec.tokenizer_sha256,
        },
        "training_shards": training_shards,
        "training_batches_per_core": TRAINING_BATCHES_PER_CORE,
        "training_rows_per_batch": TRAINING_ROWS_PER_BATCH,
        "training_split_core_counts": {
            split: expected[split] for split in _TRAINING_SPLITS
        },
    }
    release["release_payload_sha256"] = hashlib.sha256(
        json.dumps(
            release,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    release_receipt = _write_no_replace(
        output / "release.json",
        canonical_json_bytes(release),
    )
    for path in output.iterdir():
        if path.is_file():
            path.chmod(0o400)
    output.chmod(0o500)
    return {
        **release,
        "release_file_sha256": release_receipt["sha256"],
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-audit", type=Path, required=True)
    parser.add_argument("--separation-report", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--protected-checkpoint-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    release = build_training_release(
        main_audit_path=arguments.main_audit,
        separation_path=arguments.separation_report,
        data_root=arguments.data_root,
        tokenizer_path=arguments.tokenizer,
        protected_checkpoint_sha256=arguments.protected_checkpoint_sha256,
        source_commit=arguments.source_commit,
        output=arguments.output,
    )
    print(
        json.dumps(
            {
                "continuation_manifest_sha256": release[
                    "continuation_manifest_sha256"
                ],
                "release_file_sha256": release["release_file_sha256"],
                "release_payload_sha256": release["release_payload_sha256"],
                "status": release["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
