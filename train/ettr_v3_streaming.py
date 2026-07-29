"""Hash-bound streaming loader for an admitted ETTR-IL-v3 release."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Iterator, Mapping

import torch
from tokenizers import Tokenizer

from ettr_data_contract import (
    ETTRContinuationBatch,
    ETTRContinuationManifest,
    continuation_batch_payload_sha256,
    select_continuation_rows,
)
from ettr_il_v2_token_native_surface import TokenNativeSurfaceCodec


_ROOT = Path(__file__).resolve().parents[1]
_PIPELINE = _ROOT / "pipeline"
if str(_PIPELINE) not in sys.path:
    sys.path.insert(0, str(_PIPELINE))

from build_ettr_il_v3_training_release import (  # noqa: E402
    RELEASE_SCHEMA,
    STREAM_RECORD_SCHEMA,
    TRAINING_BATCHES_PER_CORE,
    TRAINING_ROWS_PER_BATCH,
)
from ettr_il_v3_materialize import rematerialize_record  # noqa: E402
from ettr_il_v3_protocol import canonical_json_bytes  # noqa: E402
from materialize_ettr_il_v3_corpus import (  # noqa: E402
    _iter_records,
    _sha256_file,
)


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SPLITS = ("train", "development")


class ETTRV3StreamingError(ValueError):
    """The release or one streamed batch differs from its frozen receipt."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _hex(value: object, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ETTRV3StreamingError(f"{label} differs")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ETTRV3StreamingError(f"{label} differs")
    return value


def _relative(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ETTRV3StreamingError(f"{label} differs")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ETTRV3StreamingError(f"{label} is unsafe")
    return value


def _identity(
    path: Path,
    label: str,
    *,
    require_immutable: bool,
) -> tuple[int, ...]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ETTRV3StreamingError(f"{label} cannot be inspected") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or (require_immutable and metadata.st_mode & 0o222)
    ):
        raise ETTRV3StreamingError(
            f"{label} is not an immutable single-link regular file"
        )
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stable_payload(
    path: Path,
    label: str,
    *,
    require_immutable: bool,
) -> tuple[bytes, str]:
    before = _identity(
        path,
        label,
        require_immutable=require_immutable,
    )
    payload = path.read_bytes()
    after = _identity(
        path,
        label,
        require_immutable=require_immutable,
    )
    if before != after or len(payload) != before[4]:
        raise ETTRV3StreamingError(f"{label} changed while being read")
    return payload, hashlib.sha256(payload).hexdigest()


def _load_release(path: Path, expected_sha256: str) -> dict[str, object]:
    expected_sha256 = _hex(expected_sha256, "expected release SHA-256")
    payload, digest = _stable_payload(
        path,
        "ETTR v3 release",
        require_immutable=True,
    )
    if digest != expected_sha256:
        raise ETTRV3StreamingError("ETTR v3 release artifact hash differs")
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ETTRV3StreamingError("ETTR v3 release is malformed") from exc
    if (
        not isinstance(value, dict)
        or canonical_json_bytes(value) != payload
        or value.get("schema") != RELEASE_SCHEMA
        or value.get("status") != "pass"
    ):
        raise ETTRV3StreamingError("ETTR v3 release contract differs")
    claimed = value.get("release_payload_sha256")
    unsigned = dict(value)
    unsigned.pop("release_payload_sha256", None)
    if (
        not isinstance(claimed, str)
        or _HEX64.fullmatch(claimed) is None
        or hashlib.sha256(_canonical_json_bytes(unsigned)).hexdigest()
        != claimed
    ):
        raise ETTRV3StreamingError("ETTR v3 release self-hash differs")
    return value


def _load_continuation(
    root: Path,
    release: Mapping[str, object],
) -> ETTRContinuationManifest:
    descriptor = release.get("continuation_manifest")
    if not isinstance(descriptor, Mapping):
        raise ETTRV3StreamingError(
            "continuation manifest descriptor differs"
        )
    path = root / _relative(
        descriptor.get("path"),
        "continuation manifest path",
    )
    payload, digest = _stable_payload(
        path,
        "continuation manifest",
        require_immutable=True,
    )
    if (
        digest != descriptor.get("sha256")
        or len(payload) != descriptor.get("bytes")
        or digest != release.get("continuation_manifest_sha256")
    ):
        raise ETTRV3StreamingError(
            "continuation manifest identity differs"
        )
    try:
        value = json.loads(payload.decode("ascii"))
        if not isinstance(value, dict):
            raise TypeError
        if isinstance(value.get("family_label_fields"), list):
            value["family_label_fields"] = tuple(
                value["family_label_fields"]
            )
        manifest = ETTRContinuationManifest(**value)
        manifest.validate()
    except (TypeError, ValueError) as exc:
        raise ETTRV3StreamingError(
            "continuation manifest contract differs"
        ) from exc
    if manifest.sha256() != digest:
        raise ETTRV3StreamingError(
            "continuation manifest logical hash differs"
        )
    return manifest


def _load_stream_index(
    root: Path,
    release: Mapping[str, object],
) -> dict[str, tuple[dict[str, object], ...]]:
    descriptor = release.get("stream_index")
    if not isinstance(descriptor, Mapping):
        raise ETTRV3StreamingError("stream-index descriptor differs")
    path = root / _relative(descriptor.get("path"), "stream-index path")
    payload, digest = _stable_payload(
        path,
        "stream index",
        require_immutable=True,
    )
    if (
        digest != descriptor.get("sha256")
        or len(payload) != descriptor.get("bytes")
    ):
        raise ETTRV3StreamingError("stream-index identity differs")
    rows: dict[str, list[dict[str, object]]] = {
        split: [] for split in _SPLITS
    }
    for line_number, line in enumerate(payload.splitlines(keepends=True), 1):
        try:
            value = json.loads(line.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ETTRV3StreamingError(
                f"stream-index row {line_number} is malformed"
            ) from exc
        if (
            not isinstance(value, dict)
            or canonical_json_bytes(value) != line
            or set(value)
            != {
                "batch_payload_sha256",
                "batch_index",
                "core_id",
                "core_sha256",
                "ordinal",
                "row_index",
                "schema",
                "shard_path",
                "split",
            }
            or value.get("schema") != STREAM_RECORD_SCHEMA
            or value.get("split") not in _SPLITS
        ):
            raise ETTRV3StreamingError(
                f"stream-index row {line_number} differs"
            )
        _hex(value["batch_payload_sha256"], "stream batch SHA-256")
        _hex(value["core_sha256"], "stream core SHA-256")
        _integer(value["batch_index"], "stream batch index")
        _integer(value["ordinal"], "stream ordinal")
        _integer(value["row_index"], "stream row index")
        _relative(value["shard_path"], "stream shard path")
        if not isinstance(value["core_id"], str) or not value["core_id"]:
            raise ETTRV3StreamingError("stream core ID differs")
        rows[str(value["split"])].append(value)
    if sum(len(values) for values in rows.values()) != descriptor.get("rows"):
        raise ETTRV3StreamingError("stream-index row count differs")
    expected_counts = release.get("training_split_core_counts")
    if not isinstance(expected_counts, Mapping):
        raise ETTRV3StreamingError("release split counts differ")
    for split, values in rows.items():
        expected_rows = (
            expected_counts.get(split) * TRAINING_BATCHES_PER_CORE
            if isinstance(expected_counts.get(split), int)
            else None
        )
        if (
            len(values) != expected_rows
            or [value["ordinal"] for value in values]
            != list(range(len(values)))
            or len(
                {
                    (
                        value["shard_path"],
                        value["row_index"],
                        value["batch_index"],
                    )
                    for value in values
                }
            )
            != len(values)
        ):
            raise ETTRV3StreamingError(
                f"stream-index {split} population differs"
            )
    return {split: tuple(values) for split, values in rows.items()}


def _training_shards(
    release: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    raw = release.get("training_shards")
    if not isinstance(raw, list) or not raw:
        raise ETTRV3StreamingError("release training shard inventory differs")
    result: dict[str, dict[str, object]] = {}
    for value in raw:
        if (
            not isinstance(value, dict)
            or set(value)
            != {"bytes", "path", "report_sha256", "rows", "sha256", "split"}
            or value.get("split") not in _SPLITS
        ):
            raise ETTRV3StreamingError(
                "release training shard descriptor differs"
            )
        path = _relative(value["path"], "training shard path")
        if path in result:
            raise ETTRV3StreamingError("training shard path repeats")
        _hex(value["sha256"], "training shard SHA-256")
        _hex(value["report_sha256"], "training report SHA-256")
        _integer(value["bytes"], "training shard bytes", 1)
        _integer(value["rows"], "training shard rows", 1)
        result[path] = value
    return result


def move_continuation_batch(
    value: ETTRContinuationBatch,
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> ETTRContinuationBatch:
    """Move every tensor in a continuation batch without changing receipts."""

    def move(item: object) -> object:
        if torch.is_tensor(item):
            return item.to(device=device, non_blocking=non_blocking)
        if is_dataclass(item):
            return type(item)(
                **{
                    field.name: move(getattr(item, field.name))
                    for field in fields(item)
                }
            )
        if isinstance(item, tuple):
            return tuple(move(child) for child in item)
        if item is None or isinstance(item, (str, int, bool)):
            return item
        raise ETTRV3StreamingError(
            f"unsupported continuation value: {type(item).__name__}"
        )

    moved = move(value)
    if not isinstance(moved, ETTRContinuationBatch):
        raise ETTRV3StreamingError("moved continuation batch type differs")
    return moved


class ETTRV3StreamingRelease:
    """Immutable release reader with deterministic equal-rank assignment."""

    def __init__(
        self,
        root: Path,
        *,
        expected_release_sha256: str,
        data_root: Path,
        tokenizer_path: Path,
    ):
        self.root = root.resolve()
        self.data_root = data_root.resolve()
        self.release = _load_release(
            self.root / "release.json",
            expected_release_sha256,
        )
        self.manifest = _load_continuation(self.root, self.release)
        self.records = _load_stream_index(self.root, self.release)
        self.shards = _training_shards(self.release)
        if (
            self.release.get("training_batches_per_core")
            != TRAINING_BATCHES_PER_CORE
            or self.release.get("training_rows_per_batch")
            != TRAINING_ROWS_PER_BATCH
        ):
            raise ETTRV3StreamingError(
                "ETTR v3 training microbatch geometry differs"
            )
        packet_descriptor = self.release.get("packet_index_manifest")
        if not isinstance(packet_descriptor, Mapping):
            raise ETTRV3StreamingError(
                "ETTR v3 packet-index descriptor differs"
            )
        packet_path = self.root / _relative(
            packet_descriptor.get("path"),
            "packet-index manifest path",
        )
        packet_payload, packet_sha256 = _stable_payload(
            packet_path,
            "packet-index manifest",
            require_immutable=True,
        )
        if (
            packet_sha256 != packet_descriptor.get("sha256")
            or len(packet_payload) != packet_descriptor.get("bytes")
        ):
            raise ETTRV3StreamingError(
                "ETTR v3 packet-index manifest identity differs"
            )
        try:
            packet_manifest = json.loads(packet_payload.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ETTRV3StreamingError(
                "ETTR v3 packet-index manifest is malformed"
            ) from exc
        if (
            not isinstance(packet_manifest, Mapping)
            or packet_manifest.get("manifest_payload_sha256")
            != packet_descriptor.get("payload_sha256")
        ):
            raise ETTRV3StreamingError(
                "ETTR v3 packet-index payload identity differs"
            )
        self.packet_index_root = packet_path.parent
        tokenizer_payload, tokenizer_sha256 = _stable_payload(
            tokenizer_path,
            "ETTR v3 tokenizer",
            require_immutable=False,
        )
        tokenizer_receipt = self.release.get("tokenizer")
        if (
            not isinstance(tokenizer_receipt, Mapping)
            or tokenizer_sha256 != tokenizer_receipt.get("sha256")
            or len(tokenizer_payload) != tokenizer_receipt.get("bytes")
            or tokenizer_sha256 != self.manifest.tokenizer_sha256
        ):
            raise ETTRV3StreamingError("ETTR v3 tokenizer identity differs")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        codec = TokenNativeSurfaceCodec(tokenizer_path)
        if codec.tokenizer_sha256 != tokenizer_sha256:
            raise ETTRV3StreamingError(
                "ETTR v3 tokenizer codec identity differs"
            )

    def verify_source_shards(self) -> dict[str, object]:
        """Hash every training shard before any GPU writer starts."""

        verified_rows = 0
        for path_value, descriptor in sorted(self.shards.items()):
            path = self.data_root / path_value
            before = _identity(
                path,
                "ETTR v3 training shard",
                require_immutable=True,
            )
            digest, size = _sha256_file(path)
            after = _identity(
                path,
                "ETTR v3 training shard",
                require_immutable=True,
            )
            if (
                before != after
                or digest != descriptor["sha256"]
                or size != descriptor["bytes"]
            ):
                raise ETTRV3StreamingError(
                    f"ETTR v3 training shard identity differs: {path_value}"
                )
            verified_rows += int(descriptor["rows"])
        split_counts = self.release.get("training_split_core_counts")
        if (
            not isinstance(split_counts, Mapping)
            or verified_rows
            != sum(int(split_counts[split]) for split in _SPLITS)
        ):
            raise ETTRV3StreamingError(
                "ETTR v3 verified shard rows do not reconcile"
            )
        return {
            "schema": "r12-ettr-il-v3-training-source-verification-v1",
            "shards": len(self.shards),
            "core_rows": verified_rows,
            "status": "pass",
        }

    def _ordered_shards(self, split: str, epoch: int, seed: int):
        values = [
            (path, descriptor)
            for path, descriptor in self.shards.items()
            if descriptor["split"] == split
        ]
        return sorted(
            values,
            key=lambda item: hashlib.sha256(
                (
                    f"{seed}\x1f{epoch}\x1f{item[0]}"
                ).encode("ascii")
            ).digest(),
        )

    def iter_batches(
        self,
        split: str,
        *,
        rank: int,
        world_size: int,
        epoch: int,
        seed: int,
        start_position: int = 0,
        device: torch.device | str | None = None,
    ) -> Iterator[ETTRContinuationBatch]:
        """Yield one equal-cardinality deterministic shard-interleaved epoch."""

        for _, batch in self.iter_positioned_batches(
            split,
            rank=rank,
            world_size=world_size,
            epoch=epoch,
            seed=seed,
            start_position=start_position,
            device=device,
        ):
            yield batch

    def iter_positioned_batches(
        self,
        split: str,
        *,
        rank: int,
        world_size: int,
        epoch: int,
        seed: int,
        start_position: int = 0,
        device: torch.device | str | None = None,
    ) -> Iterator[tuple[int, ETTRContinuationBatch]]:
        """Yield global positions plus equal-cardinality rank-local batches."""

        if (
            split not in _SPLITS
            or not isinstance(world_size, int)
            or world_size < 1
            or not isinstance(rank, int)
            or not 0 <= rank < world_size
            or not isinstance(epoch, int)
            or epoch < 0
            or not isinstance(seed, int)
            or not isinstance(start_position, int)
            or start_position < 0
            or start_position % world_size
        ):
            raise ETTRV3StreamingError("ETTR v3 stream cursor differs")
        expected = {
            (
                str(value["shard_path"]),
                int(value["row_index"]),
                int(value["batch_index"]),
            ): value
            for value in self.records[split]
        }
        total = len(expected)
        usable = total - total % world_size
        if start_position > usable:
            raise ETTRV3StreamingError(
                "ETTR v3 stream cursor exceeds the usable epoch"
            )
        iterators = []
        identities = []
        for path_value, descriptor in self._ordered_shards(
            split,
            epoch,
            seed,
        ):
            path = self.data_root / path_value
            before = _identity(
                path,
                "ETTR v3 training shard",
                require_immutable=True,
            )
            digest, size = _sha256_file(path)
            if (
                digest != descriptor["sha256"]
                or size != descriptor["bytes"]
            ):
                raise ETTRV3StreamingError(
                    f"ETTR v3 shard identity differs: {path_value}"
                )
            iterators.append(
                [path_value, descriptor, enumerate(_iter_records(path)), 0]
            )
            identities.append((path, before))

        position = 0
        while iterators:
            next_round = []
            for path_value, descriptor, iterator, observed in iterators:
                try:
                    row_index, (payload, record) = next(iterator)
                except StopIteration:
                    if observed != descriptor["rows"]:
                        raise ETTRV3StreamingError(
                            f"ETTR v3 shard row count differs: {path_value}"
                        )
                    continue
                next_round.append(
                    [path_value, descriptor, iterator, observed + 1]
                )
                if (
                    record.canonical_bytes() != payload
                    or record.identity.split != split
                ):
                    raise ETTRV3StreamingError(
                        "ETTR v3 streamed semantic core differs"
                    )
                core_batch = None
                for batch_index in range(TRAINING_BATCHES_PER_CORE):
                    key = (
                        str(path_value),
                        int(row_index),
                        batch_index,
                    )
                    stream = expected.pop(key, None)
                    if (
                        stream is None
                        or record.identity.core_id != stream["core_id"]
                        or record.core_sha256() != stream["core_sha256"]
                    ):
                        raise ETTRV3StreamingError(
                            "ETTR v3 stream core identity differs"
                        )
                    if (
                        start_position <= position < usable
                        and position % world_size == rank
                    ):
                        if core_batch is None:
                            core_batch = rematerialize_record(
                                record,
                                self.tokenizer,
                            )
                        start = batch_index * TRAINING_ROWS_PER_BATCH
                        batch = select_continuation_rows(
                            core_batch,
                            torch.arange(
                                start,
                                start + TRAINING_ROWS_PER_BATCH,
                                dtype=torch.long,
                            ),
                        )
                        if (
                            continuation_batch_payload_sha256(batch)
                            != stream["batch_payload_sha256"]
                        ):
                            raise ETTRV3StreamingError(
                                "ETTR v3 streamed tensor batch differs"
                            )
                        batch = replace(
                            batch,
                            manifest_sha256=self.manifest.sha256(),
                            dataset_sha256=self.manifest.dataset_sha256,
                        )
                        if device is not None:
                            batch = move_continuation_batch(batch, device)
                        yield position, batch
                    position += 1
            iterators = next_round
        if expected or position != total:
            raise ETTRV3StreamingError(
                "ETTR v3 stream index and shard population differ"
            )
        for path, before in identities:
            if (
                _identity(
                    path,
                    "ETTR v3 training shard",
                    require_immutable=True,
                )
                != before
            ):
                raise ETTRV3StreamingError(
                    "ETTR v3 training shard changed during iteration"
                )


__all__ = [
    "ETTRV3StreamingError",
    "ETTRV3StreamingRelease",
    "move_continuation_batch",
]
