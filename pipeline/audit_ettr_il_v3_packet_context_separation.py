#!/usr/bin/env python3
"""Audit ETTR v3 splits by the exact deployed packet/query statistic."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import Future, ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
from typing import Iterator, Mapping, Sequence


_ROOT = Path(__file__).resolve().parents[1]
_TRAIN = _ROOT / "train"
for _path in (_ROOT / "pipeline", _TRAIN):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ettr_il_v2_token_native_surface import (  # noqa: E402
    TokenNativeSurfaceCodec,
)
import ettr_il_v3_packet_contexts as packet_contexts_module  # noqa: E402
from ettr_il_v3_packet_contexts import (  # noqa: E402
    compact_packet_context_rows,
)
from ettr_il_v3_protocol import PROTOCOL, ROWS_PER_CORE  # noqa: E402
from materialize_ettr_il_v3_corpus import (  # noqa: E402
    AUDIT_SCHEMA,
    _canonical_file,
    _iter_records,
    _relative,
    _sha256_file,
    _verify_self_hash,
)


REPORT_SCHEMA = "r12-ettr-il-v3-packet-context-separation-v1"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_SPLITS = (
    "train",
    "development",
    "train_reserve",
    "development_reserve",
)
_OWNER_MASK = {
    "train": 1,
    "train_reserve": 1,
    "development": 2,
    "development_reserve": 2,
}
_WORKER_TOKENIZER: object | None = None


class ETTRV3PacketSeparationError(ValueError):
    """The exact packet-context separation audit cannot be completed."""


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _regular_identity(
    path: Path,
    label: str,
    *,
    immutable: bool,
) -> tuple[int, ...]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ETTRV3PacketSeparationError(
            f"{label} cannot be inspected"
        ) from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or (immutable and metadata.st_mode & 0o222)
    ):
        raise ETTRV3PacketSeparationError(
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


def _stable_sha256(
    path: Path,
    label: str,
    *,
    immutable: bool,
) -> tuple[str, int]:
    before = _regular_identity(path, label, immutable=immutable)
    digest, size = _sha256_file(path)
    after = _regular_identity(path, label, immutable=immutable)
    if before != after or size != before[2]:
        raise ETTRV3PacketSeparationError(
            f"{label} changed while being measured"
        )
    return digest, size


def _write_no_replace(path: Path, payload: bytes) -> None:
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


def _load_audit(path: Path) -> tuple[dict[str, object], str]:
    value = _canonical_file(path, "main materialization audit")
    if (
        value.get("schema") != AUDIT_SCHEMA
        or value.get("protocol") != PROTOCOL
        or value.get("status") != "pass"
        or value.get("role") != "main"
    ):
        raise ETTRV3PacketSeparationError(
            "main materialization audit differs"
        )
    return value, _verify_self_hash(
        value,
        field="audit_sha256",
        label="main materialization audit",
    )


def _descriptors(
    audit: Mapping[str, object],
    data_root: Path,
) -> dict[str, tuple[dict[str, object], ...]]:
    raw_shards = audit.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ETTRV3PacketSeparationError("main shard inventory differs")
    result: dict[str, list[dict[str, object]]] = {
        split: [] for split in _SPLITS
    }
    observed: set[str] = set()
    for raw in raw_shards:
        if (
            not isinstance(raw, dict)
            or set(raw)
            != {"bytes", "path", "report_sha256", "rows", "sha256", "split"}
        ):
            raise ETTRV3PacketSeparationError(
                "main shard descriptor differs"
            )
        split = raw["split"]
        relative = _relative(raw["path"], "main shard path")
        if split not in result or relative in observed:
            raise ETTRV3PacketSeparationError(
                "main shard split or path differs"
            )
        observed.add(relative)
        digest, size = _stable_sha256(
            data_root / relative,
            "main shard",
            immutable=True,
        )
        if digest != raw["sha256"] or size != raw["bytes"]:
            raise ETTRV3PacketSeparationError(
                "main shard identity differs"
            )
        result[str(split)].append(dict(raw))
    return {
        split: tuple(sorted(values, key=lambda item: str(item["path"])))
        for split, values in result.items()
    }


def _initialize_worker(tokenizer_path: str) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = TokenNativeSurfaceCodec(Path(tokenizer_path)).tokenizer


def _project_record(
    item: tuple[int, tuple[bytes, object]],
    split: str,
    shard_path: str,
) -> dict[str, object]:
    row_index, (payload, record) = item
    if (
        _WORKER_TOKENIZER is None
        or not hasattr(record, "identity")
        or record.identity.split != split
        or record.canonical_bytes() != payload
    ):
        raise ETTRV3PacketSeparationError(
            "packet audit record split or canonical form differs"
        )
    theory = record.assessor_only.semantic_factors.theory
    family = theory.get("family") if isinstance(theory, Mapping) else None
    if not isinstance(family, str) or not family:
        raise ETTRV3PacketSeparationError(
            "packet audit record family differs"
        )
    rows = compact_packet_context_rows(record, _WORKER_TOKENIZER)
    if len(rows) != ROWS_PER_CORE:
        raise ETTRV3PacketSeparationError(
            "packet audit row geometry differs"
        )
    return {
        "core_id": record.identity.core_id,
        "core_sha256": record.core_sha256(),
        "family": family,
        "rows": rows,
        "row_index": row_index,
        "shard_path": shard_path,
        "split": split,
        "stage": record.identity.curriculum_stage,
    }


def _bounded_projections(
    records: Iterator[tuple[int, tuple[bytes, object]]],
    *,
    split: str,
    shard_path: str,
    executor: ProcessPoolExecutor,
    workers: int,
) -> Iterator[dict[str, object]]:
    pending: deque[Future[dict[str, object]]] = deque()

    def submit_next() -> bool:
        try:
            item = next(records)
        except StopIteration:
            return False
        pending.append(
            executor.submit(_project_record, item, split, shard_path)
        )
        return True

    for _ in range(2 * workers):
        if not submit_next():
            break
    while pending:
        yield pending.popleft().result()
        submit_next()


class _UnionFind:
    def __init__(self) -> None:
        self.parent: list[int] = []
        self.size: list[int] = []

    def add(self) -> int:
        index = len(self.parent)
        self.parent.append(index)
        self.size.append(1)
        return index

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left = self.find(left)
        right = self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def audit_packet_context_separation(
    *,
    main_audit_path: Path,
    data_root: Path,
    tokenizer_path: Path,
    source_commit: str,
    output: Path,
    workers: int,
    progress_every: int = 1_000,
) -> dict[str, object]:
    """Audit and componentize all model-visible packet/query contexts."""

    if _HEX40.fullmatch(source_commit) is None:
        raise ETTRV3PacketSeparationError("source commit differs")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ETTRV3PacketSeparationError("worker count differs")
    if (
        isinstance(progress_every, bool)
        or not isinstance(progress_every, int)
        or progress_every < 1
    ):
        raise ETTRV3PacketSeparationError("progress interval differs")
    if output.exists():
        raise ETTRV3PacketSeparationError("output already exists")

    audit, audit_sha256 = _load_audit(main_audit_path)
    auditor_sha256, auditor_bytes = _stable_sha256(
        Path(__file__).resolve(),
        "packet-context auditor",
        immutable=False,
    )
    extractor_sha256, extractor_bytes = _stable_sha256(
        Path(packet_contexts_module.__file__).resolve(),
        "packet-context extractor",
        immutable=False,
    )
    codec = TokenNativeSurfaceCodec(tokenizer_path)
    if (
        audit.get("tokenizer_sha256") != codec.tokenizer_sha256
        or audit.get("codebook_sha256") != codec.codebook_sha256
    ):
        raise ETTRV3PacketSeparationError(
            "audit tokenizer or codebook differs"
        )
    split_descriptors = _descriptors(audit, data_root)
    expected_counts = audit.get("split_counts")
    if not isinstance(expected_counts, dict) or set(expected_counts) != set(
        _SPLITS
    ):
        raise ETTRV3PacketSeparationError("audit split counts differ")

    started = time.monotonic()
    union = _UnionFind()
    cores: list[dict[str, object]] = []
    core_ids: set[str] = set()
    contexts: dict[bytes, list[int]] = {}
    conflict_contexts: set[bytes] = set()
    split_counts: Counter[str] = Counter()
    split_rows: Counter[str] = Counter()
    duplicate_rows = 0
    stream_digest = hashlib.sha256()

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize_worker,
        initargs=(str(tokenizer_path.resolve()),),
    ) as executor:
        for split in _SPLITS:
            for descriptor in split_descriptors[split]:
                shard_path = _relative(
                    descriptor["path"],
                    "main shard path",
                )
                records = enumerate(_iter_records(data_root / shard_path))
                observed_rows = 0
                for result in _bounded_projections(
                    records,
                    split=split,
                    shard_path=shard_path,
                    executor=executor,
                    workers=workers,
                ):
                    core_id = result["core_id"]
                    if not isinstance(core_id, str) or core_id in core_ids:
                        raise ETTRV3PacketSeparationError(
                            "packet audit core ID repeats"
                        )
                    core_ids.add(core_id)
                    core_index = union.add()
                    core = {
                        key: result[key]
                        for key in (
                            "core_id",
                            "core_sha256",
                            "family",
                            "row_index",
                            "shard_path",
                            "split",
                            "stage",
                        )
                    }
                    cores.append(core)
                    rows = result["rows"]
                    if not isinstance(rows, tuple) or len(rows) != ROWS_PER_CORE:
                        raise ETTRV3PacketSeparationError(
                            "packet audit compact rows differ"
                        )
                    stream_digest.update(
                        _canonical_bytes(
                            {
                                "core_id": core_id,
                                "core_sha256": result["core_sha256"],
                                "rows": [
                                    [digest.hex(), target]
                                    for digest, target in rows
                                ],
                                "split": split,
                            }
                        )
                    )
                    owner_mask = _OWNER_MASK[split]
                    for digest, target in rows:
                        prior = contexts.get(digest)
                        if prior is None:
                            contexts[digest] = [
                                target,
                                core_index,
                                owner_mask,
                                1,
                            ]
                            continue
                        duplicate_rows += 1
                        if prior[0] != target:
                            conflict_contexts.add(digest)
                        union.union(core_index, prior[1])
                        prior[2] |= owner_mask
                        prior[3] += 1
                    split_counts[split] += 1
                    split_rows[split] += len(rows)
                    observed_rows += 1
                    completed = len(cores)
                    if completed % progress_every == 0:
                        elapsed = time.monotonic() - started
                        print(
                            json.dumps(
                                {
                                    "cores": completed,
                                    "elapsed_seconds": round(elapsed, 3),
                                    "rows": sum(split_rows.values()),
                                    "unique_contexts": len(contexts),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                if observed_rows != descriptor["rows"]:
                    raise ETTRV3PacketSeparationError(
                        "packet audit shard row count differs"
                    )

    for split in _SPLITS:
        if split_counts[split] != expected_counts[split]:
            raise ETTRV3PacketSeparationError(
                f"packet audit {split} count differs"
            )

    members: dict[int, list[int]] = {}
    component_owner_mask: dict[int, int] = {}
    for index, core in enumerate(cores):
        root = union.find(index)
        members.setdefault(root, []).append(index)
        component_owner_mask[root] = (
            component_owner_mask.get(root, 0)
            | _OWNER_MASK[str(core["split"])]
        )
    cross_roots = sorted(
        (
            root
            for root, mask in component_owner_mask.items()
            if mask == 3
        ),
        key=lambda root: str(cores[members[root][0]]["core_id"]),
    )
    cross_components: list[dict[str, object]] = []
    pair_breakdown: Counter[str] = Counter()
    for root in cross_roots:
        component_cores = sorted(
            (cores[index] for index in members[root]),
            key=lambda value: (str(value["split"]), str(value["core_id"])),
        )
        labels = Counter(
            f"{core['split']}|{core['family']}|{core['stage']}"
            for core in component_cores
        )
        for label, count in labels.items():
            pair_breakdown[label] += count
        component_value = {
            "cores": component_cores,
            "owner_splits": sorted(
                {
                    str(core["split"]).removesuffix("_reserve")
                    for core in component_cores
                }
            ),
        }
        component_value["component_sha256"] = hashlib.sha256(
            _canonical_bytes(component_value)
        ).hexdigest()
        cross_components.append(component_value)

    cross_context_digests = sorted(
        digest for digest, value in contexts.items() if value[2] == 3
    )
    cross_context_set_sha256 = hashlib.sha256(
        b"".join(cross_context_digests)
    ).hexdigest()
    report: dict[str, object] = {
        "audit_sha256": audit_sha256,
        "auditor": {
            "bytes": auditor_bytes,
            "path": "pipeline/audit_ettr_il_v3_packet_context_separation.py",
            "sha256": auditor_sha256,
        },
        "context_stream_sha256": stream_digest.hexdigest(),
        "cross_owner_component_count": len(cross_components),
        "cross_owner_components": cross_components,
        "cross_owner_context_count": len(cross_context_digests),
        "cross_owner_context_set_sha256": cross_context_set_sha256,
        "cross_owner_label_counts": dict(sorted(pair_breakdown.items())),
        "duplicate_context_rows": duplicate_rows,
        "extractor": {
            "bytes": extractor_bytes,
            "path": "pipeline/ettr_il_v3_packet_contexts.py",
            "sha256": extractor_sha256,
        },
        "packet_context_rows": sum(split_rows.values()),
        "protocol": PROTOCOL,
        "schema": REPORT_SCHEMA,
        "source_commit": source_commit,
        "split_core_counts": dict(split_counts),
        "split_row_counts": dict(split_rows),
        "status": (
            "pass"
            if not cross_components and not conflict_contexts
            else "fail"
        ),
        "target_conflict_context_count": len(conflict_contexts),
        "target_conflict_context_set_sha256": hashlib.sha256(
            b"".join(sorted(conflict_contexts))
        ).hexdigest(),
        "tokenizer_sha256": codec.tokenizer_sha256,
        "unique_contexts": len(contexts),
    }
    report["report_sha256"] = hashlib.sha256(
        _canonical_bytes(report)
    ).hexdigest()
    payload = _canonical_bytes(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_no_replace(output, payload)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-audit", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=1_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = audit_packet_context_separation(
        main_audit_path=arguments.main_audit,
        data_root=arguments.data_root,
        tokenizer_path=arguments.tokenizer,
        source_commit=arguments.source_commit,
        output=arguments.output,
        workers=arguments.workers,
        progress_every=arguments.progress_every,
    )
    print(
        json.dumps(
            {
                "cross_owner_components": report[
                    "cross_owner_component_count"
                ],
                "cross_owner_contexts": report[
                    "cross_owner_context_count"
                ],
                "report_sha256": report["report_sha256"],
                "status": report["status"],
                "target_conflicts": report[
                    "target_conflict_context_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "pass" else 4


if __name__ == "__main__":
    raise SystemExit(main())
