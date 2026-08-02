#!/usr/bin/env python3
"""Audit the deployed ETTR document mask against the exact public cover.

The deployed GPU router accepts the first grammar-complete root. Reverse
postfix documents can contain a nested root-shaped node before the true root,
while deterministic cover can accidentally form later root-shaped nodes. This
CPU-only audit compares that historical heuristic with the unique boundary
whose public suffix matches the deterministic cover hash. It reads WORLD and
COMMAND source transports only; QUERY, targets, traces, and answers are never
read.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

from audit_ettr_public_opcode_identifiability import (
    _CALL_END,
    _CALL_STRIDE,
    _FRAME_A,
    _FRAME_B,
    _REIFY_BASE,
    _REIFY_END,
    _ROOT_CODES,
    public_document_indices,
)
from ettr_il_v2_token_native_surface import TokenNativeSurfaceCodec
from ettr_il_v3_protocol import canonical_json_bytes
from materialize_ettr_il_v3_corpus import _iter_records, _sha256_file


REPORT_SCHEMA = "r12-ettr-public-document-mask-integrity-audit-v1"
_SPLITS = ("train", "development")
_STAGES = ("world", "command")


class DocumentMaskIntegrityError(ValueError):
    """One source transport or audit contract differs."""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def legacy_public_document_end(
    physical: Sequence[int],
    *,
    codebook_size: int,
) -> int:
    """Return the historical first-completion boundary, exclusive."""

    codes = tuple(int(value) for value in physical)
    if (
        len(codes) < 3
        or codes[0] not in {_FRAME_A, _FRAME_B}
        or codes[1] not in {_FRAME_A, _FRAME_B}
        or not isinstance(codebook_size, int)
        or codebook_size <= _REIFY_END
        or min(codes) < 0
        or max(codes) >= codebook_size
    ):
        raise DocumentMaskIntegrityError("public transport geometry differs")
    prefix = codes[0] == _FRAME_A
    state = 1 if prefix else 0
    for body_index, code in enumerate(codes[2:]):
        if 0 <= code < _CALL_END:
            arity = code % _CALL_STRIDE
        elif _REIFY_BASE <= code < _REIFY_END:
            arity = code - _REIFY_BASE + 1
        else:
            arity = 0
        state += arity - 1 if prefix else 1 - arity
        if (prefix and state == 0) or (
            not prefix and state == 1 and code in _ROOT_CODES
        ):
            return body_index + 3
    raise DocumentMaskIntegrityError("legacy public document mask does not terminate")


def _empty_counts() -> dict[str, object]:
    return {
        "cover_included": 0,
        "delta_histogram": Counter(),
        "exact": 0,
        "total": 0,
        "truncated": 0,
    }


def _observe(counts: dict[str, object], exact: int, legacy: int) -> None:
    counts["total"] = int(counts["total"]) + 1
    delta = exact - legacy
    histogram = counts["delta_histogram"]
    if not isinstance(histogram, Counter):
        raise DocumentMaskIntegrityError("mask delta histogram differs")
    histogram[str(delta)] += 1
    if delta == 0:
        counts["exact"] = int(counts["exact"]) + 1
    elif delta > 0:
        counts["truncated"] = int(counts["truncated"]) + 1
    else:
        counts["cover_included"] = int(counts["cover_included"]) + 1


def _merge(destination: dict[str, object], source: Mapping[str, object]) -> None:
    for key in ("cover_included", "exact", "total", "truncated"):
        destination[key] = int(destination[key]) + int(source[key])
    destination_histogram = destination["delta_histogram"]
    source_histogram = source["delta_histogram"]
    if not isinstance(destination_histogram, Counter) or not isinstance(
        source_histogram, Counter
    ):
        raise DocumentMaskIntegrityError("mask delta histogram differs")
    destination_histogram.update(source_histogram)


def _audit_shard(
    arguments: tuple[Path, Path, str, Path],
) -> tuple[dict[str, object], dict[str, object]]:
    path, data_root, split, tokenizer = arguments
    codec = TokenNativeSurfaceCodec(tokenizer)
    totals = _empty_counts()
    by_renderer = {
        str(renderer): {stage: _empty_counts() for stage in _STAGES}
        for renderer in range(4)
    }
    rows = 0
    core_ids: set[str] = set()
    digest, size = _sha256_file(path)
    for payload, record in _iter_records(path):
        if record.canonical_bytes() != payload or record.identity.split != split:
            raise DocumentMaskIntegrityError("semantic-core record differs")
        if record.identity.core_id in core_ids:
            raise DocumentMaskIntegrityError("semantic-core identity repeats")
        core_ids.add(record.identity.core_id)
        rows += 1
        views = tuple(record.source_visible.views)
        if len(views) != 4 or {int(view.renderer) for view in views} != set(range(4)):
            raise DocumentMaskIntegrityError("renderer orbit differs")
        for view in views:
            renderer = str(int(view.renderer))
            for stage, sources in (
                ("world", tuple(view.world_sources)),
                ("command", tuple(view.command_sources)),
            ):
                if len(sources) != 4:
                    raise DocumentMaskIntegrityError(
                        "corner-conditioned source geometry differs"
                    )
                for source in sources:
                    physical = codec._payload_indices(source.encode("ascii"))
                    exact = len(public_document_indices(codec, source))
                    legacy = legacy_public_document_end(
                        physical,
                        codebook_size=len(codec.codebook.token_ids),
                    )
                    _observe(totals, exact, legacy)
                    _observe(by_renderer[renderer][stage], exact, legacy)
    return (
        {
            "by_renderer": by_renderer,
            "totals": totals,
        },
        {
            "bytes": size,
            "path": path.relative_to(data_root).as_posix(),
            "rows": rows,
            "sha256": digest,
        },
    )


def _shards(data_root: Path, split: str) -> tuple[Path, ...]:
    root = data_root / split
    paths = tuple(sorted(root.glob("*.jsonl.gz"))) if root.is_dir() else ()
    if not paths or any(path.is_symlink() or not path.is_file() for path in paths):
        raise DocumentMaskIntegrityError(f"split shard set differs: {split}")
    return paths


def _finalize(counts: Mapping[str, object]) -> dict[str, object]:
    total = int(counts["total"])
    exact = int(counts["exact"])
    truncated = int(counts["truncated"])
    cover_included = int(counts["cover_included"])
    histogram = counts["delta_histogram"]
    if (
        total < 1
        or exact + truncated + cover_included != total
        or not isinstance(histogram, Counter)
        or sum(histogram.values()) != total
    ):
        raise DocumentMaskIntegrityError("mask counts do not reconcile")
    return {
        "cover_included": cover_included,
        "cover_included_rate": cover_included / total,
        "delta_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        "exact": exact,
        "exact_rate": exact / total,
        "total": total,
        "truncated": truncated,
        "truncated_rate": truncated / total,
    }


def _audit_split(
    data_root: Path,
    split: str,
    tokenizer: Path,
    workers: int,
) -> dict[str, object]:
    paths = _shards(data_root, split)
    arguments = tuple((path, data_root, split, tokenizer) for path in paths)
    if workers == 1:
        results = tuple(_audit_shard(argument) for argument in arguments)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = tuple(pool.map(_audit_shard, arguments))
    totals = _empty_counts()
    by_renderer = {
        str(renderer): {stage: _empty_counts() for stage in _STAGES}
        for renderer in range(4)
    }
    shards = []
    for counts, descriptor in results:
        _merge(totals, counts["totals"])
        for renderer in by_renderer:
            for stage in _STAGES:
                _merge(
                    by_renderer[renderer][stage],
                    counts["by_renderer"][renderer][stage],
                )
        shards.append(descriptor)
    return {
        "by_renderer": {
            renderer: {
                stage: _finalize(counts)
                for stage, counts in stages.items()
            }
            for renderer, stages in by_renderer.items()
        },
        "shards": shards,
        "totals": _finalize(totals),
    }


def audit(
    data_root: Path,
    tokenizer: Path,
    *,
    workers: int,
) -> dict[str, object]:
    if not isinstance(workers, int) or workers < 1:
        raise DocumentMaskIntegrityError("worker count differs")
    data_root = data_root.resolve()
    tokenizer = tokenizer.resolve()
    if tokenizer.is_symlink() or not tokenizer.is_file():
        raise DocumentMaskIntegrityError("tokenizer artifact differs")
    tokenizer_sha256 = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    splits = {
        split: _audit_split(data_root, split, tokenizer, workers)
        for split in _SPLITS
    }
    payload = {
        "data_root": str(data_root),
        "forbidden_inputs": ["query_sources", "targets", "traces", "answers"],
        "schema": REPORT_SCHEMA,
        "splits": splits,
        "tokenizer_sha256": tokenizer_sha256,
        "workers": workers,
    }
    return {
        **payload,
        "payload_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        "status": "pass",
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise DocumentMaskIntegrityError("output already exists")
    report = audit(
        args.data_root,
        args.tokenizer,
        workers=args.workers,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(report)
    args.output.write_bytes(payload)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
