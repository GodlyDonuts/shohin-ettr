#!/usr/bin/env python3
"""Compare candidate Phase-2 tokenizers on private retained-document packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Iterable

from tokenizers import Tokenizer


SCHEMA = "shohin-phase2-tokenizer-comparison-v1"


class TokenizerComparisonError(ValueError):
    """The tokenizer comparison cannot produce a trustworthy receipt."""


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    path = Path(raw_path)
    if (
        not separator
        or not name
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in name)
        or not path.is_absolute()
    ):
        raise argparse.ArgumentTypeError("expected lowercase_name=/absolute/path")
    return name, path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        raise TokenizerComparisonError("token count distribution is empty")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _source_rows(path: Path, maximum: int) -> list[str]:
    texts: list[str] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if len(texts) >= maximum:
                    break
                value = json.loads(line)
                text = value.get("review_text") if isinstance(value, dict) else None
                if isinstance(text, str) and text:
                    texts.append(text)
    except (OSError, json.JSONDecodeError) as error:
        raise TokenizerComparisonError(f"source packet is unreadable: {path}") from error
    if not texts:
        raise TokenizerComparisonError(f"source packet has no review text: {path}")
    return texts


def compare(
    *,
    tokenizers: list[tuple[str, Path]],
    sources: list[tuple[str, Path]],
    maximum_documents: int,
) -> dict:
    if (
        len(tokenizers) < 2
        or len({name for name, _path in tokenizers}) != len(tokenizers)
        or len({name for name, _path in sources}) != len(sources)
        or not sources
        or maximum_documents < 1
    ):
        raise TokenizerComparisonError("comparison arguments differ")
    source_texts = {
        name: _source_rows(path, maximum_documents) for name, path in sources
    }
    source_receipts = {
        name: {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "sampled_documents": len(source_texts[name]),
        }
        for name, path in sources
    }
    tokenizer_reports = {}
    for tokenizer_name, tokenizer_path in tokenizers:
        if not tokenizer_path.is_file() or tokenizer_path.is_symlink():
            raise TokenizerComparisonError(
                f"tokenizer is not a physical file: {tokenizer_path}"
            )
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        domains = {}
        aggregate_counts: list[int] = []
        aggregate_bytes = 0
        roundtrip_mismatches = 0
        for source_name, texts in source_texts.items():
            counts: list[int] = []
            byte_count = 0
            source_mismatches = 0
            for text in texts:
                encoding = tokenizer.encode(text, add_special_tokens=False)
                counts.append(len(encoding.ids))
                byte_count += len(text.encode("utf-8"))
                decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
                if tokenizer.encode(decoded, add_special_tokens=False).ids != encoding.ids:
                    source_mismatches += 1
            aggregate_counts.extend(counts)
            aggregate_bytes += byte_count
            roundtrip_mismatches += source_mismatches
            total_tokens = sum(counts)
            domains[source_name] = {
                "documents": len(counts),
                "utf8_bytes": byte_count,
                "tokens": total_tokens,
                "bytes_per_token": byte_count / total_tokens,
                "mean_tokens_per_document": statistics.fmean(counts),
                "p50_tokens": _percentile(counts, 0.50),
                "p90_tokens": _percentile(counts, 0.90),
                "p99_tokens": _percentile(counts, 0.99),
                "encode_decode_reencode_mismatches": source_mismatches,
            }
        total_tokens = sum(aggregate_counts)
        tokenizer_reports[tokenizer_name] = {
            "path": str(tokenizer_path.resolve()),
            "sha256": _sha256(tokenizer_path),
            "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
            "domains": domains,
            "aggregate": {
                "documents": len(aggregate_counts),
                "utf8_bytes": aggregate_bytes,
                "tokens": total_tokens,
                "bytes_per_token": aggregate_bytes / total_tokens,
                "mean_tokens_per_document": statistics.fmean(aggregate_counts),
                "p50_tokens": _percentile(aggregate_counts, 0.50),
                "p90_tokens": _percentile(aggregate_counts, 0.90),
                "p99_tokens": _percentile(aggregate_counts, 0.99),
                "encode_decode_reencode_mismatches": roundtrip_mismatches,
            },
        }
    report = {
        "schema": SCHEMA,
        "maximum_documents_per_source": maximum_documents,
        "sources": source_receipts,
        "tokenizers": tokenizer_reports,
        "contains_document_text": False,
    }
    payload = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("ascii")
    report["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    return report


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", action="append", type=_named_path, required=True)
    parser.add_argument("--source", action="append", type=_named_path, required=True)
    parser.add_argument("--maximum-documents", type=int, default=2_000)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(f"refusing existing output: {arguments.output}")
    report = compare(
        tokenizers=arguments.tokenizer,
        sources=arguments.source,
        maximum_documents=arguments.maximum_documents,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = arguments.output.with_name(
        f".{arguments.output.name}.tmp.{os.getpid()}"
    )
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        os.replace(temporary, arguments.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
