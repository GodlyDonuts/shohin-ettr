#!/usr/bin/env python3
"""Profile a pinned general-pretraining source without creating training data.

The aggregate report contains no document text. A separate deterministic review
packet may contain bounded excerpts and must remain private; it exists only for
human adjudication and is never a training input.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse


WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
HTML_RE = re.compile(r"</?[a-zA-Z][^>]{0,200}>")
URL_RE = re.compile(r"https?://\S+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
BOILERPLATE_MARKERS = (
    "accept cookies",
    "cookie policy",
    "privacy policy",
    "terms of service",
    "all rights reserved",
    "sign in",
    "log in",
    "subscribe to our newsletter",
    "javascript is disabled",
    "enable javascript",
)
IDENTITY_FIELDS = (
    "id",
    "blob_id",
    "content_id",
    "url",
    "repo_path",
    "repo_name",
    "commit_id",
    "path",
    "file_path",
    "source",
)
REVIEW_METADATA_FIELDS = (
    "id",
    "url",
    "dump",
    "date",
    "file_path",
    "language",
    "language_score",
    "score",
    "int_score",
    "token_count",
    "extractor",
    "is_truncated",
    "full_doc_lid",
    "full_doc_lid_score",
    "page_average_lid",
    "page_average_lid_score",
    "per_page_languages",
    "page_ends",
    "duplicate_count",
    "fw_edu_scores",
    "minhash_cluster_size",
    "source",
    "version",
    "created",
    "added",
    "blob_id",
    "content_id",
    "detected_licenses",
    "license_type",
    "is_vendor",
    "path",
    "repo_path",
    "repo_name",
    "commit_id",
    "github_metadata",
    "num_files",
    "src_encoding",
    "size_bytes",
    "length_bytes",
    "file_timestamp",
    "finish_reason",
    "usage",
)
CATEGORICAL_PROFILE_FIELDS = (
    "finish_reason",
    "is_truncated",
    "extractor",
    "license_type",
    "language",
)
NUMERIC_METRICS = (
    "chars",
    "words",
    "lines",
    "nonempty_lines",
    "unique_line_fraction",
    "max_line_repeat_fraction",
    "alpha_fraction",
    "digit_fraction",
    "control_fraction",
    "replacement_fraction",
    "html_tags",
    "urls",
    "boilerplate_markers",
)


class ProfileError(ValueError):
    """The source cannot produce a trustworthy profile."""


def normalize(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower()))


def grams(text: str, n: int = 13) -> set[str]:
    words = normalize(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index : index + n]) for index in range(len(words) - n + 1)}


def load_eval_index(evals_dir: str | Path) -> dict[str, Any]:
    exact: set[str] = set()
    ngrams: set[str] = set()
    paths = sorted(Path(evals_dir).glob("*.jsonl"))
    prompt_fields = ("question", "problem", "prompt", "task", "text")
    for path in paths:
        with path.open(errors="replace") as source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = next((row[field] for field in prompt_fields if row.get(field)), "")
                clean = normalize(str(prompt))
                if clean:
                    exact.add(clean)
                    ngrams.update(grams(clean))
    return {
        "exact": exact,
        "ngrams": ngrams,
        "files": [str(path) for path in paths],
    }


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    result = {}
    for label, probability in (
        ("min", 0.0),
        ("p10", 0.1),
        ("p50", 0.5),
        ("p90", 0.9),
        ("p99", 0.99),
        ("max", 1.0),
    ):
        index = round((len(ordered) - 1) * probability)
        result[label] = float(ordered[index])
    return result


def text_metrics(text: str) -> dict[str, float | int]:
    chars = len(text)
    words = WORD_RE.findall(text)
    lines = text.splitlines() or ([text] if text else [])
    normalized_lines = [" ".join(line.split()).lower() for line in lines if line.strip()]
    line_counts = Counter(normalized_lines)
    nonempty = len(normalized_lines)
    lower = text.lower()
    denominator = max(chars, 1)
    return {
        "chars": chars,
        "words": len(words),
        "lines": len(lines),
        "nonempty_lines": nonempty,
        "unique_line_fraction": len(line_counts) / max(nonempty, 1),
        "max_line_repeat_fraction": max(line_counts.values(), default=0) / max(nonempty, 1),
        "alpha_fraction": sum(character.isalpha() for character in text) / denominator,
        "digit_fraction": sum(character.isdigit() for character in text) / denominator,
        "control_fraction": len(CONTROL_RE.findall(text)) / denominator,
        "replacement_fraction": text.count("\ufffd") / denominator,
        "html_tags": len(HTML_RE.findall(text)),
        "urls": len(URL_RE.findall(text)),
        "boilerplate_markers": sum(marker in lower for marker in BOILERPLATE_MARKERS),
    }


def _bounded_value(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:2048]
    if depth >= 2:
        return type(value).__name__
    if isinstance(value, (list, tuple)):
        return [_bounded_value(item, depth=depth + 1) for item in value[:32]]
    if isinstance(value, Mapping):
        return {
            str(key)[:128]: _bounded_value(nested, depth=depth + 1)
            for key, nested in list(value.items())[:64]
            if str(key).lower() not in {"text", "content", "body", "code"}
        }
    return str(value)[:2048]


def review_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        field: _bounded_value(row[field])
        for field in REVIEW_METADATA_FIELDS
        if field in row
    }
    if isinstance(row.get("metadata"), Mapping):
        result["metadata"] = _bounded_value(row["metadata"])
    return result


def flatten_nested_files(
    rows: Iterable[Mapping[str, Any]],
    *,
    files_field: str,
    nested_text_field: str,
    max_files_per_record: int,
    parent_review_context_field: str | None = None,
) -> Iterable[Mapping[str, Any]]:
    """Yield one auditable document per nested repository file."""
    if max_files_per_record <= 0:
        raise ProfileError("max_files_per_record must be positive")
    for row in rows:
        files = row.get(files_field)
        if not isinstance(files, (list, tuple)):
            continue
        parent = {
            key: value
            for key, value in row.items()
            if key != files_field and key not in {"text", "content", "body", "code"}
        }
        ranked_files = sorted(
            (file_row for file_row in files if isinstance(file_row, Mapping)),
            key=lambda file_row: hashlib.sha256(
                (
                    str(row.get("repo_path", row.get("repo_name", "")))
                    + "\x1f"
                    + str(file_row.get("content_id", file_row.get("file_path", "")))
                ).encode()
            ).digest(),
        )
        for file_row in ranked_files[:max_files_per_record]:
            if not isinstance(file_row, Mapping):
                continue
            merged = dict(parent)
            merged.update(
                {
                    key: value
                    for key, value in file_row.items()
                    if key != nested_text_field
                }
            )
            if nested_text_field in file_row:
                merged["text"] = file_row[nested_text_field]
            if parent_review_context_field:
                context = row.get(parent_review_context_field)
                if isinstance(context, str):
                    merged["_review_context_text"] = context
            if "repo_path" in row and "repo_name" not in merged:
                merged["repo_name"] = row["repo_path"]
            yield merged


def stable_identity(row: Mapping[str, Any], text_hash: str) -> str:
    values = [str(row[field]) for field in IDENTITY_FIELDS if row.get(field) not in (None, "")]
    if not values:
        values = [text_hash]
    return "\x1f".join(values)


def review_excerpt(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    half = max((limit - 80) // 2, 1)
    return (
        text[:half]
        + "\n\n[... PRIVATE REVIEW EXCERPT TRUNCATED ...]\n\n"
        + text[-half:],
        True,
    )


def _counter_value(value: Any) -> str:
    if value is None:
        return "<missing>"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(_bounded_value(value), sort_keys=True, ensure_ascii=True)


def profile_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset: str,
    config: str,
    text_field: str,
    scan_rows: int,
    review_rows: int,
    max_review_chars: int,
    eval_index: Mapping[str, Any],
    review_context_field: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if scan_rows <= 0 or review_rows <= 0 or max_review_chars <= 0:
        raise ProfileError("row and excerpt limits must be positive")

    metric_values: dict[str, list[float]] = defaultdict(list)
    languages: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    quality_scores: Counter[str] = Counter()
    quality_metric_values: dict[str, list[float]] = defaultdict(list)
    license_values: Counter[str] = Counter()
    categorical_values: dict[str, Counter[str]] = defaultdict(Counter)
    field_presence: Counter[str] = Counter()
    text_hashes: set[str] = set()
    exact_duplicates = 0
    exact_eval_rows = 0
    ngram_eval_rows = 0
    overlap_rows = 0
    overlap_receipts: list[dict[str, Any]] = []
    text_rows = 0
    scanned = 0
    # Negative priority makes heap[0] the largest retained hash.
    review_heap: list[tuple[int, int, dict[str, Any]]] = []

    for row_index, row in enumerate(rows):
        if row_index >= scan_rows:
            break
        scanned += 1
        field_presence.update(row.keys())
        language = row.get("language", row.get("full_doc_lid"))
        languages[_counter_value(language)] += 1
        for field in ("detected_licenses", "license_type", "license"):
            if field in row:
                license_values[_counter_value(row[field])] += 1
        for field in CATEGORICAL_PROFILE_FIELDS:
            if field in row:
                categorical_values[field][_counter_value(row[field])] += 1
        for field in ("int_score", "score", "fw_edu_scores"):
            if field in row:
                quality_scores[f"{field}:{_counter_value(row[field])}"] += 1
                value = row[field]
                values = value if isinstance(value, (list, tuple)) else [value]
                quality_metric_values[field].extend(
                    float(item)
                    for item in values
                    if isinstance(item, (int, float)) and not isinstance(item, bool)
                )

        url = str(row.get("url", ""))
        host = urlparse(url).hostname
        if host:
            domains[host.lower()] += 1

        raw_text = row.get(text_field, "")
        text = raw_text if isinstance(raw_text, str) else ""
        text_hash = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        identity = stable_identity(row, text_hash)
        identity_hash = hashlib.sha256(identity.encode()).hexdigest()
        if text:
            text_rows += 1
            metrics = text_metrics(text)
            for key in NUMERIC_METRICS:
                metric_values[key].append(float(metrics[key]))
            if text_hash in text_hashes:
                exact_duplicates += 1
            else:
                text_hashes.add(text_hash)
            clean = normalize(text)
            exact_hit = bool(clean) and clean in eval_index["exact"]
            exact_eval_rows += int(exact_hit)
            # A bounded head/tail overlap probe avoids materializing millions
            # of n-grams for a pathological PDF while still checking both ends.
            overlap_text = text if len(text) <= 100_000 else text[:50_000] + text[-50_000:]
            matched_ngrams = grams(overlap_text).intersection(eval_index["ngrams"])
            ngram_eval_rows += int(bool(matched_ngrams))
            overlap_rows += int(bool(exact_hit or matched_ngrams))
            if (exact_hit or matched_ngrams) and len(overlap_receipts) < 1000:
                overlap_receipts.append(
                    {
                        "stable_identity_sha256": identity_hash,
                        "document_sha256": text_hash,
                        "exact_normalized_prompt_sha256": (
                            hashlib.sha256(clean.encode()).hexdigest() if exact_hit else None
                        ),
                        "matched_13gram_sha256": [
                            hashlib.sha256(item.encode()).hexdigest()
                            for item in sorted(matched_ngrams)[:100]
                        ],
                    }
                )
        else:
            metrics = {}

        priority = int.from_bytes(
            hashlib.sha256(f"{dataset}\x1f{config}\x1f{identity}".encode()).digest(),
            "big",
        )
        excerpt, truncated = review_excerpt(text, max_review_chars)
        raw_context = row.get(review_context_field, "") if review_context_field else ""
        context = raw_context if isinstance(raw_context, str) else ""
        context_excerpt, context_truncated = review_excerpt(context, max_review_chars)
        review = {
            "schema": "shohin-private-general-source-review-v2",
            "admission_status": "private_human_review_only_not_training_data",
            "dataset": dataset,
            "config": config,
            "stable_identity_sha256": identity_hash,
            "document_sha256": text_hash if text else None,
            "metadata": review_metadata(row),
            "metrics": metrics,
            "review_text": excerpt,
            "review_text_truncated": truncated,
            "review_context_sha256": (
                hashlib.sha256(context.encode("utf-8", errors="replace")).hexdigest()
                if context
                else None
            ),
            "review_context_text": context_excerpt,
            "review_context_text_truncated": context_truncated,
        }
        heap_item = (-priority, row_index, review)
        if len(review_heap) < review_rows:
            heapq.heappush(review_heap, heap_item)
        elif priority < -review_heap[0][0]:
            heapq.heapreplace(review_heap, heap_item)

    if scanned == 0:
        raise ProfileError("source yielded zero rows")

    reviews = [item[2] for item in sorted(review_heap, key=lambda item: (-item[0], item[1]))]
    report = {
        "schema": "shohin-general-source-profile-v1",
        "admission_status": "profile_only_not_training_data",
        "dataset": dataset,
        "config": config,
        "text_field": text_field,
        "scanned_rows": scanned,
        "text_rows": text_rows,
        "metadata_only_rows": scanned - text_rows,
        "exact_duplicate_text_rows": exact_duplicates,
        "unique_text_hashes": len(text_hashes),
        "field_presence": dict(field_presence.most_common()),
        "languages": dict(languages.most_common(50)),
        "domains": dict(domains.most_common(100)),
        "quality_scores": dict(quality_scores.most_common(100)),
        "quality_metric_quantiles": {
            key: quantiles(values) for key, values in quality_metric_values.items()
        },
        "license_values": dict(license_values.most_common(100)),
        "categorical_values": {
            field: dict(values.most_common(100))
            for field, values in sorted(categorical_values.items())
        },
        "metrics": {key: quantiles(metric_values[key]) for key in NUMERIC_METRICS},
        "sample_eval_overlap": {
            "exact_prompt_rows": exact_eval_rows,
            "eval_13gram_rows_bounded_head_tail": ngram_eval_rows,
            "eval_prompt_count": len(eval_index["exact"]),
            "eval_13gram_count": len(eval_index["ngrams"]),
            "eval_files": list(eval_index.get("files", [])),
            "hashed_overlap_receipts": overlap_receipts,
            "overlap_receipts_truncated": overlap_rows > len(overlap_receipts),
        },
        "review_rows": len(reviews),
        "review_selection": "lowest SHA-256 priorities over stable source identities",
        "review_packet_contains_text": bool(text_rows),
        "review_packet_contains_context_text": any(
            bool(review["review_context_text"]) for review in reviews
        ),
    }
    return report, reviews


def _write_json_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ProfileError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists() or temporary.is_symlink():
        raise ProfileError(f"refusing existing partial output {temporary}")
    with temporary.open("x") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl_no_replace(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise ProfileError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists() or temporary.is_symlink():
        raise ProfileError(f"refusing existing partial output {temporary}")
    with temporary.open("x") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--nested-files-field")
    parser.add_argument("--nested-text-field", default="content")
    parser.add_argument("--nested-review-context-field")
    parser.add_argument("--max-files-per-record", type=int, default=8)
    parser.add_argument("--scan-rows", type=int, default=10_000)
    parser.add_argument("--review-rows", type=int, default=100)
    parser.add_argument("--max-review-chars", type=int, default=16_000)
    parser.add_argument("--shuffle-seed", type=int, default=20_260_728)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--evals-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--review-out", required=True)
    args = parser.parse_args()

    from datasets import (
        get_dataset_config_names,
        get_dataset_split_names,
        load_dataset,
    )
    from huggingface_hub import HfApi, hf_hub_download

    resolved_revision = args.revision or HfApi().dataset_info(args.dataset).sha
    configs = get_dataset_config_names(args.dataset, revision=resolved_revision)
    config = args.config
    if config is None and len(configs) == 1:
        config = configs[0]
    if config is None or config not in configs:
        raise ProfileError(
            f"select one valid config for {args.dataset}: {', '.join(configs)}"
        )
    splits = get_dataset_split_names(args.dataset, config, revision=resolved_revision)
    if args.split not in splits:
        raise ProfileError(f"split {args.split!r} not in {splits}")

    card_path = Path(
        hf_hub_download(
            repo_id=args.dataset,
            filename="README.md",
            repo_type="dataset",
            revision=resolved_revision,
        )
    )
    card_sha256 = hashlib.sha256(card_path.read_bytes()).hexdigest()
    stream = load_dataset(
        args.dataset,
        name=config,
        split=args.split,
        streaming=True,
        revision=resolved_revision,
    )
    if args.shuffle_buffer <= 0:
        raise ProfileError("--shuffle-buffer must be positive")
    stream = stream.shuffle(seed=args.shuffle_seed, buffer_size=args.shuffle_buffer)
    if args.nested_files_field:
        stream = flatten_nested_files(
            stream,
            files_field=args.nested_files_field,
            nested_text_field=args.nested_text_field,
            max_files_per_record=args.max_files_per_record,
            parent_review_context_field=args.nested_review_context_field,
        )
        args.text_field = "text"
    eval_index = load_eval_index(args.evals_dir)
    report, reviews = profile_rows(
        stream,
        dataset=args.dataset,
        config=config,
        text_field=args.text_field,
        scan_rows=args.scan_rows,
        review_rows=args.review_rows,
        max_review_chars=args.max_review_chars,
        eval_index=eval_index,
        review_context_field=(
            "_review_context_text" if args.nested_review_context_field else None
        ),
    )
    report.update(
        {
            "requested_revision": args.revision,
            "resolved_revision": resolved_revision,
            "dataset_card_sha256": card_sha256,
            "split": args.split,
            "discovered_configs": configs,
            "discovered_splits": splits,
            "shuffle_seed": args.shuffle_seed,
            "shuffle_buffer": args.shuffle_buffer,
            "nested_files_field": args.nested_files_field,
            "nested_text_field": (
                args.nested_text_field if args.nested_files_field else None
            ),
            "nested_review_context_field": (
                args.nested_review_context_field if args.nested_files_field else None
            ),
            "max_files_per_record": (
                args.max_files_per_record if args.nested_files_field else None
            ),
        }
    )
    _write_json_no_replace(Path(args.out), report)
    _write_jsonl_no_replace(Path(args.review_out), reviews)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "config": config,
                "revision": resolved_revision,
                "scanned_rows": report["scanned_rows"],
                "text_rows": report["text_rows"],
                "review_rows": report["review_rows"],
                "exact_duplicates": report["exact_duplicate_text_rows"],
                "exact_eval_rows": report["sample_eval_overlap"]["exact_prompt_rows"],
                "eval_13gram_rows": report["sample_eval_overlap"][
                    "eval_13gram_rows_bounded_head_tail"
                ],
                "out": args.out,
                "review_out": args.review_out,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    # Streaming backends can leave fsspec worker threads alive after both
    # no-replace outputs are durable. There is no cleanup-sensitive writer.
    os._exit(0)


if __name__ == "__main__":
    main()
