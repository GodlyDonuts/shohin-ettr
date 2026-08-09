#!/usr/bin/env python3
"""Generate one deterministic trained-owner draft per unique MPR2 source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any


SCHEMA = "shohin-mpr2-trained-owner-draft-v1"
REPORT_SCHEMA = "shohin-mpr2-trained-owner-draft-shard-v1"
SOURCE_REPORT_SCHEMA = "shohin-mpr1-revision-data-report-v1"
OWNER_ARCHITECTURE = "shohin-rme1-moe-revision-v1"


class MPR2DraftError(RuntimeError):
    """The frozen MPR2 owner, source, or shard contract differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sources(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_source: dict[str, dict[str, Any]] = {}
    for row in rows:
        source = str(row.get("source_identity_sha256", ""))
        question = str(row.get("question", ""))
        if len(source) != 64 or not question.strip():
            raise MPR2DraftError("MPR2 source identity or prompt is invalid")
        prior = by_source.setdefault(source, row)
        if str(prior.get("question")) != question:
            raise MPR2DraftError("repeated MPR2 source prompt differs")
    return [by_source[key] for key in sorted(by_source)]


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise MPR2DraftError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise MPR2DraftError(f"refusing existing report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    from hf_product_reasoning_eval import (
        _generate_completions,
        _generation_stop_token_ids,
        _load_model,
        _render_prompt,
    )

    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    expected = source_report.get("outputs", {}).get("aligned", {})
    if (
        source_report.get("schema") != SOURCE_REPORT_SCHEMA
        or source_report.get("status") != "complete"
        or expected.get("sha256") != sha256_file(args.source)
        or Path(str(expected.get("path", ""))).resolve() != args.source.resolve()
        or source_report.get("holdout_used") is not False
    ):
        raise MPR2DraftError("MPR2 source report binding differs")
    source_rows = [json.loads(line) for line in args.source.read_text().splitlines() if line]
    rows = canonical_sources(source_rows)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise MPR2DraftError("MPR2 shard geometry differs")
    start = len(rows) * args.shard_index // args.shard_count
    end = len(rows) * (args.shard_index + 1) // args.shard_count
    rows = rows[start:end]

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, resolved_loader = _load_model(
        args.model_root, args.owner_checkpoint, args.model_loader
    )
    if (
        metadata.get("architecture") != OWNER_ARCHITECTURE
        or metadata.get("rme1_draft_control") != "draft_unavailable"
        or int(metadata.get("update", -1)) != args.expected_owner_update
    ):
        raise MPR2DraftError("MPR2 trained draft owner differs")
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    outputs: list[dict[str, Any]] = []
    generated_tokens = exhausted = 0
    started = time.monotonic()
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        rendered = [
            _render_prompt(tokenizer, str(row["question"]), True, False)
            for row in batch
        ]
        completions, usage = _generate_completions(
            model, tokenizer, rendered, True, "greedy", args.max_new_tokens, stop_ids
        )
        for row, completion, (tokens, hit_limit) in zip(batch, completions, usage, strict=True):
            if not completion.strip():
                raise MPR2DraftError("MPR2 trained owner emitted an empty draft")
            outputs.append(
                {
                    "schema": SCHEMA,
                    "source_identity_sha256": row["source_identity_sha256"],
                    "source_prompt_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
                    "completion": completion,
                    "generated_tokens": tokens,
                    "max_token_exhausted": hit_limit,
                }
            )
            generated_tokens += tokens
            exhausted += int(hit_limit)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    output_sha256 = atomic_lines(args.output, outputs)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "source_report_sha256": sha256_file(args.source_report),
        "unique_sources": len(canonical_sources(source_rows)),
        "owner_checkpoint": str(args.owner_checkpoint.resolve()),
        "owner_checkpoint_sha256": sha256_file(args.owner_checkpoint),
        "owner_architecture": metadata["architecture"],
        "owner_update": metadata["update"],
        "expected_owner_update": args.expected_owner_update,
        "owner_draft_control": metadata["rme1_draft_control"],
        "model_root": str(args.model_source_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": resolved_loader,
        "generation_mode": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "row_start": start,
        "row_end": end,
        "rows": len(outputs),
        "generated_tokens": generated_tokens,
        "max_token_exhausted": exhausted,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
    }
    atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", default="causal")
    parser.add_argument("--owner-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-owner-update", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=2026080921)
    args = parser.parse_args()
    if args.expected_owner_update <= 0:
        parser.error("expected owner update must be positive")
    print(json.dumps(run(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
