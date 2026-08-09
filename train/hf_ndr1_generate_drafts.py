#!/usr/bin/env python3
"""Generate one deterministic model-owned B1 draft per verified NDR1 source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any


SCHEMA = "shohin-ndr1-natural-drafts-v1"
REPORT_SCHEMA = "shohin-ndr1-natural-draft-report-v1"
SOURCE_REPORT_SCHEMA = "shohin-token-balanced-reasoning-mix-v1"


class NDR1DraftError(RuntimeError):
    """NDR1 source, model, shard, or output custody differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(row: dict[str, Any]) -> str:
    question = str(row.get("question", "")).strip()
    if not question:
        raise NDR1DraftError("NDR1 source question is empty")
    normalized = " ".join(question.casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise NDR1DraftError(f"refusing existing output: {path}")
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
        raise NDR1DraftError(f"refusing existing report: {path}")
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
    if (
        source_report.get("schema") != SOURCE_REPORT_SCHEMA
        or source_report.get("status") != "complete"
        or source_report.get("output_sha256") != sha256_file(args.source)
        or Path(str(source_report.get("output", ""))).resolve()
        != args.source.resolve()
    ):
        raise NDR1DraftError("NDR1 source report binding differs")
    all_rows = [json.loads(line) for line in args.source.read_text().splitlines() if line]
    if (
        args.shard_count < 1
        or not 0 <= args.shard_index < args.shard_count
        or not all_rows
    ):
        raise NDR1DraftError("NDR1 shard geometry differs")
    start = len(all_rows) * args.shard_index // args.shard_count
    end = len(all_rows) * (args.shard_index + 1) // args.shard_count
    rows = all_rows[start:end]
    identities = [source_identity(row) for row in rows]
    if len(identities) != len(set(identities)):
        raise NDR1DraftError("NDR1 shard source identity is duplicated")

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, resolved_loader = _load_model(
        args.model_root,
        args.adapter_checkpoint,
        args.model_loader,
    )
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    output_rows: list[dict[str, Any]] = []
    generated_tokens = exhausted = 0
    started = time.monotonic()
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        rendered = [
            _render_prompt(tokenizer, str(row["question"]), True, False)
            for row in batch
        ]
        completions, usage = _generate_completions(
            model,
            tokenizer,
            rendered,
            True,
            "greedy",
            args.max_new_tokens,
            stop_ids,
        )
        for batch_offset, (row, identity, completion, (tokens, hit_limit)) in enumerate(
            zip(
                batch,
                identities[offset : offset + len(batch)],
                completions,
                usage,
                strict=True,
            )
        ):
            output_rows.append(
                {
                    "schema": SCHEMA,
                    "source_identity_sha256": identity,
                    "source_index": start + offset + batch_offset,
                    "training_group": row["training_group"],
                    "completion": completion,
                    "generated_tokens": tokens,
                    "max_token_exhausted": hit_limit,
                }
            )
            generated_tokens += tokens
            exhausted += int(hit_limit)
    elapsed = time.monotonic() - started
    output_sha256 = atomic_lines(args.output, output_rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "source_report_sha256": sha256_file(args.source_report),
        "model_root": str(args.model_source_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": resolved_loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
        "adapter_update": metadata.get("update"),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "row_start": start,
        "row_end": end,
        "rows": len(output_rows),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
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
    parser.add_argument("--model-loader", default="multimodal")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=2026080919)
    report = run(parser.parse_args())
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
