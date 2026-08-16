#!/usr/bin/env python3
"""Generate a development-only Q36 owner preview without changing live shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from hf_q36_mtr_generate_drafts import (
    _atomic_json,
    _atomic_lines,
    exact_model_owned_completion,
    load_sources,
    sha256_file,
    validate_owner_metadata,
)
from q36_mtr_roles import (
    DRAFT_MAX_NEW_TOKENS,
    DRAFT_SEED,
    DRAFT_SHARDS,
    MODEL_REVISION,
)


class Q36MTROwnerPreviewError(RuntimeError):
    """Raised when owner preview geometry differs."""


def select_preview_rows(
    rows: list[dict[str, Any]], canonical_shard_index: int
) -> tuple[list[dict[str, Any]], int, int]:
    if not 0 <= canonical_shard_index < DRAFT_SHARDS:
        raise Q36MTROwnerPreviewError("owner preview shard index differs")
    row_start = len(rows) * canonical_shard_index // DRAFT_SHARDS
    row_end = len(rows) * (canonical_shard_index + 1) // DRAFT_SHARDS
    selected = [
        row for row in rows[row_start:row_end] if row.get("split") == "development"
    ]
    if not selected:
        raise Q36MTROwnerPreviewError("owner preview development slice is empty")
    return selected, row_start, row_end


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    from hf_product_reasoning_eval import (
        GENERATED_ONLY_SEQUENCE_CONTRACT,
        _generate_completions,
        _generation_stop_token_ids,
        _render_prompt,
    )
    from hf_q36_mtr_evaluate import (
        load_q36_adapter_model,
        q36_nonpadding_prompt_tokens,
    )

    if (
        args.model_revision != MODEL_REVISION
        or args.seed != DRAFT_SEED
        or args.max_new_tokens != DRAFT_MAX_NEW_TOKENS
        or args.batch_size != 4
        or args.output.exists()
        or args.output.is_symlink()
        or args.report.exists()
        or args.report.is_symlink()
    ):
        raise Q36MTROwnerPreviewError("owner preview settings differ")
    if sha256_file(args.environment_receipt) != args.environment_receipt_sha256:
        raise Q36MTROwnerPreviewError("owner preview environment receipt differs")
    environment = json.loads(args.environment_receipt.read_text(encoding="utf-8"))
    if (
        environment.get("schema") != "shohin-q36-mtr-environment-v1"
        or environment.get("status") != "pass"
        or environment.get("model_revision") != MODEL_REVISION
        or environment.get("environment_tree_sha256") != args.environment_tree_sha256
    ):
        raise Q36MTROwnerPreviewError("owner preview environment contract differs")
    rows, freeze_report = load_sources(
        args.train_source, args.development_source, args.freeze_report
    )
    selected, row_start, row_end = select_preview_rows(rows, args.canonical_shard_index)
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = load_q36_adapter_model(
        args.model_root, args.owner_checkpoint
    )
    validate_owner_metadata(metadata)
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    outputs: list[dict[str, Any]] = []
    prompt_tokens = generated_tokens = exhausted = 0
    checkpoint_sha256 = sha256_file(args.owner_checkpoint)
    started = time.monotonic()
    for offset in range(0, len(selected), args.batch_size):
        batch = selected[offset : offset + args.batch_size]
        rendered = [
            _render_prompt(tokenizer, str(row["source_prompt"]), True, False)
            for row in batch
        ]
        prompt_tokens += q36_nonpadding_prompt_tokens(tokenizer, rendered)
        batch_started = time.monotonic()
        completions, usage = _generate_completions(
            model,
            tokenizer,
            rendered,
            True,
            "greedy",
            args.max_new_tokens,
            stop_ids,
            add_special_tokens=False,
        )
        wall_seconds = (time.monotonic() - batch_started) / len(batch)
        for row, completion, (token_count, hit_limit) in zip(
            batch, completions, usage, strict=True
        ):
            outputs.append(
                {
                    "schema": "shohin-q36-mtr-model-draft-v1",
                    "identity_sha256": row["identity_sha256"],
                    "split": "development",
                    "task": row["task"],
                    "prompt_sha256": hashlib.sha256(
                        str(row["source_prompt"]).encode()
                    ).hexdigest(),
                    "owner_checkpoint_sha256": checkpoint_sha256,
                    "model_revision": MODEL_REVISION,
                    "completion": exact_model_owned_completion(completion),
                    "generated_tokens": int(token_count),
                    "max_token_exhausted": bool(hit_limit),
                    "finish_reason": "length" if hit_limit else "stop",
                    "wall_seconds": wall_seconds,
                }
            )
            generated_tokens += int(token_count)
            exhausted += int(hit_limit)
    torch.cuda.synchronize()
    output_sha256 = _atomic_lines(args.output, outputs)
    report = {
        "schema": "shohin-q36-mtr-owner-preview-generation-v1",
        "status": "complete",
        "interpretation": "exploratory_development_only_owner_comparison",
        "model_revision": MODEL_REVISION,
        "model_loader": loader,
        "owner_checkpoint": str(args.owner_checkpoint.resolve()),
        "owner_checkpoint_sha256": checkpoint_sha256,
        "owner_metadata_interpolation": metadata.get("interpolation"),
        "environment_receipt_sha256": args.environment_receipt_sha256,
        "environment_tree_sha256": args.environment_tree_sha256,
        "freeze_report_sha256": sha256_file(args.freeze_report),
        "freeze_identity_receipts": freeze_report["identity_receipts"],
        "canonical_shard_index": args.canonical_shard_index,
        "canonical_shard_count": DRAFT_SHARDS,
        "canonical_row_start": row_start,
        "canonical_row_end": row_end,
        "split": "development",
        "rows": len(outputs),
        "batch_size": args.batch_size,
        "generation_mode": "greedy",
        "generation_sequence_contract": GENERATED_ONLY_SEQUENCE_CONTRACT,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "max_token_exhausted": exhausted,
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "ordered_identity_sha256": hashlib.sha256(
            ("\n".join(row["identity_sha256"] for row in outputs) + "\n").encode()
        ).hexdigest(),
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--development-source", type=Path, required=True)
    parser.add_argument("--freeze-report", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--owner-checkpoint", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt-sha256", required=True)
    parser.add_argument("--environment-tree-sha256", required=True)
    parser.add_argument("--canonical-shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=DRAFT_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=DRAFT_SEED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
