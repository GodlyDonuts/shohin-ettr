#!/usr/bin/env python3
"""Measure the frozen K=8 information ceiling of the trained OLMoE owner."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from hf_idr1_evaluate_reviser import load_rows, shard_bounds
from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
    _render_prompt,
)
from hf_product_reasoning_rollouts import score_completion


SCHEMA = "shohin-dpr1-panel-row-v1"
REPORT_SCHEMA = "shohin-dpr1-panel-ceiling-shard-v1"
DATA_REPORT_SCHEMA = "shohin-idr1-revision-data-report-v1"
OWNER_ARCHITECTURE = "shohin-rme1-moe-revision-v1"


class DPR1Error(RuntimeError):
    """The frozen DPR1 owner, data, or sampling contract differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise DPR1Error(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("xb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded); digest.update(encoded)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise DPR1Error(f"refusing existing report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    data_report = json.loads(args.data_report.read_text())
    expected = data_report.get("outputs", {}).get("development", {})
    if data_report.get("schema") != DATA_REPORT_SCHEMA or data_report.get("status") != "complete" or expected.get("sha256") != sha256_file(args.data) or Path(str(expected.get("path", ""))).resolve() != args.data.resolve():
        raise DPR1Error("DPR1 development binding differs")
    all_rows = load_rows(args.data, "development")
    start, end = shard_bounds(len(all_rows), args.shard_index, args.shard_count, 1)
    rows = all_rows[start:end]
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = _load_model(args.model_root, args.owner_checkpoint, args.model_loader)
    if metadata.get("architecture") != OWNER_ARCHITECTURE or metadata.get("rme1_draft_control") != "draft_unavailable" or int(metadata.get("update", -1)) != 256:
        raise DPR1Error("DPR1 trained owner differs")
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed + args.shard_index)
    torch.cuda.manual_seed_all(args.seed + args.shard_index)
    torch.cuda.reset_peak_memory_stats()
    output_rows = []
    fixed_correct = [0] * args.panel_size
    domain_oracle = Counter(); total_tokens = exhausted = diverse = 0
    started = time.monotonic()
    for index, row in enumerate(rows):
        prompt = _render_prompt(tokenizer, row["question"], True, False)
        completions, usage = _generate_completions(model, tokenizer, [prompt] * args.panel_size, True, "qwen-thinking", args.max_new_tokens, stop_ids)
        candidates = []
        for candidate_index, (completion, (tokens, hit_limit)) in enumerate(zip(completions, usage, strict=True)):
            result = score_completion(row["assessor"], completion, code_timeout=args.code_timeout)
            fixed_correct[candidate_index] += int(result["correct"])
            total_tokens += tokens; exhausted += int(hit_limit)
            candidates.append({"candidate_index": candidate_index, "completion": completion, "generated_tokens": tokens, "max_token_exhausted": hit_limit, **result})
        oracle = any(candidate["correct"] for candidate in candidates)
        domain_oracle[row["task"]] += int(oracle)
        distinct = len({normalize(candidate["completion"]) for candidate in candidates})
        diverse += int(distinct >= 2)
        output_rows.append({"schema": SCHEMA, "identity_sha256": row["identity_sha256"], "task": row["task"], "oracle_correct": oracle, "distinct_normalized_completions": distinct, "candidates": candidates})
        if (index + 1) % 16 == 0 or index + 1 == len(rows): print(f"[dpr1-panel] {index+1}/{len(rows)}", flush=True)
    torch.cuda.synchronize(); elapsed = time.monotonic() - started
    output_sha = atomic_lines(args.output, output_rows)
    report = {"schema": REPORT_SCHEMA, "status": "complete", "split": "development", "data": str(args.data.resolve()), "data_sha256": sha256_file(args.data), "data_report_sha256": sha256_file(args.data_report), "model_root": str(args.model_source_root.resolve()), "model_revision": args.model_revision, "model_loader": loader, "owner_checkpoint": str(args.owner_checkpoint.resolve()), "owner_checkpoint_sha256": sha256_file(args.owner_checkpoint), "owner_architecture": metadata["architecture"], "owner_update": metadata["update"], "owner_draft_control": metadata["rme1_draft_control"], "panel_size": args.panel_size, "sampling": {"mode": "qwen-thinking", "temperature": 1.0, "top_p": 0.95, "top_k": 20}, "max_new_tokens": args.max_new_tokens, "seed": args.seed, "effective_seed": args.seed + args.shard_index, "shard_index": args.shard_index, "shard_count": args.shard_count, "full_rows": len(all_rows), "row_start": start, "row_end": end, "rows": len(rows), "fixed_index_correct": fixed_correct, "oracle_correct": sum(row["oracle_correct"] for row in output_rows), "domain_oracle": dict(domain_oracle), "diverse_rows": diverse, "generated_tokens": total_tokens, "max_token_exhausted": exhausted, "elapsed_seconds": elapsed, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()), "output": str(args.output.resolve()), "output_sha256": output_sha}
    atomic_json(args.report, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True); parser.add_argument("--model-source-root", type=Path, required=True); parser.add_argument("--model-revision", required=True); parser.add_argument("--model-loader", default="causal"); parser.add_argument("--owner-checkpoint", type=Path, required=True); parser.add_argument("--data", type=Path, required=True); parser.add_argument("--data-report", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--report", type=Path, required=True); parser.add_argument("--shard-index", type=int, required=True); parser.add_argument("--shard-count", type=int, required=True); parser.add_argument("--panel-size", type=int, default=8); parser.add_argument("--max-new-tokens", type=int, default=768); parser.add_argument("--code-timeout", type=float, default=3.0); parser.add_argument("--seed", type=int, default=2026080924)
    args = parser.parse_args()
    if args.panel_size != 8: parser.error("DPR1 panel size differs")
    print(json.dumps(run(args), sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())

