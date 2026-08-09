#!/usr/bin/env python3
"""Run the frozen untrained DSET script ceiling on a stronger MoE host."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import time

import torch

from dset1_edit_transducer import DSET1Error, execute_script, parse_script
from hf_product_reasoning_eval import _generate_completions, _generation_stop_token_ids, _load_model, _render_prompt
from pset1_runtime import load_rows, sha256_file


SCHEMA = "shohin-dset-q35-ceiling-shard-v1"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    if args.output.exists() or not 0 <= args.shard_index < args.shard_count:
        raise RuntimeError("DSET-Q35 output exists or shard differs")
    pset_rows, _ = load_rows(args.pset_data, args.pset_report, "diagnostic")
    selected_sources = {row["source_identity_sha256"] for row in pset_rows}
    dset_rows = [json.loads(line) for line in args.dset_data.read_text().splitlines() if line]
    dset_rows = [row for row in dset_rows if row["source_identity_sha256"] in selected_sources]
    dset_rows.sort(key=lambda row: (row["source_identity_sha256"], row["pair_member"]))
    if len(dset_rows) != 512:
        raise RuntimeError("DSET-Q35 mapped row count differs")
    selected = [row for index, row in enumerate(dset_rows) if index % args.shard_count == args.shard_index]
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = _load_model(args.model_root, None, "causal")
    if metadata is not None:
        raise RuntimeError("DSET-Q35 host is unexpectedly adapted")
    stop_ids = _generation_stop_token_ids(tokenizer)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    results = []
    for row in selected:
        rendered = _render_prompt(tokenizer, row["question"], True, False)
        completions, usages = _generate_completions(
            model, tokenizer, [rendered], False, "greedy", 32, stop_ids
        )
        completion = completions[0]
        used, exhausted = usages[0]
        executed = ""
        error = None
        try:
            executed = execute_script(row["draft"], parse_script(completion))
        except DSET1Error as exc:
            error = str(exc)
        results.append({
            "source_identity_sha256": row["source_identity_sha256"],
            "pair_member": row["pair_member"],
            "corruption_family": row["corruption_family"],
            "completion": completion,
            "script_exact": completion.strip() == row["script"].strip(),
            "execution_correct": executed == row["final_response"],
            "execution_error": error,
            "generated_tokens": used,
            "max_token_exhausted": exhausted,
        })
        print(f"[dset-q35] shard={args.shard_index} rows={len(results)}/{len(selected)}", flush=True)
    elapsed = time.monotonic() - started
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "row_count": len(results),
        "model_root": str(args.model_root.resolve()),
        "model_config_sha256": sha256_file(args.model_root / "config.json"),
        "model_loader": loader,
        "dset_data_sha256": sha256_file(args.dset_data),
        "pset_data_sha256": sha256_file(args.pset_data),
        "script_exact": sum(row["script_exact"] for row in results),
        "execution_correct": sum(row["execution_correct"] for row in results),
        "execution_errors": dict(Counter(row["execution_error"] for row in results if row["execution_error"])),
        "max_token_exhausted": sum(row["max_token_exhausted"] for row in results),
        "generated_tokens": sum(row["generated_tokens"] for row in results),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--dset-data", type=Path, required=True)
    parser.add_argument("--pset-data", type=Path, required=True)
    parser.add_argument("--pset-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026080918)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
