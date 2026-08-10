#!/usr/bin/env python3
"""Evaluate a frozen temporal reviser with model-owned executor receipts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import torch

from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
    _render_prompt,
    extract_gsm8k,
    has_explicit_final_answer,
    match_gsm8k,
)
from ttr1_revision import internal_revision_prompt


SCHEMA = "shohin-ectr0-executor-conditioned-revision-v1"
CONTROLS = ("aligned", "receipt_absent", "receipt_shuffled")


class ECTR0Error(RuntimeError):
    """The frozen ECTR0 contract was violated."""


def extract_ctf_claimed_final(completion: str) -> str | None:
    """Use CTF1's strict claimed-final marker, never a trailing-number fallback."""
    matches = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", completion)
    return matches[-1].replace(",", "") if matches else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise ECTR0Error(f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_jsonl(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise ECTR0Error(f"input hash differs: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    identities = [str(row.get("identity_sha256")) for row in rows]
    if len(rows) != 666 or len(set(identities)) != len(rows):
        raise ECTR0Error("development identity coverage differs")
    return rows


def load_ctf(path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise ECTR0Error("CTF1 report hash differs")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "shohin-ctf1-capability-floor-evaluation-v1"
        or report.get("status") != "complete"
        or report.get("control") != "normal"
        or report.get("holdout_used") is not False
        or report.get("public_test_opened") is not False
        or len(report.get("details", ())) != 666
    ):
        raise ECTR0Error("CTF1 report boundary differs")
    return report


def receipt_text(detail: dict[str, Any]) -> str:
    """Serialize only model-owned execution fields, never assessor fields."""
    if "prediction" not in detail:
        status = "COMPILE_INVALID"
        result = "UNAVAILABLE"
        transactions = state_reads = source_reads = literal_reads = 0
    elif "execution_error" in detail:
        status = "EXECUTION_INVALID"
        result = "UNAVAILABLE"
        transactions = int(detail.get("transactions", 0))
        state_reads = int(detail.get("state_reads", 0))
        source_reads = int(detail.get("source_reads", 0))
        literal_reads = int(detail.get("literal_reads", 0))
    else:
        status = "EXECUTED"
        result = str(detail["prediction"])
        transactions = int(detail.get("transactions", 0))
        state_reads = int(detail.get("state_reads", 0))
        source_reads = int(detail.get("source_reads", 0))
        literal_reads = int(detail.get("literal_reads", 0))
    return (
        "Learned executor receipt:\n"
        f"status={status}\n"
        f"result={result}\n"
        f"transactions={transactions}\n"
        f"state_reads={state_reads}\n"
        f"source_reads={source_reads}\n"
        f"literal_reads={literal_reads}"
    )


def shuffled_donors(
    rows: list[dict[str, Any]], details: dict[str, dict[str, Any]]
) -> dict[str, str]:
    groups: dict[tuple[bool, int], list[str]] = defaultdict(list)
    for row in rows:
        identity = str(row["identity_sha256"])
        groups[("prediction" in details[identity], int(row["register_depth"]))].append(identity)
    donors: dict[str, str] = {}
    for identities in groups.values():
        ordered = sorted(identities)
        if len(ordered) == 1:
            status = "prediction" in details[ordered[0]]
            fallback = sorted(
                identity
                for identity, detail in details.items()
                if ("prediction" in detail) == status and identity != ordered[0]
            )
            if not fallback:
                raise ECTR0Error("no shuffled receipt donor exists")
            donors[ordered[0]] = fallback[0]
            continue
        for index, identity in enumerate(ordered):
            donors[identity] = ordered[(index + 1) % len(ordered)]
    if set(donors) != {str(row["identity_sha256"]) for row in rows}:
        raise ECTR0Error("shuffled receipt donor coverage differs")
    if any(identity == donor for identity, donor in donors.items()):
        raise ECTR0Error("shuffled receipt retained an identity")
    return donors


def shard_bounds(total: int, index: int, count: int, batch_size: int) -> tuple[int, int]:
    if not 0 <= index < count or count <= 0 or batch_size <= 0:
        raise ECTR0Error("shard geometry differs")
    batches = (total + batch_size - 1) // batch_size
    first = batches * index // count
    last = batches * (index + 1) // count
    start = min(total, first * batch_size)
    end = min(total, last * batch_size)
    if start >= end:
        raise ECTR0Error("empty shard")
    return start, end


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.control not in CONTROLS:
        raise ECTR0Error("control differs")
    if args.max_new_tokens != 512 or args.seed != 2026081061 or args.max_sequence_length != 4096:
        raise ECTR0Error("frozen generation geometry differs")
    if sha256_file(args.adapter_checkpoint) != args.expected_adapter_sha256:
        raise ECTR0Error("revision checkpoint hash differs")
    rows = load_jsonl(args.data, args.expected_data_sha256)
    ctf = load_ctf(args.ctf_report, args.expected_ctf_sha256)
    details = {str(item["identity_sha256"]): item for item in ctf["details"]}
    if set(details) != {str(row["identity_sha256"]) for row in rows}:
        raise ECTR0Error("CTF/source identity join differs")
    donors = shuffled_donors(rows, details)
    start, end = shard_bounds(len(rows), args.shard_index, args.shard_count, args.batch_size)
    selected = rows[start:end]

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = _load_model(args.model_root, args.adapter_checkpoint, "auto")
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    output_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    generated_tokens = 0
    max_input_tokens = 0
    started = time.monotonic()
    for offset in range(0, len(selected), args.batch_size):
        batch = selected[offset : offset + args.batch_size]
        rendered: list[str] = []
        batch_receipt_sources: list[str | None] = []
        for row in batch:
            identity = str(row["identity_sha256"])
            detail = details[identity]
            if args.control == "aligned":
                receipt = receipt_text(detail)
                receipt_source: str | None = identity
            elif args.control == "receipt_shuffled":
                receipt_source = donors[identity]
                receipt = receipt_text(details[receipt_source])
            else:
                receipt_source = None
                receipt = "Learned executor receipt:\n<RECEIPT_UNAVAILABLE>"
            owner_completion = str(detail["completion"])
            draft = f"{owner_completion.rstrip()}\n\n{receipt}"
            prompt = internal_revision_prompt(str(row["original_question"]), draft, "gsm8k")
            chat = _render_prompt(tokenizer, prompt, True, False)
            token_count = len(tokenizer(chat, add_special_tokens=False)["input_ids"])
            if token_count > args.max_sequence_length:
                raise ECTR0Error("revision prompt truncates")
            max_input_tokens = max(max_input_tokens, token_count)
            rendered.append(chat)
            batch_receipt_sources.append(receipt_source)
        completions, usage = _generate_completions(
            model,
            tokenizer,
            rendered,
            True,
            "greedy",
            args.max_new_tokens,
            stop_ids,
        )
        for row, completion, usage_row, receipt_source in zip(
            batch, completions, usage, batch_receipt_sources, strict=True
        ):
            identity = str(row["identity_sha256"])
            detail = details[identity]
            gold = str(row["gold_answer"])
            prediction = extract_gsm8k(completion)
            correct = match_gsm8k(prediction, gold)
            direct_prediction = extract_ctf_claimed_final(str(detail["completion"]))
            direct_correct = match_gsm8k(direct_prediction, gold)
            explicit = has_explicit_final_answer(completion)
            token_count, exhausted = usage_row
            generated_tokens += token_count
            counters["rows"] += 1
            counters["correct"] += int(correct)
            counters["direct_correct"] += int(direct_correct)
            counters["explicit_final"] += int(explicit)
            counters["exhausted"] += int(exhausted)
            counters["repairs"] += int(correct and not direct_correct)
            counters["breaks"] += int(direct_correct and not correct)
            counters["executor_correct"] += int(bool(detail.get("correct")))
            output_rows.append(
                {
                    "identity_sha256": identity,
                    "register_depth": int(row["register_depth"]),
                    "receipt_source_identity_sha256": receipt_source,
                    "completion": completion,
                    "prediction": prediction,
                    "gold": gold,
                    "correct": correct,
                    "direct_prediction": direct_prediction,
                    "direct_correct": direct_correct,
                    "executor_prediction": detail.get("prediction"),
                    "executor_correct": bool(detail.get("correct")),
                    "explicit_final_answer": explicit,
                    "generated_tokens": token_count,
                    "exhausted": exhausted,
                }
            )
        processed = offset + len(batch)
        if processed % 32 == 0 or processed == len(selected):
            print(f"[ectr0:{args.control}] {processed}/{len(selected)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "public_test_opened": False,
        "control": args.control,
        "model_root": str(args.model_source_root.resolve()),
        "loaded_model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
        "adapter_metadata": metadata,
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "ctf_report": str(args.ctf_report.resolve()),
        "ctf_report_sha256": sha256_file(args.ctf_report),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "max_sequence_length": args.max_sequence_length,
        "batch_size": args.batch_size,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "row_start": start,
        "row_end": end,
        "max_input_tokens": max_input_tokens,
        "generated_tokens": generated_tokens,
        "elapsed_seconds": elapsed,
        "generated_tokens_per_second": generated_tokens / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "counts": dict(sorted(counters.items())),
        "details": output_rows,
    }
    atomic_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=CONTROLS, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-adapter-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--ctf-report", type=Path, required=True)
    parser.add_argument("--expected-ctf-sha256", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-sequence-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
