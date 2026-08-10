#!/usr/bin/env python3
"""Evaluate a trained source-to-ledger compiler on source-disjoint rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
)
from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages
from materialize_structured_ledger_sft import LedgerMaterializationError, parse_ledger


SCHEMA = "shohin-structured-ledger-compiler-evaluation-v1"


class LedgerCompilerEvalError(ValueError):
    """Raised when evaluation custody or geometry differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    identities = [row.get("identity_sha256") for row in rows]
    if not rows or any(not isinstance(identity, str) for identity in identities):
        raise LedgerCompilerEvalError("evaluation rows lack identities")
    if len(set(identities)) != len(identities):
        raise LedgerCompilerEvalError("evaluation identities are not unique")
    return rows


def shuffled_sources(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("family", "")), int(row.get("record_count", 0)))].append(row)
    donors: dict[str, dict[str, Any]] = {}
    for group in groups.values():
        ordered = sorted(group, key=lambda row: str(row["identity_sha256"]))
        if len(ordered) < 2:
            raise LedgerCompilerEvalError("source-shuffle stratum has one row")
        for index, row in enumerate(ordered):
            donor = next(
                (
                    ordered[(index + offset) % len(ordered)]
                    for offset in range(1, len(ordered))
                    if ordered[(index + offset) % len(ordered)]["response"]
                    != row["response"]
                ),
                None,
            )
            if donor is None:
                raise LedgerCompilerEvalError("source-shuffle stratum has one target")
            donors[str(row["identity_sha256"])] = donor
    return donors


def score_completion(completion: str, gold: str) -> dict[str, bool]:
    outcome = {
        "syntax_valid": False,
        "canonical_exact": completion.strip() == gold.strip(),
        "record_count_exact": False,
        "operation_sequence_exact": False,
        "all_records_exact": False,
        "terminal_exact": False,
    }
    try:
        predicted = parse_ledger(completion)
        expected = parse_ledger(gold)
    except LedgerMaterializationError:
        return outcome
    outcome["syntax_valid"] = True
    predicted_records = predicted["records"]
    expected_records = expected["records"]
    outcome["record_count_exact"] = len(predicted_records) == len(expected_records)
    outcome["operation_sequence_exact"] = [
        record["operation"] for record in predicted_records
    ] == [record["operation"] for record in expected_records]
    outcome["all_records_exact"] = predicted_records == expected_records
    outcome["terminal_exact"] = predicted["commit"]["value"] == expected["commit"]["value"]
    return outcome


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise LedgerCompilerEvalError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    data_sha256 = sha256_file(args.data)
    if data_sha256 != args.expected_data_sha256:
        raise LedgerCompilerEvalError("evaluation data SHA-256 differs")
    rows = load_rows(args.data)
    donors = shuffled_sources(rows) if args.control == "source_shuffled" else {}
    shard = [
        row
        for row in rows
        if int(str(row["identity_sha256"])[:16], 16) % args.shard_count
        == args.shard_index
    ]
    if args.limit:
        shard = sorted(shard, key=lambda row: str(row["identity_sha256"]))[: args.limit]
    if not shard:
        raise LedgerCompilerEvalError("evaluation shard is empty")

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = _load_model(args.model_root, args.checkpoint, "auto")
    if metadata is None or metadata.get("model_revision") != args.model_revision:
        raise LedgerCompilerEvalError("checkpoint revision differs")
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    totals: Counter[str] = Counter()
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    by_depth: dict[int, Counter[str]] = defaultdict(Counter)
    details: list[dict[str, Any]] = []
    generated_tokens = 0
    exhausted = 0
    for offset in range(0, len(shard), args.batch_size):
        batch = shard[offset : offset + args.batch_size]
        prompts = []
        for row in batch:
            source = donors.get(str(row["identity_sha256"]), row)
            prompts.append(
                render_reasoning_messages(
                    tokenizer,
                    [
                        {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                        {"role": "user", "content": str(source["question"])},
                    ],
                    enable_thinking=False,
                )
            )
        completions, usage = _generate_completions(
            model,
            tokenizer,
            prompts,
            True,
            "greedy",
            args.max_new_tokens,
            stop_ids,
        )
        for row, completion, (token_count, row_exhausted) in zip(
            batch, completions, usage, strict=True
        ):
            scored = score_completion(completion, str(row["response"]))
            family = str(row.get("family", ""))
            depth = int(row.get("record_count", 0))
            totals["rows"] += 1
            by_family[family]["rows"] += 1
            by_depth[depth]["rows"] += 1
            for name, passed in scored.items():
                totals[name] += int(passed)
                by_family[family][name] += int(passed)
                by_depth[depth][name] += int(passed)
            generated_tokens += token_count
            exhausted += int(row_exhausted)
            donor = donors.get(str(row["identity_sha256"]))
            details.append(
                {
                    "identity_sha256": row["identity_sha256"],
                    "donor_identity_sha256": donor["identity_sha256"] if donor else None,
                    "family": family,
                    "record_count": depth,
                    "completion": completion,
                    "generated_tokens": token_count,
                    "exhausted": row_exhausted,
                    **scored,
                }
            )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "control": args.control,
        "model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": loader,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_update": metadata["update"],
        "data": str(args.data.resolve()),
        "data_sha256": data_sha256,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": generated_tokens,
        "exhausted": exhausted,
        "counts": dict(sorted(totals.items())),
        "by_family": {
            key: dict(sorted(value.items())) for key, value in sorted(by_family.items())
        },
        "by_depth": {
            str(key): dict(sorted(value.items())) for key, value in sorted(by_depth.items())
        },
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "holdout_used": False,
        "details": details,
    }
    _atomic_json(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", choices=("normal", "source_shuffled"), required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--seed", type=int, default=2026081023)
    args = parser.parse_args()
    if (
        args.shard_count <= 0
        or args.shard_index < 0
        or args.shard_index >= args.shard_count
        or args.limit < 0
        or args.batch_size <= 0
        or args.max_new_tokens <= 0
    ):
        parser.error("evaluation dimensions differ")
    return args


def main() -> int:
    args = parse_args()
    report = run(args)
    print(
        f"[ledger-eval] control={args.control} shard={args.shard_index}/"
        f"{args.shard_count} terminal={report['counts'].get('terminal_exact', 0)}/"
        f"{report['counts']['rows']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
