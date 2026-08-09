#!/usr/bin/env python3
"""Build a complete-retention tokenizer-specific view of immutable DSET rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path

from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages
from train_dset1_span_edit import DATA_REPORT_SCHEMA, DATA_SCHEMA, sha256_file
from ttr1_revision import tokenize_with_draft_mask


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def filter_split(path: Path, tokenizer, maximum: int, script_maximum: int):
    grouped = defaultdict(list)
    for line in path.read_text().splitlines():
        if line:
            row = json.loads(line)
            if row.get("schema") != DATA_SCHEMA:
                raise RuntimeError("DSET transfer source row differs")
            grouped[row["pair_identity_sha256"]].append(row)
    kept, drops, maxima = [], Counter(), Counter()
    for pair_id in sorted(grouped):
        pair = sorted(grouped[pair_id], key=lambda row: row["pair_member"])
        if len(pair) != 2 or {row["pair_member"] for row in pair} != {"clean", "fault"}:
            raise RuntimeError("DSET transfer source pair differs")
        reason = None
        local = []
        for row in pair:
            rendered = render_reasoning_messages(
                tokenizer,
                [
                    {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                    {"role": "user", "content": row["question"]},
                ],
                enable_thinking=False,
            )
            prompt, _, _ = tokenize_with_draft_mask(tokenizer, rendered)
            script = tokenizer.encode(row["script"], add_special_tokens=False) + [
                tokenizer.eos_token_id
            ]
            local.append((len(prompt), len(script)))
            if len(script) > script_maximum:
                reason = "script_overflow"
            elif len(prompt) + len(script) > maximum:
                reason = "complete_sequence_overflow"
        if reason:
            drops[reason] += 1
            continue
        kept.extend(pair)
        for prompt, script in local:
            maxima["prompt"] = max(maxima["prompt"], prompt)
            maxima["script"] = max(maxima["script"], script)
            maxima["complete"] = max(maxima["complete"], prompt + script)
    return kept, dict(drops), dict(maxima)


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    if args.output.exists():
        raise RuntimeError("DSET transfer view output exists")
    source_report = json.loads(args.source_report.read_text())
    if source_report.get("schema") != DATA_REPORT_SCHEMA:
        raise RuntimeError("DSET transfer source report differs")
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    train, train_drops, train_max = filter_split(
        args.train, tokenizer, args.maximum, args.script_maximum
    )
    diagnostic, diagnostic_drops, diagnostic_max = filter_split(
        args.diagnostic, tokenizer, args.maximum, args.script_maximum
    )
    train_sources = {row["source_identity_sha256"] for row in train}
    diagnostic_sources = {row["source_identity_sha256"] for row in diagnostic}
    if not train or not diagnostic or train_sources & diagnostic_sources:
        raise RuntimeError("DSET transfer retained split differs")
    args.output.mkdir(parents=True)
    train_path = args.output / "train.jsonl"
    diagnostic_path = args.output / "diagnostic.jsonl"
    atomic_jsonl(train_path, train)
    atomic_jsonl(diagnostic_path, diagnostic)
    report = {
        "schema": DATA_REPORT_SCHEMA,
        "status": "complete",
        "holdout_used": False,
        "complete_retention": True,
        "max_script_tokens": args.script_maximum,
        "max_sequence_length": args.maximum,
        "train_diagnostic_source_overlap": 0,
        "source_report_sha256": sha256_file(args.source_report),
        "source_train_sha256": sha256_file(args.train),
        "source_diagnostic_sha256": sha256_file(args.diagnostic),
        "model_config_sha256": sha256_file(args.model_root / "config.json"),
        "drops": {"train": train_drops, "diagnostic": diagnostic_drops},
        "maximum_tokens": {"train": train_max, "diagnostic": diagnostic_max},
        "outputs": {
            "train": {
                "path": str(train_path.resolve()),
                "sha256": sha256_file(train_path),
                "rows": len(train),
                "sources": len(train_sources),
            },
            "diagnostic": {
                "path": str(diagnostic_path.resolve()),
                "sha256": sha256_file(diagnostic_path),
                "rows": len(diagnostic),
                "sources": len(diagnostic_sources),
            },
        },
    }
    atomic_json(args.output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum", type=int, default=4096)
    parser.add_argument("--script-maximum", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
