#!/usr/bin/env python3
"""Convert paired DSET rows into unconditional idempotent edit transactions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path

from dset1_edit_transducer import REPLACE_LAST, execute_script, parse_script, render_script
from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages
from train_dset1_span_edit import DATA_REPORT_SCHEMA, DATA_SCHEMA, sha256_file
from ttr1_revision import tokenize_with_draft_mask


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def identity(row: dict, script: str) -> str:
    payload = f"iset1\0{row['identity_sha256']}\0{script}"
    return hashlib.sha256(payload.encode()).hexdigest()


def transform(path: Path, tokenizer, maximum: int, script_maximum: int):
    grouped = defaultdict(list)
    for line in path.read_text().splitlines():
        if line:
            row = json.loads(line)
            if row.get("schema") != DATA_SCHEMA:
                raise RuntimeError("ISET1 source row differs")
            grouped[row["pair_identity_sha256"]].append(row)
    output, maxima = [], Counter()
    for pair_id in sorted(grouped):
        pair = sorted(grouped[pair_id], key=lambda row: row["pair_member"])
        if len(pair) != 2 or [row["pair_member"] for row in pair] != ["clean", "fault"]:
            raise RuntimeError("ISET1 source pair differs")
        clean, fault = pair
        fault_script = parse_script(fault["script"])
        if fault_script.action != REPLACE_LAST or not fault_script.new:
            raise RuntimeError("ISET1 fault transaction differs")
        clean_script = render_script(REPLACE_LAST, fault_script.new, fault_script.new)
        scripts = {"clean": clean_script, "fault": fault["script"]}
        swapped = {"clean": fault["script"], "fault": clean_script}
        for row in pair:
            member = row["pair_member"]
            converted = dict(row)
            converted["source_dset_identity_sha256"] = row["identity_sha256"]
            converted["identity_sha256"] = identity(row, scripts[member])
            converted["action"] = REPLACE_LAST
            converted["script"] = scripts[member]
            converted["swapped_script"] = swapped[member]
            if execute_script(converted["draft"], parse_script(converted["script"])) != converted["final_response"]:
                raise RuntimeError("ISET1 transaction does not restore the trajectory")
            rendered = render_reasoning_messages(
                tokenizer,
                [
                    {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                    {"role": "user", "content": converted["question"]},
                ],
                enable_thinking=False,
            )
            prompt, _, _ = tokenize_with_draft_mask(tokenizer, rendered)
            for key in ("script", "swapped_script"):
                encoded = tokenizer.encode(converted[key], add_special_tokens=False) + [tokenizer.eos_token_id]
                maxima[f"{key}_tokens"] = max(maxima[f"{key}_tokens"], len(encoded))
                if len(encoded) > script_maximum or len(prompt) + len(encoded) > maximum:
                    raise RuntimeError("ISET1 complete transaction overflows")
            maxima["prompt_tokens"] = max(maxima["prompt_tokens"], len(prompt))
            output.append(converted)
    return output, dict(maxima)


def run(args: argparse.Namespace) -> dict:
    from transformers import AutoTokenizer

    if args.output.exists():
        raise RuntimeError("ISET1 output exists")
    source_report = json.loads(args.source_report.read_text())
    if source_report.get("schema") != DATA_REPORT_SCHEMA or source_report.get("status") != "complete":
        raise RuntimeError("ISET1 source report differs")
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    train, train_max = transform(args.train, tokenizer, args.maximum, args.script_maximum)
    diagnostic, diagnostic_max = transform(args.diagnostic, tokenizer, args.maximum, args.script_maximum)
    train_sources = {row["source_identity_sha256"] for row in train}
    diagnostic_sources = {row["source_identity_sha256"] for row in diagnostic}
    if train_sources & diagnostic_sources:
        raise RuntimeError("ISET1 source split overlap differs")
    args.output.mkdir(parents=True)
    train_path = args.output / "train.jsonl"
    diagnostic_path = args.output / "diagnostic.jsonl"
    atomic_jsonl(train_path, train)
    atomic_jsonl(diagnostic_path, diagnostic)
    report = {
        "schema": DATA_REPORT_SCHEMA,
        "status": "complete",
        "architecture": "shohin-iset1-idempotent-edit-v1",
        "holdout_used": False,
        "complete_retention": True,
        "max_script_tokens": args.script_maximum,
        "max_sequence_length": args.maximum,
        "train_diagnostic_source_overlap": 0,
        "source_report_sha256": sha256_file(args.source_report),
        "source_train_sha256": sha256_file(args.train),
        "source_diagnostic_sha256": sha256_file(args.diagnostic),
        "model_config_sha256": sha256_file(args.model_root / "config.json"),
        "maximum_tokens": {"train": train_max, "diagnostic": diagnostic_max},
        "outputs": {
            "train": {
                "path": str(train_path.resolve()), "sha256": sha256_file(train_path),
                "rows": len(train), "sources": len(train_sources),
            },
            "diagnostic": {
                "path": str(diagnostic_path.resolve()), "sha256": sha256_file(diagnostic_path),
                "rows": len(diagnostic), "sources": len(diagnostic_sources),
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
