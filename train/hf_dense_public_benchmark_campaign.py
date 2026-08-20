#!/usr/bin/env python3
"""Run a resumable matched dense benchmark campaign on one GPU.

The runner deliberately separates model-visible question files from assessors.
It writes one durable JSONL record after every completion, so a preemption can
resume without regenerating completed identities or changing their prompts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable

from hf_dense_public_benchmark_pair import (
    DenseBenchmarkGenerationError,
    matched_render_prompt,
    model_context_limit,
    sha256_file,
    validate_model_receipt,
)
from hf_idr_interact import revision_prompt
from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
)

MANIFEST_SCHEMA = "shohin-dense-public-campaign-manifest-v1"
QUESTION_SCHEMA = "shohin-dense-public-benchmark-question-v1"
LEDGER_SCHEMA = "shohin-dense-public-campaign-ledger-v1"
REPORT_SCHEMA = "shohin-dense-public-campaign-report-v1"
STAGES = ("draft", "unchanged_continuation", "trained_revision")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise DenseBenchmarkGenerationError("campaign manifest schema differs")
    benchmarks = manifest.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        raise DenseBenchmarkGenerationError("campaign benchmark list is empty")
    names: set[str] = set()
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    for entry in benchmarks:
        if not isinstance(entry, dict):
            raise DenseBenchmarkGenerationError("campaign benchmark entry differs")
        name = str(entry.get("name", ""))
        question_path = Path(str(entry.get("questions", "")))
        maximum = entry.get("max_new_tokens")
        if (
            not name
            or name in names
            or not question_path.is_file()
            or isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum <= 0
        ):
            raise DenseBenchmarkGenerationError("campaign benchmark binding differs")
        names.add(name)
        count = 0
        with question_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                source = json.loads(line)
                identity = source.get("id")
                if (
                    source.get("schema") != QUESTION_SCHEMA
                    or source.get("benchmark") != name
                    or not isinstance(identity, str)
                    or len(identity) != 64
                    or identity in identities
                    or source.get("response_mode") not in {"general", "math", "code"}
                    or not isinstance(source.get("question"), str)
                    or not source["question"].strip()
                ):
                    raise DenseBenchmarkGenerationError("campaign question row differs")
                identities.add(identity)
                rows.append(
                    {
                        "id": identity,
                        "benchmark": name,
                        "upstream_id": str(source.get("upstream_id", "")),
                        "question": source["question"],
                        "response_mode": source["response_mode"],
                        "max_new_tokens": maximum,
                    }
                )
                count += 1
        if count <= 0 or ("rows" in entry and entry["rows"] != count):
            raise DenseBenchmarkGenerationError("campaign question cardinality differs")
    return manifest, rows


def load_ledger(path: Path, stage: str, ordered_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = row.get("id")
            if (
                row.get("schema") != LEDGER_SCHEMA
                or row.get("stage") != stage
                or index >= len(ordered_ids)
                or identity != ordered_ids[index]
                or identity in completed
                or not isinstance(row.get("completion"), str)
                or not isinstance(row.get("prompt_sha256"), str)
            ):
                raise DenseBenchmarkGenerationError(f"{stage} resume ledger differs")
            completed[identity] = row
    return completed


def append_ledger(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def stage_prompt(
    stage: str, row: dict[str, Any], drafts: dict[str, dict[str, Any]]
) -> str:
    if stage == "draft":
        return row["question"]
    draft = drafts.get(row["id"])
    if draft is None:
        raise DenseBenchmarkGenerationError("second pass lacks its bound draft")
    return revision_prompt(row["question"], draft["completion"], row["response_mode"])


def run_stage(
    *,
    stage: str,
    rows: list[dict[str, Any]],
    drafts: dict[str, dict[str, Any]],
    ledger_path: Path,
    model_root: Path,
    checkpoint: Path,
    model_loader: str,
    tokenizer: Any,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    import torch

    ordered_ids = [row["id"] for row in rows]
    completed = load_ledger(ledger_path, stage, ordered_ids)
    if len(completed) == len(rows):
        return {"rows": len(rows), "resumed_rows": len(rows), "generated_rows": 0}
    model, metadata, resolved_loader = _load_model(model_root, checkpoint, model_loader)
    stop_ids = _generation_stop_token_ids(tokenizer)
    context_limit = model_context_limit(model, tokenizer)
    started = time.monotonic()
    generated = 0
    for index, row in enumerate(rows[: len(completed)]):
        expected = stage_prompt(stage, row, drafts)
        if completed[row["id"]]["prompt_sha256"] != text_sha256(expected):
            raise DenseBenchmarkGenerationError(f"{stage} resumed prompt differs")
    cursor = len(completed)
    while cursor < len(rows):
        maximum = rows[cursor]["max_new_tokens"]
        batch_rows = []
        while (
            cursor + len(batch_rows) < len(rows)
            and len(batch_rows) < batch_size
            and rows[cursor + len(batch_rows)]["max_new_tokens"] == maximum
        ):
            batch_rows.append(rows[cursor + len(batch_rows)])
        prompts = [stage_prompt(stage, row, drafts) for row in batch_rows]
        rendered = [matched_render_prompt(tokenizer, prompt) for prompt in prompts]
        prompt_tokens = [
            len(tokenizer(text, add_special_tokens=True)["input_ids"])
            for text in rendered
        ]
        for row, tokens in zip(batch_rows, prompt_tokens, strict=True):
            if tokens + maximum > context_limit:
                raise DenseBenchmarkGenerationError(
                    f"{row['benchmark']}:{row['id']} exceeds context contract "
                    f"({tokens}+{maximum}>{context_limit})"
                )
        batch_seed = seed + cursor
        torch.manual_seed(batch_seed)
        torch.cuda.manual_seed_all(batch_seed)
        completions, usage = _generate_completions(
            model,
            tokenizer,
            rendered,
            True,
            "greedy",
            maximum,
            stop_ids,
        )
        for offset, (row, prompt, completion, token_count, token_usage) in enumerate(
            zip(batch_rows, prompts, completions, prompt_tokens, usage, strict=True)
        ):
            tokens, exhausted = token_usage
            record = {
                "schema": LEDGER_SCHEMA,
                "stage": stage,
                "id": row["id"],
                "benchmark": row["benchmark"],
                "prompt_sha256": text_sha256(prompt),
                "completion": completion,
                "prompt_tokens": token_count,
                "generated_tokens": int(tokens),
                "max_token_exhausted": bool(exhausted),
                "seed": batch_seed + offset,
            }
            append_ledger(ledger_path, record)
            completed[row["id"]] = record
            if stage == "draft":
                drafts[row["id"]] = record
            generated += 1
        cursor += len(batch_rows)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "rows": len(rows),
        "resumed_rows": len(rows) - generated,
        "generated_rows": generated,
        "elapsed_seconds": elapsed,
        "model_loader": resolved_loader,
        "adapter_metadata": metadata,
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise DenseBenchmarkGenerationError("refusing to replace campaign report")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.output_root.exists() and not args.output_root.is_dir():
        raise DenseBenchmarkGenerationError("campaign output root is not a directory")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest, rows = load_manifest(args.manifest)
    for checkpoint, expected, label in (
        (args.draft_checkpoint, args.draft_checkpoint_sha256, "draft"),
        (args.revision_checkpoint, args.revision_checkpoint_sha256, "revision"),
    ):
        if not checkpoint.is_file() or checkpoint.is_symlink():
            raise DenseBenchmarkGenerationError(f"{label} checkpoint is missing")
        if sha256_file(checkpoint) != expected:
            raise DenseBenchmarkGenerationError(f"{label} checkpoint hash differs")
    receipt = validate_model_receipt(
        args.model_receipt,
        args.model_source_root,
        args.model_revision,
        args.model_config_sha256,
    )
    if sha256_file(args.model_root / "config.json") != args.model_config_sha256:
        raise DenseBenchmarkGenerationError("loaded model config differs")
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    drafts = load_ledger(
        args.output_root / "draft.jsonl", "draft", [row["id"] for row in rows]
    )
    stage_reports = {}
    for stage, checkpoint, offset in (
        ("draft", args.draft_checkpoint, 0),
        ("unchanged_continuation", args.draft_checkpoint, len(rows)),
        ("trained_revision", args.revision_checkpoint, len(rows)),
    ):
        stage_reports[stage] = run_stage(
            stage=stage,
            rows=rows,
            drafts=drafts,
            ledger_path=args.output_root / f"{stage}.jsonl",
            model_root=args.model_root,
            checkpoint=checkpoint,
            model_loader=args.model_loader,
            tokenizer=tokenizer,
            seed=args.seed + offset,
            batch_size=args.batch_size,
        )
        if stage == "draft":
            drafts = load_ledger(
                args.output_root / "draft.jsonl", "draft", [row["id"] for row in rows]
            )
    coverage = {
        stage: len(
            load_ledger(
                args.output_root / f"{stage}.jsonl",
                stage,
                [row["id"] for row in rows],
            )
        )
        for stage in STAGES
    }
    if any(value != len(rows) for value in coverage.values()):
        raise DenseBenchmarkGenerationError("campaign terminal coverage differs")
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "host": args.host,
        "model_revision": args.model_revision,
        "model_tree_sha256": receipt["tree_sha256"],
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_payload": manifest,
        "rows": len(rows),
        "coverage": coverage,
        "draft_checkpoint_sha256": args.draft_checkpoint_sha256,
        "revision_checkpoint_sha256": args.revision_checkpoint_sha256,
        "generation_mode": "greedy",
        "matched_second_pass": True,
        "stage_reports": stage_reports,
    }
    _atomic_json(args.output_root / "report.json", report)
    return report


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-receipt", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-config-sha256", required=True)
    parser.add_argument("--model-loader", choices=("causal", "multimodal"), required=True)
    parser.add_argument("--draft-checkpoint", type=Path, required=True)
    parser.add_argument("--draft-checkpoint-sha256", required=True)
    parser.add_argument("--revision-checkpoint", type=Path, required=True)
    parser.add_argument("--revision-checkpoint-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026081901)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def main() -> int:
    report = run(parse_args())
    print(json.dumps({"status": report["status"], "rows": report["rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
