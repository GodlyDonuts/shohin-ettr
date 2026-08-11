#!/usr/bin/env python3
"""Generate source-only PCF1 model drafts without loading or scoring supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
    _render_prompt,
)
from pcf1_environment import (
    PCF1EnvironmentError,
    validate_environment_receipt as validate_exact_environment_receipt,
)

DRAFT_SCHEMA = "shohin-pcf1-model-draft-v1"
REPORT_SCHEMA = "shohin-pcf1-draft-shard-v1"
FREEZE_REPORT_SCHEMA = "shohin-pcf1-data-freeze-report-v1"
SOURCE_SCHEMAS = {
    "train": "shohin-pcf1-train-source-v1",
    "development": "shohin-pcf1-development-source-v1",
}
TASKS = ("math500", "bbh_logic", "mbpp")
PINNED_MODEL_REVISION = "81eaece1948f3875421d9a45bc55487d10e2d894"
PINNED_SEED = 2026080818
PINNED_BATCH_SIZE = 2
PINNED_SHARDS = 16
PINNED_ROWS = 7113


class PCF1DraftError(RuntimeError):
    """A source, model, shard, or output violates the frozen draft contract."""


def shard_bounds(
    total: int, index: int, count: int, batch_size: int
) -> tuple[int, int]:
    if total <= 0 or count <= 0 or not 0 <= index < count or batch_size <= 0:
        raise PCF1DraftError("PCF1 draft shard geometry differs")
    batches = (total + batch_size - 1) // batch_size
    start_batch = batches * index // count
    end_batch = batches * (index + 1) // count
    start = min(total, start_batch * batch_size)
    end = min(total, end_batch * batch_size)
    if start >= end:
        raise PCF1DraftError("PCF1 draft shard is empty")
    return start, end


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_protected_path(path: Path) -> None:
    if any(word in str(path).casefold() for word in ("holdout", "product", "public")):
        raise PCF1DraftError(f"protected path supplied to PCF1 draft job: {path}")


def validate_environment_receipt(
    path: Path,
    expected_sha256: str,
    required_source: str = "train/hf_pcf1_generate_drafts.py",
) -> dict[str, Any]:
    try:
        return validate_exact_environment_receipt(
            path, expected_sha256, required_source
        )
    except PCF1EnvironmentError as error:
        raise PCF1DraftError("PCF1 environment receipt differs") from error


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise PCF1DraftError(f"refusing existing PCF1 draft output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    try:
        with temporary.open("xb") as handle:
            for row in rows:
                encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
                handle.write(encoded)
                digest.update(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except FileExistsError as error:
        raise PCF1DraftError(f"refusing existing PCF1 draft output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise PCF1DraftError(f"refusing existing PCF1 draft report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except FileExistsError as error:
        raise PCF1DraftError(f"refusing existing PCF1 draft report: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def load_source_split(
    source_root: Path, split: str
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if split not in SOURCE_SCHEMAS:
        raise PCF1DraftError("PCF1 source split differs")
    report_path = source_root / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != FREEZE_REPORT_SCHEMA
        or report.get("status") != "complete"
        or report.get("source_disjoint") is not True
        or report.get("sealed_content_materialized") is not False
    ):
        raise PCF1DraftError("PCF1 source freeze report differs")
    rows: list[dict[str, str]] = []
    identities: set[str] = set()
    schema = SOURCE_SCHEMAS[split]
    path = source_root / f"{split}_sources.jsonl"
    receipt = report.get("outputs", {}).get(path.name)
    if (
        not isinstance(receipt, dict)
        or receipt.get("sha256") != sha256_file(path)
        or receipt.get("rows") != report.get("counts", {}).get(split)
    ):
        raise PCF1DraftError(f"PCF1 {split} source receipt differs")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        identity = raw.get("identity_sha256")
        prompt = raw.get("source_prompt")
        task = raw.get("task")
        if (
            raw.get("schema") != schema
            or raw.get("split") != split
            or raw.get("runtime_fields") != ["source_prompt"]
            or not isinstance(identity, str)
            or len(identity) != 64
            or identity in identities
            or task not in TASKS
            or not isinstance(prompt, str)
            or not prompt.strip()
        ):
            raise PCF1DraftError("PCF1 source-only row differs")
        identities.add(identity)
        # Deliberately project the only model-visible field before model load.
        rows.append(
            {
                "identity_sha256": identity,
                "split": split,
                "task": str(task),
                "source_prompt": prompt,
            }
        )
    expected = int(report["counts"][split])
    if len(rows) != expected:
        raise PCF1DraftError(f"PCF1 {split} source coverage differs")
    return rows, report


def load_sources(source_root: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows: list[dict[str, str]] = []
    report: dict[str, Any] | None = None
    for split in SOURCE_SCHEMAS:
        split_rows, split_report = load_source_split(source_root, split)
        rows.extend(split_rows)
        if report is None:
            report = split_report
        elif split_report != report:
            raise PCF1DraftError("PCF1 source report changed while loading")
    assert report is not None
    identities = [row["identity_sha256"] for row in rows]
    if len(set(identities)) != len(identities):
        raise PCF1DraftError("PCF1 nonsealed source identities overlap")
    return rows, report


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    for path in (
        args.source_root,
        args.adapter_checkpoint,
        args.candidates_output,
        args.report,
    ):
        reject_protected_path(path)
    if (
        args.model_revision != PINNED_MODEL_REVISION
        or args.model_loader != "multimodal"
        or args.seed != PINNED_SEED
        or args.batch_size != PINNED_BATCH_SIZE
        or args.shard_count != PINNED_SHARDS
    ):
        raise PCF1DraftError("PCF1 frozen draft settings differ")
    if args.candidates_output.exists() or args.report.exists():
        raise PCF1DraftError("PCF1 draft output already exists")
    environment = validate_environment_receipt(
        args.environment_receipt, args.environment_receipt_sha256
    )
    rows, freeze_report = load_sources(args.source_root)
    if len(rows) != PINNED_ROWS:
        raise PCF1DraftError("PCF1 frozen source-only coverage differs")
    row_start, row_end = shard_bounds(
        len(rows), args.shard_index, args.shard_count, args.batch_size
    )
    shard = rows[row_start:row_end]

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = _load_model(
        args.model_root, args.adapter_checkpoint, args.model_loader
    )
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    generated: list[dict[str, Any]] = []
    prompt_tokens = generated_tokens = exhausted_count = 0
    for start in range(0, len(shard), args.batch_size):
        batch = shard[start : start + args.batch_size]
        rendered = [
            _render_prompt(tokenizer, row["source_prompt"], True, False)
            for row in batch
        ]
        prompt_tokens += sum(
            len(tokenizer.encode(prompt, add_special_tokens=False))
            for prompt in rendered
        )
        completions, usage = _generate_completions(
            model,
            tokenizer,
            rendered,
            True,
            "greedy",
            768,
            stop_ids,
        )
        for row, completion, (tokens, exhausted) in zip(
            batch, completions, usage, strict=True
        ):
            if not completion.strip():
                raise PCF1DraftError("PCF1 model-owned draft is empty")
            generated_tokens += tokens
            exhausted_count += int(exhausted)
            generated.append(
                {
                    "schema": DRAFT_SCHEMA,
                    "identity_sha256": row["identity_sha256"],
                    "split": row["split"],
                    "task": row["task"],
                    "completion": completion,
                    "generated_tokens": tokens,
                    "max_token_exhausted": exhausted,
                    "prompt_sha256": hashlib.sha256(
                        row["source_prompt"].encode()
                    ).hexdigest(),
                    "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
                    "model_revision": args.model_revision,
                    "finish_reason": "length" if exhausted else "stop",
                    "wall_seconds": 0.0,
                }
            )
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    per_row = elapsed / len(generated)
    for row in generated:
        row["wall_seconds"] = per_row
    candidates_sha256 = atomic_lines(args.candidates_output, generated)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "model_root": str(args.model_source_root.resolve()),
        "model_revision": args.model_revision,
        "model_loader": loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint),
        "adapter_metadata": metadata,
        "environment_receipt": str(args.environment_receipt.resolve()),
        "environment_receipt_sha256": args.environment_receipt_sha256,
        "environment_tree_sha256": environment["environment_tree"]["sha256"],
        "source_root": str(args.source_root.resolve()),
        "source_report_sha256": sha256_file(args.source_root / "report.json"),
        "source_counts": {
            split: freeze_report["counts"][split] for split in SOURCE_SCHEMAS
        },
        "runtime_fields": ["source_prompt"],
        "supervisor_fields_visible_to_model": False,
        "generation_mode": "greedy",
        "thinking_enabled": False,
        "max_new_tokens": 768,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "row_start": row_start,
        "row_end": row_end,
        "full_row_count": len(rows),
        "candidates_output": str(args.candidates_output.resolve()),
        "candidates_sha256": candidates_sha256,
        "rows": len(generated),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "max_token_exhausted": exhausted_count,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-loader", choices=("multimodal",), default="multimodal")
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--environment-receipt", type=Path, required=True)
    parser.add_argument("--environment-receipt-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--candidates-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=PINNED_BATCH_SIZE)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=PINNED_SHARDS)
    parser.add_argument("--seed", type=int, default=PINNED_SEED)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.shard_count <= 0:
        parser.error("PCF1 draft batch/shard geometry must be positive")
    return args


def main() -> int:
    report = run(parse_args())
    print(
        json.dumps(
            {"rows": report["rows"], "shard": report["shard_index"]}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
