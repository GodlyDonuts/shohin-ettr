#!/usr/bin/env python3
"""Generate one deterministic Q36 owner draft for each nonsealed identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from q36_mtr_roles import (
    ARCHITECTURE,
    DRAFT_IDENTITIES,
    DRAFT_MAX_NEW_TOKENS,
    DRAFT_SEED,
    DRAFT_SHARDS,
    MODEL_REVISION,
    OWNER_UPDATES,
    Q36MTRRoleError,
    TRAINABLE_PARAMETERS,
    validate_contract,
)

SCHEMA = "shohin-q36-mtr-model-draft-v1"
REPORT_SCHEMA = "shohin-q36-mtr-draft-shard-v1"
FREEZE_SCHEMA = "shohin-pcf1-data-freeze-report-v1"
SOURCE_SCHEMAS = {
    "train": "shohin-pcf1-train-source-v1",
    "development": "shohin-pcf1-development-source-v1",
}
EXPECTED_COUNTS = {"train": 5_824, "development": 1_289}


class Q36MTRDraftError(RuntimeError):
    """The Q36-MTR source-owner draft contract differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise Q36MTRDraftError(f"refusing existing Q36 draft output: {path}")
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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise Q36MTRDraftError(f"refusing existing Q36 draft report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_sources(
    train_path: Path,
    development_path: Path,
    freeze_report_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report = json.loads(freeze_report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != FREEZE_SCHEMA
        or report.get("status") != "complete"
        or report.get("source_disjoint") is not True
        or report.get("sealed_content_materialized") is not False
        or report.get("counts")
        != {"train": 5_824, "development": 1_289, "holdout": 1_279}
    ):
        raise Q36MTRDraftError("Q36-MTR source freeze differs")
    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    for split, path in (("train", train_path), ("development", development_path)):
        expected_output = report.get("outputs", {}).get(f"{split}_sources.jsonl", {})
        if (
            expected_output.get("sha256") != sha256_file(path)
            or int(expected_output.get("rows", -1)) != EXPECTED_COUNTS[split]
        ):
            raise Q36MTRDraftError(f"Q36-MTR {split} source receipt differs")
        split_rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(split_rows) != EXPECTED_COUNTS[split]:
            raise Q36MTRDraftError(f"Q36-MTR {split} source count differs")
        for row in split_rows:
            identity = str(row.get("identity_sha256", ""))
            if (
                row.get("schema") != SOURCE_SCHEMAS[split]
                or row.get("split") != split
                or row.get("runtime_fields") != ["source_prompt"]
                or len(identity) != 64
                or not isinstance(row.get("source_prompt"), str)
                or not row["source_prompt"].strip()
                or identity in identities
            ):
                raise Q36MTRDraftError("Q36-MTR source projection differs")
            if split == "development" and any(
                field in row for field in ("assessor", "response", "answer", "gold")
            ):
                raise Q36MTRDraftError("Q36-MTR development source exposes supervision")
            identities.add(identity)
            rows.append(row)
    rows.sort(key=lambda row: str(row["identity_sha256"]))
    if len(rows) != DRAFT_IDENTITIES:
        raise Q36MTRDraftError("Q36-MTR source identity geometry differs")
    return rows, report


def validate_owner_metadata(metadata: dict[str, Any]) -> None:
    try:
        validate_contract(metadata, "owner")
    except Q36MTRRoleError as error:
        raise Q36MTRDraftError(str(error)) from error
    if (
        int(metadata.get("update", -1)) != OWNER_UPDATES
        or metadata.get("trainable_parameters") != TRAINABLE_PARAMETERS
        or metadata.get("source_only_model_visible") is not True
        or metadata.get("internal_draft_visible") is not False
        or metadata.get("draft_control") != "normal"
    ):
        raise Q36MTRDraftError("Q36-MTR source owner differs")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    from hf_product_reasoning_eval import (
        _generate_completions,
        _generation_stop_token_ids,
        _load_model,
        _render_prompt,
    )
    from hf_pcf1_evaluate import nonpadding_prompt_tokens

    if (
        args.model_revision != MODEL_REVISION
        or args.seed != DRAFT_SEED
        or args.shard_count != DRAFT_SHARDS
        or args.max_new_tokens != DRAFT_MAX_NEW_TOKENS
        or args.batch_size != 4
        or not 0 <= args.shard_index < DRAFT_SHARDS
    ):
        raise Q36MTRDraftError("Q36-MTR draft settings differ")
    rows, freeze_report = load_sources(
        args.train_source, args.development_source, args.freeze_report
    )
    row_start = len(rows) * args.shard_index // args.shard_count
    row_end = len(rows) * (args.shard_index + 1) // args.shard_count
    shard_rows = rows[row_start:row_end]

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, loader = _load_model(
        args.model_root, args.owner_checkpoint, "causal", quantization="nf4"
    )
    validate_owner_metadata(metadata)
    stop_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    outputs: list[dict[str, Any]] = []
    prompt_tokens = generated_tokens = exhausted = 0
    started = time.monotonic()
    for offset in range(0, len(shard_rows), args.batch_size):
        batch = shard_rows[offset : offset + args.batch_size]
        rendered = [
            _render_prompt(tokenizer, str(row["source_prompt"]), True, False)
            for row in batch
        ]
        prompt_tokens += nonpadding_prompt_tokens(tokenizer, rendered)
        batch_started = time.monotonic()
        completions, usage = _generate_completions(
            model,
            tokenizer,
            rendered,
            True,
            "greedy",
            args.max_new_tokens,
            stop_ids,
        )
        batch_wall_seconds = (time.monotonic() - batch_started) / len(batch)
        for row, completion, (token_count, hit_limit) in zip(
            batch, completions, usage, strict=True
        ):
            completion = completion.strip()
            if not completion:
                raise Q36MTRDraftError("Q36-MTR source owner emitted an empty draft")
            outputs.append(
                {
                    "schema": SCHEMA,
                    "identity_sha256": row["identity_sha256"],
                    "split": row["split"],
                    "task": row["task"],
                    "prompt_sha256": hashlib.sha256(
                        str(row["source_prompt"]).encode()
                    ).hexdigest(),
                    "owner_checkpoint_sha256": sha256_file(args.owner_checkpoint),
                    "model_revision": MODEL_REVISION,
                    "completion": completion,
                    "generated_tokens": int(token_count),
                    "max_token_exhausted": bool(hit_limit),
                    "finish_reason": "length" if hit_limit else "stop",
                    "wall_seconds": batch_wall_seconds,
                }
            )
            generated_tokens += int(token_count)
            exhausted += int(hit_limit)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    output_sha256 = _atomic_lines(args.output, outputs)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "model_revision": MODEL_REVISION,
        "model_loader": loader,
        "owner_architecture": ARCHITECTURE,
        "owner_checkpoint": str(args.owner_checkpoint.resolve()),
        "owner_checkpoint_sha256": sha256_file(args.owner_checkpoint),
        "owner_update": metadata["update"],
        "owner_role": metadata["role"],
        "freeze_report_sha256": sha256_file(args.freeze_report),
        "freeze_identity_receipts": freeze_report["identity_receipts"],
        "train_source_sha256": sha256_file(args.train_source),
        "development_source_sha256": sha256_file(args.development_source),
        "generation_mode": "greedy",
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "full_rows": len(rows),
        "row_start": row_start,
        "row_end": row_end,
        "rows": len(outputs),
        "ordered_identity_sha256": hashlib.sha256(
            ("\n".join(row["identity_sha256"] for row in outputs) + "\n").encode()
        ).hexdigest(),
        "generated_tokens": generated_tokens,
        "prompt_tokens": prompt_tokens,
        "max_token_exhausted": exhausted,
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "capability_scored": False,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--development-source", type=Path, required=True)
    parser.add_argument("--freeze-report", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--owner-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=DRAFT_SHARDS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=DRAFT_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=DRAFT_SEED)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(run(parse_args()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
