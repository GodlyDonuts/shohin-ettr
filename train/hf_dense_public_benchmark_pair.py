#!/usr/bin/env python3
"""Generate a matched unchanged/trained-revision dense benchmark shard."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from hf_idr_aqc_interact import commit_trajectories, verify_release
from hf_idr_interact import revision_prompt
from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
)
from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages

QUESTION_SCHEMA = "shohin-dense-public-benchmark-question-v1"
REPORT_SCHEMA = "shohin-dense-public-benchmark-generation-v1"


class DenseBenchmarkGenerationError(RuntimeError):
    """The dense pair, model, data, or output contract differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_questions(path: Path, benchmark: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    identities: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("schema") != QUESTION_SCHEMA
                or row.get("benchmark") != benchmark
                or row.get("response_mode") not in {"general", "math", "code"}
                or not isinstance(row.get("question"), str)
                or not row["question"].strip()
            ):
                raise DenseBenchmarkGenerationError("question row differs")
            identity = row.get("id")
            if (
                not isinstance(identity, str)
                or len(identity) != 64
                or identity in identities
            ):
                raise DenseBenchmarkGenerationError("question identity differs")
            identities.add(identity)
            rows.append(
                {
                    "id": identity,
                    "upstream_id": str(row["upstream_id"]),
                    "question": row["question"],
                    "response_mode": row["response_mode"],
                }
            )
    if not rows:
        raise DenseBenchmarkGenerationError("question board is empty")
    return rows


def shard_bounds(total: int, index: int, count: int) -> tuple[int, int]:
    if total <= 0 or count <= 0 or not 0 <= index < count:
        raise DenseBenchmarkGenerationError("shard geometry differs")
    start = total * index // count
    end = total * (index + 1) // count
    if start >= end:
        raise DenseBenchmarkGenerationError("shard is empty")
    return start, end


def _stage(
    model_root: Path,
    checkpoint: Path | None,
    model_loader: str,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    seed: int,
    batch_size: int,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    import torch

    model, metadata, resolved_loader = _load_model(model_root, checkpoint, model_loader)
    stop_ids = _generation_stop_token_ids(tokenizer)
    outputs: list[str] = []
    usage_rows: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for offset in range(0, len(prompts), batch_size):
        batch = prompts[offset : offset + batch_size]
        batch_seed = seed + offset
        torch.manual_seed(batch_seed)
        torch.cuda.manual_seed_all(batch_seed)
        rendered = [matched_render_prompt(tokenizer, prompt) for prompt in batch]
        encoded_lengths = [
            len(tokenizer(text, add_special_tokens=True)["input_ids"])
            for text in rendered
        ]
        context_limit = model_context_limit(model, tokenizer)
        if any(length + max_new_tokens > context_limit for length in encoded_lengths):
            raise DenseBenchmarkGenerationError(
                "rendered prompt plus generation exceeds the pinned context limit"
            )
        completions, usage = _generate_completions(
            model,
            tokenizer,
            rendered,
            checkpoint is not None,
            "greedy",
            max_new_tokens,
            stop_ids,
        )
        outputs.extend(completions)
        usage_rows.extend(
            {
                "prompt_tokens": encoded_lengths[index],
                "generated_tokens": int(tokens),
                "max_token_exhausted": bool(exhausted),
                "seed": batch_seed + index,
            }
            for index, (tokens, exhausted) in enumerate(usage)
        )
    torch.cuda.synchronize()
    receipt = {
        "rows": len(prompts),
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "generated_tokens": sum(row["generated_tokens"] for row in usage_rows),
        "prompt_tokens": sum(row["prompt_tokens"] for row in usage_rows),
        "max_token_exhausted": sum(
            int(row["max_token_exhausted"]) for row in usage_rows
        ),
        "model_loader": resolved_loader,
        "adapter_checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
        "adapter_metadata": metadata,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outputs, usage_rows, receipt


def matched_render_prompt(tokenizer: Any, prompt: str) -> str:
    """Render one byte-identical model-visible envelope for both matched arms."""

    return render_reasoning_messages(
        tokenizer,
        [
            {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        enable_thinking=False,
    )


def model_context_limit(model: Any, tokenizer: Any) -> int:
    config = getattr(model, "config", None)
    candidates = [
        getattr(config, "max_position_embeddings", None),
        getattr(getattr(config, "text_config", None), "max_position_embeddings", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    limits = [
        int(value)
        for value in candidates
        if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 10**7
    ]
    if not limits:
        raise DenseBenchmarkGenerationError("model context limit is unavailable")
    return min(limits)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise DenseBenchmarkGenerationError("refusing to replace generation report")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_model_receipt(
    path: Path, source_model_root: Path, revision: str, config_sha256: str
) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != "shohin-dense-model-restoration-v1"
        or receipt.get("status") != "complete"
        or Path(receipt.get("model_root", "")).resolve() != source_model_root.resolve()
        or receipt.get("model_revision") != revision
        or receipt.get("config_sha256") != config_sha256
        or receipt.get("manifest_verified") is not True
        or receipt.get("symlinks") != 0
        or receipt.get("special_files") != 0
    ):
        raise DenseBenchmarkGenerationError("model restoration receipt differs")
    return receipt


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if args.report.exists():
        raise DenseBenchmarkGenerationError("generation report already exists")
    if args.draft_base and args.draft_checkpoint is not None:
        raise DenseBenchmarkGenerationError("base draft cannot also name a checkpoint")
    if not args.draft_base and args.draft_checkpoint is None:
        raise DenseBenchmarkGenerationError("draft checkpoint is required")
    for checkpoint, expected, label in (
        (args.draft_checkpoint, args.draft_checkpoint_sha256, "draft"),
        (args.revision_checkpoint, args.revision_checkpoint_sha256, "revision"),
    ):
        if checkpoint is None:
            continue
        if not checkpoint.is_file() or checkpoint.is_symlink():
            raise DenseBenchmarkGenerationError(f"{label} checkpoint is missing")
        if sha256_file(checkpoint) != expected:
            raise DenseBenchmarkGenerationError(f"{label} checkpoint hash differs")
    model_receipt = validate_model_receipt(
        args.model_receipt,
        args.model_source_root,
        args.model_revision,
        args.model_config_sha256,
    )
    if sha256_file(args.model_root / "config.json") != args.model_config_sha256:
        raise DenseBenchmarkGenerationError("loaded model config differs")
    all_rows = load_questions(args.questions, args.benchmark)
    start, end = shard_bounds(len(all_rows), args.shard_index, args.shard_count)
    rows = all_rows[start:end]
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    draft_checkpoint = None if args.draft_base else args.draft_checkpoint
    questions = [row["question"] for row in rows]
    drafts, draft_usage, draft_receipt = _stage(
        args.model_root,
        draft_checkpoint,
        args.model_loader,
        tokenizer,
        questions,
        args.max_new_tokens,
        args.seed,
        args.batch_size,
    )
    second_prompts = [
        revision_prompt(row["question"], draft, row["response_mode"])
        for row, draft in zip(rows, drafts, strict=True)
    ]
    controls, control_usage, control_receipt = _stage(
        args.model_root,
        draft_checkpoint,
        args.model_loader,
        tokenizer,
        second_prompts,
        args.max_new_tokens,
        args.seed + len(rows),
        args.batch_size,
    )
    revisions, revision_usage, revision_receipt = _stage(
        args.model_root,
        args.revision_checkpoint,
        args.model_loader,
        tokenizer,
        second_prompts,
        args.max_new_tokens,
        args.seed + len(rows),
        args.batch_size,
    )
    commits = None
    commit_receipt = None
    release_manifest = None
    if args.release_root is not None:
        release_manifest = verify_release(args.release_root, args.model_root)
        if (
            args.draft_checkpoint is None
            or Path(args.release_root / "draft_adapter.pt").resolve()
            != args.draft_checkpoint.resolve()
        ):
            raise DenseBenchmarkGenerationError("release draft checkpoint differs")
        commits, commit_receipt = commit_trajectories(
            args.model_root,
            args.draft_checkpoint,
            args.release_root / "commit.pt",
            args.release_root / "commit_report.json",
            tokenizer,
            questions,
            revisions,
            controls,
            args.batch_pairs,
        )

    interactions = []
    for index, (row, draft, control, revision) in enumerate(
        zip(rows, drafts, controls, revisions, strict=True)
    ):
        interaction = {
            **row,
            "internal_draft": draft,
            "unchanged_continuation": control,
            "trained_revision": revision,
            "generation": {
                "draft": draft_usage[index],
                "unchanged": control_usage[index],
                "revision": revision_usage[index],
            },
        }
        if commits is not None:
            interaction.update(commits[index])
        interactions.append(interaction)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "host": args.host,
        "benchmark": args.benchmark,
        "model_root": str(args.model_source_root.resolve()),
        "loaded_model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_config_sha256": args.model_config_sha256,
        "model_receipt": str(args.model_receipt.resolve()),
        "model_receipt_sha256": sha256_file(args.model_receipt),
        "model_tree_sha256": model_receipt["tree_sha256"],
        "questions": str(args.questions.resolve()),
        "questions_sha256": sha256_file(args.questions),
        "draft_checkpoint": (
            str(args.draft_checkpoint.resolve()) if args.draft_checkpoint else None
        ),
        "draft_checkpoint_sha256": args.draft_checkpoint_sha256,
        "revision_checkpoint": str(args.revision_checkpoint.resolve()),
        "revision_checkpoint_sha256": args.revision_checkpoint_sha256,
        "generation_mode": "greedy",
        "max_new_tokens_per_stage": args.max_new_tokens,
        "matched_two_pass_budget": True,
        "identical_second_pass_decoding": True,
        "seed": args.seed,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "full_rows": len(all_rows),
        "row_start": start,
        "row_end": end,
        "stages": {
            "draft": draft_receipt,
            "unchanged": control_receipt,
            "trained_revision": revision_receipt,
            "commit_diagnostic": commit_receipt,
        },
        "release_manifest_sha256": (
            sha256_file(args.release_root / "manifest.json")
            if args.release_root
            else None
        ),
        "release_model_revision": (
            release_manifest["model_revision"] if release_manifest else None
        ),
        "interactions": interactions,
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument(
        "--benchmark", choices=("mmlu_pro", "ifeval", "musr"), required=True
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path, required=True)
    parser.add_argument("--model-receipt", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-config-sha256", required=True)
    parser.add_argument(
        "--model-loader", choices=("causal", "multimodal"), required=True
    )
    parser.add_argument("--draft-checkpoint", type=Path)
    parser.add_argument("--draft-checkpoint-sha256")
    parser.add_argument("--draft-base", action="store_true")
    parser.add_argument("--revision-checkpoint", type=Path, required=True)
    parser.add_argument("--revision-checkpoint-sha256", required=True)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-pairs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026081519)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.batch_size <= 0 or args.batch_pairs <= 0:
        parser.error("token and batch limits must be positive")
    if args.draft_checkpoint is not None and not args.draft_checkpoint_sha256:
        parser.error("draft checkpoint hash is required")
    return args


def main() -> int:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "host": report["host"],
                "benchmark": report["benchmark"],
                "rows": len(report["interactions"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
