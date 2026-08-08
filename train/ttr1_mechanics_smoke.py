#!/usr/bin/env python3
"""Run the non-capability TTR1 cross-family mechanics gate."""

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
    _task_prompt,
)
from ttr1_revision import internal_revision_prompt, tokenize_with_draft_mask


SCHEMA = "shohin-ttr1-mechanics-smoke-v1"


class TTR1SmokeError(RuntimeError):
    """The mechanics-only cross-family gate differs from its contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bank(path: Path, expected_sha256: str, count: int) -> list[dict[str, Any]]:
    if sha256_file(path) != expected_sha256:
        raise TTR1SmokeError(f"bank SHA-256 differs: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) == count:
                break
    if len(rows) != count or len({row.get("identity_sha256") for row in rows}) != count:
        raise TTR1SmokeError(f"bank mechanics sample differs: {path}")
    return rows


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing existing smoke report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as destination:
        json.dump(payload, destination, indent=2, sort_keys=True)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    if sha256_file(args.model_root / "config.json") != args.model_config_sha256:
        raise TTR1SmokeError("model config SHA-256 differs")
    if sha256_file(args.adapter_checkpoint) != args.adapter_checkpoint_sha256:
        raise TTR1SmokeError("adapter checkpoint SHA-256 differs")
    rows = [
        *load_bank(args.math_bank, args.math_bank_sha256, 8),
        *load_bank(args.science_bank, args.science_bank_sha256, 8),
        *load_bank(args.code_bank, args.code_bank_sha256, 8),
    ]
    if {str(row.get("task")) for row in rows} != {"math500", "bbh_logic", "mbpp"}:
        raise TTR1SmokeError("balanced mechanics task coverage differs")

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model, metadata, model_loader = _load_model(
        args.model_root, args.adapter_checkpoint, "auto"
    )
    if (
        model_loader != "causal"
        or metadata is None
        or metadata.get("update") != args.adapter_update
        or metadata.get("model_revision") != args.model_revision
        or metadata.get("arm") != "baseline"
        or metadata.get("unfreeze_layers") != 0
    ):
        raise TTR1SmokeError("causal checkpoint restoration metadata differs")

    stop_token_ids = _generation_stop_token_ids(tokenizer)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    source_prompts = [_task_prompt(str(row["task"]), row) for row in rows]
    rendered_drafts = [
        _render_prompt(tokenizer, prompt, True, False) for prompt in source_prompts
    ]
    started = time.monotonic()
    drafts, draft_usage = _generate_completions(
        model,
        tokenizer,
        rendered_drafts,
        True,
        "greedy",
        args.draft_tokens,
        stop_token_ids,
    )
    if len(drafts) != len(rows) or any(not draft.strip() for draft in drafts):
        raise TTR1SmokeError("draft generation cardinality/content differs")

    revision_questions = [
        internal_revision_prompt(prompt, draft, str(row["task"]))
        for row, prompt, draft in zip(rows, source_prompts, drafts, strict=True)
    ]
    treatment_rendered = [
        _render_prompt(tokenizer, question, True, False)
        for question in revision_questions
    ]
    unchanged_rendered = [
        _render_prompt(tokenizer, question, True, False)
        for question in revision_questions
    ]
    if treatment_rendered != unchanged_rendered:
        raise TTR1SmokeError("treatment and unchanged-second-pass prompts differ")

    prompt_tokens = 0
    masked_tokens = 0
    for rendered in treatment_rendered:
        treatment_ids = tokenizer.encode(rendered, add_special_tokens=False)
        control_ids, control_attention, _ = tokenize_with_draft_mask(
            tokenizer, rendered
        )
        if treatment_ids != control_ids or len(control_attention) != len(treatment_ids):
            raise TTR1SmokeError("independent control changes token geometry")
        hidden = control_attention.count(0)
        if hidden <= 0 or hidden >= len(control_attention):
            raise TTR1SmokeError("independent control draft mask differs")
        prompt_tokens += len(treatment_ids)
        masked_tokens += hidden

    revisions, revision_usage = _generate_completions(
        model,
        tokenizer,
        treatment_rendered,
        True,
        "greedy",
        args.revision_tokens,
        stop_token_ids,
    )
    if len(revisions) != len(rows):
        raise TTR1SmokeError("revision generation cardinality differs")
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    report = {
        "schema": SCHEMA,
        "status": "pass",
        "capability_result": False,
        "model_root": str((args.model_source_root or args.model_root).resolve()),
        "loaded_model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "model_config_sha256": args.model_config_sha256,
        "model_loader": model_loader,
        "adapter_checkpoint": str(args.adapter_checkpoint.resolve()),
        "adapter_checkpoint_sha256": args.adapter_checkpoint_sha256,
        "adapter_update": args.adapter_update,
        "rows": len(rows),
        "tasks": {task: 8 for task in ("math500", "bbh_logic", "mbpp")},
        "draft_generated_tokens": sum(count for count, _ in draft_usage),
        "revision_generated_tokens": sum(count for count, _ in revision_usage),
        "prompt_tokens": prompt_tokens,
        "masked_draft_tokens": masked_tokens,
        "treatment_unchanged_prompt_ids_identical": True,
        "masked_control_input_ids_identical": True,
        "masked_control_attention_differs_only_on_draft": True,
        "drafts_nonempty": all(bool(draft.strip()) for draft in drafts),
        "revisions_complete": len(revisions) == len(rows),
        "elapsed_seconds": elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "seed": args.seed,
    }
    atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-source-root", type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-config-sha256", required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint-sha256", required=True)
    parser.add_argument("--adapter-update", type=int, default=1000)
    parser.add_argument("--math-bank", type=Path, required=True)
    parser.add_argument("--math-bank-sha256", required=True)
    parser.add_argument("--science-bank", type=Path, required=True)
    parser.add_argument("--science-bank-sha256", required=True)
    parser.add_argument("--code-bank", type=Path, required=True)
    parser.add_argument("--code-bank-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--draft-tokens", type=int, default=64)
    parser.add_argument("--revision-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026080820)
    args = parser.parse_args()
    if args.draft_tokens <= 0 or args.revision_tokens <= 0 or args.adapter_update <= 0:
        parser.error("token and checkpoint dimensions must be positive")
    return args


def main() -> int:
    report = run(parse_args())
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
