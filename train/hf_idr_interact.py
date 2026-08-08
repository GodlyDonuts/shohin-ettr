#!/usr/bin/env python3
"""Interact with a same-family internal-draft/revision reasoning model."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from hf_product_reasoning_eval import (
    _generate_completions,
    _generation_stop_token_ids,
    _load_model,
    _render_prompt,
)


class IDRInteractionError(RuntimeError):
    """The interactive IDR model or prompt contract differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def revision_prompt(question: str, draft: str, response_mode: str) -> str:
    question = question.strip()
    draft = draft.strip()
    if not question or not draft:
        raise IDRInteractionError("question and internal draft must be nonempty")
    instructions = {
        "general": "Return the complete corrected answer, not a critique of the draft.",
        "math": "Return a complete corrected solution with the exact final answer in \\boxed{}.",
        "code": "Return only the complete executable code, without Markdown fences.",
    }
    if response_mode not in instructions:
        raise IDRInteractionError("unsupported response mode")
    return (
        "Solve the original request by checking and revising the model's earlier "
        "draft. The draft may contain useful work or errors; independently verify "
        "it and finish the task rather than returning a critique.\n\n"
        f"Original request:\n{question}\n\nInternal draft:\n{draft}\n\n"
        f"{instructions[response_mode]}\n\nOriginal request:\n{question}"
    )


def generate_one(
    model_root: Path,
    checkpoint: Path,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    seed: int,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, adapter_metadata, model_loader = _load_model(
        model_root, checkpoint, "multimodal"
    )
    rendered = _render_prompt(tokenizer, prompt, True, False)
    completions, usage = _generate_completions(
        model,
        tokenizer,
        [rendered],
        True,
        "greedy",
        max_new_tokens,
        _generation_stop_token_ids(tokenizer),
    )
    if len(completions) != 1 or len(usage) != 1:
        raise IDRInteractionError("interactive generation cardinality differs")
    token_count, exhausted = usage[0]
    generation = {
        "generated_tokens": token_count,
        "max_token_exhausted": exhausted,
        "seed": seed,
    }
    metadata = {
        "model_loader": model_loader,
        "adapter_metadata": adapter_metadata,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return completions[0], generation, metadata


def generate_many(
    model_root: Path,
    checkpoint: Path,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    seed: int,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Generate several independent prompts while loading one owner only once."""
    import torch

    if not prompts:
        raise IDRInteractionError("interactive prompt set must be nonempty")
    model, adapter_metadata, model_loader = _load_model(
        model_root, checkpoint, "multimodal"
    )
    outputs: list[str] = []
    generations: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        prompt_seed = seed + index
        torch.manual_seed(prompt_seed)
        torch.cuda.manual_seed_all(prompt_seed)
        rendered = _render_prompt(tokenizer, prompt, True, False)
        completions, usage = _generate_completions(
            model,
            tokenizer,
            [rendered],
            True,
            "greedy",
            max_new_tokens,
            _generation_stop_token_ids(tokenizer),
        )
        if len(completions) != 1 or len(usage) != 1:
            raise IDRInteractionError("interactive generation cardinality differs")
        token_count, exhausted = usage[0]
        outputs.append(completions[0])
        generations.append(
            {
                "generated_tokens": token_count,
                "max_token_exhausted": exhausted,
                "seed": prompt_seed,
            }
        )
    metadata = {
        "model_loader": model_loader,
        "adapter_metadata": adapter_metadata,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outputs, generations, metadata


def load_question_rows(path: Path) -> list[dict[str, str]]:
    """Load a small, explicit interaction set without accepting silent defaults."""
    if not path.is_file():
        raise IDRInteractionError(f"question file does not exist: {path}")
    rows: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IDRInteractionError(
                    f"invalid question JSON on line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise IDRInteractionError(f"question line {line_number} is not an object")
            row_id = str(row.get("id", "")).strip()
            question = str(row.get("question", "")).strip()
            response_mode = str(row.get("response_mode", "")).strip()
            if not row_id or row_id in seen_ids:
                raise IDRInteractionError(
                    f"question line {line_number} has missing or duplicate id"
                )
            if not question:
                raise IDRInteractionError(f"question line {line_number} is empty")
            if response_mode not in {"general", "math", "code"}:
                raise IDRInteractionError(
                    f"question line {line_number} has unsupported response mode"
                )
            seen_ids.add(row_id)
            rows.append(
                {"id": row_id, "question": question, "response_mode": response_mode}
            )
    if not rows:
        raise IDRInteractionError("question file has no records")
    return rows


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    for path in (args.model_root, args.draft_checkpoint, args.revision_checkpoint):
        if not path.exists():
            raise IDRInteractionError(f"required input does not exist: {path}")
    if args.report is None:
        raise IDRInteractionError("batch interaction requires --report")
    if args.report.exists():
        raise IDRInteractionError("refusing to replace interaction report")
    rows = load_question_rows(args.questions_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    drafts, draft_usage, draft_metadata = generate_many(
        args.model_root,
        args.draft_checkpoint,
        tokenizer,
        [row["question"] for row in rows],
        args.max_new_tokens,
        args.seed,
    )
    revision_prompts = [
        revision_prompt(row["question"], draft, row["response_mode"])
        for row, draft in zip(rows, drafts, strict=True)
    ]
    revisions, revision_usage, revision_metadata = generate_many(
        args.model_root,
        args.revision_checkpoint,
        tokenizer,
        revision_prompts,
        args.max_new_tokens,
        args.seed + len(rows),
    )
    report = {
        "schema": "shohin-idr-interaction-batch-v1",
        "status": "complete",
        "model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "draft_checkpoint": str(args.draft_checkpoint.resolve()),
        "draft_checkpoint_sha256": sha256_file(args.draft_checkpoint),
        "revision_checkpoint": str(args.revision_checkpoint.resolve()),
        "revision_checkpoint_sha256": sha256_file(args.revision_checkpoint),
        "questions": str(args.questions_jsonl.resolve()),
        "questions_sha256": sha256_file(args.questions_jsonl),
        "draft_model": draft_metadata,
        "revision_model": revision_metadata,
        "interactions": [
            {
                **row,
                "draft": draft,
                "revision": revision,
                "draft_generation": draft_generation,
                "revision_generation": revision_generation,
            }
            for row, draft, revision, draft_generation, revision_generation in zip(
                rows, drafts, revisions, draft_usage, revision_usage, strict=True
            )
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(f".{args.report.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    for path in (args.model_root, args.draft_checkpoint, args.revision_checkpoint):
        if not path.exists():
            raise IDRInteractionError(f"required input does not exist: {path}")
    question = args.question.strip()
    if not question:
        raise IDRInteractionError("question must be nonempty")
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    draft, draft_usage, draft_metadata = generate_one(
        args.model_root,
        args.draft_checkpoint,
        tokenizer,
        question,
        args.max_new_tokens,
        args.seed,
    )
    revised, revision_usage, revision_metadata = generate_one(
        args.model_root,
        args.revision_checkpoint,
        tokenizer,
        revision_prompt(question, draft, args.response_mode),
        args.max_new_tokens,
        args.seed + 1,
    )
    report = {
        "schema": "shohin-idr-interaction-v1",
        "status": "complete",
        "model_root": str(args.model_root.resolve()),
        "model_revision": args.model_revision,
        "draft_checkpoint": str(args.draft_checkpoint.resolve()),
        "draft_checkpoint_sha256": sha256_file(args.draft_checkpoint),
        "revision_checkpoint": str(args.revision_checkpoint.resolve()),
        "revision_checkpoint_sha256": sha256_file(args.revision_checkpoint),
        "response_mode": args.response_mode,
        "question": question,
        "draft": draft,
        "revision": revised,
        "draft_generation": draft_usage,
        "revision_generation": revision_usage,
        "draft_model": draft_metadata,
        "revision_model": revision_metadata,
    }
    if args.report is not None:
        if args.report.exists():
            raise IDRInteractionError("refusing to replace interaction report")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(args.report.suffix + ".partial")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--draft-checkpoint", type=Path, required=True)
    parser.add_argument("--revision-checkpoint", type=Path, required=True)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--question")
    prompt_group.add_argument("--questions-jsonl", type=Path)
    parser.add_argument("--response-mode", choices=("general", "math", "code"), default="general")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=2026080819)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.questions_jsonl is not None:
        report = run_batch(args)
        print(f"interactions={len(report['interactions'])}")
        print(f"report={args.report}")
        return 0
    report = run(args)
    print("=== INTERNAL DRAFT ===")
    print(report["draft"])
    print("\n=== REVISED ANSWER ===")
    print(report["revision"])
    if args.report is not None:
        print(f"\nreport={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
