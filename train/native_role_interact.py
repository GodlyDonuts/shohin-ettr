#!/usr/bin/env python3
"""Run native Shohin draft-then-revision interactions from one shared trunk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from tokenizers import Tokenizer

from eval_suite import generate
from model import GPT, GPTConfig
from native_role_lora import (
    NativeRoleLoRAConfig,
    attach_role_lora,
    load_role_adapter,
    set_active_role,
)
from native_role_sft import atomic_json, load_adapter_payload, sha256_file, verify_sha256


INTERACTION_REPORT_SCHEMA = "shohin-native-role-interaction-v1"


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, raw in enumerate(source, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"prompt row {line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError("prompt board is empty")
    return rows


def render_draft_prompt(row: dict) -> str:
    explicit = row.get("draft_prompt")
    if explicit:
        return str(explicit)
    question = row.get("question") or row.get("prompt")
    if not question:
        raise ValueError("prompt row lacks question, prompt, or draft_prompt")
    return f"Question: {question}\nDraft answer:"


def render_revision_prompt(row: dict, draft: str) -> str:
    explicit = row.get("revision_prompt")
    if explicit:
        template = str(explicit)
        if template.count("{draft}") != 1:
            raise ValueError("explicit revision_prompt must contain one {draft}")
        return template.replace("{draft}", draft)
    question = row.get("question") or row.get("prompt")
    if not question:
        raise ValueError("default revision prompt lacks a question")
    return (
        f"Question: {question}\n"
        f"Candidate draft:\n{draft}\n"
        "Review the candidate, correct any error, and give the final answer.\n"
        "Final answer:"
    )


def bounded_generate(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    device: str,
    *,
    max_new: int,
    temperature: float,
) -> tuple[str, int]:
    input_tokens = len(tokenizer.encode(prompt).ids)
    available = model.cfg.seq_len - input_tokens
    if available <= 0:
        raise ValueError(
            f"interaction prompt consumes {input_tokens} tokens, exceeding context "
            f"{model.cfg.seq_len}"
        )
    generated = generate(
        model,
        tokenizer,
        prompt,
        device,
        max_new=min(max_new, available),
        temp=temperature,
    )
    return generated, input_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True)
    parser.add_argument("--draft-adapter", required=True)
    parser.add_argument("--revision-adapter", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-draft", type=int, default=512)
    parser.add_argument("--max-new-revision", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--expected-init-sha256", default="")
    parser.add_argument("--expected-draft-adapter-sha256", default="")
    parser.add_argument("--expected-revision-adapter-sha256", default="")
    parser.add_argument("--expected-tokenizer-sha256", default="")
    parser.add_argument("--expected-prompts-sha256", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_draft <= 0 or args.max_new_revision <= 0:
        raise SystemExit("generation budgets must be positive")
    if args.temperature < 0:
        raise SystemExit("temperature cannot be negative")
    output = Path(args.out)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"interaction report already exists: {output}")

    bindings = {
        "init": {
            "path": args.init,
            "sha256": verify_sha256(
                args.init, args.expected_init_sha256, "init"
            ),
        },
        "draft_adapter": {
            "path": args.draft_adapter,
            "sha256": verify_sha256(
                args.draft_adapter,
                args.expected_draft_adapter_sha256,
                "draft adapter",
            ),
        },
        "revision_adapter": {
            "path": args.revision_adapter,
            "sha256": verify_sha256(
                args.revision_adapter,
                args.expected_revision_adapter_sha256,
                "revision adapter",
            ),
        },
        "tokenizer": {
            "path": args.tokenizer,
            "sha256": verify_sha256(
                args.tokenizer, args.expected_tokenizer_sha256, "tokenizer"
            ),
        },
        "prompts": {
            "path": args.prompts,
            "sha256": verify_sha256(
                args.prompts, args.expected_prompts_sha256, "prompts"
            ),
        },
    }

    checkpoint = torch.load(args.init, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("cfg"), dict):
        raise ValueError("init checkpoint lacks an exact model config")
    draft_payload = load_adapter_payload(args.draft_adapter)
    revision_payload = load_adapter_payload(args.revision_adapter)
    draft_config = NativeRoleLoRAConfig(**draft_payload.get("adapter_config", {}))
    revision_config = NativeRoleLoRAConfig(
        **revision_payload.get("adapter_config", {})
    )
    if draft_config != revision_config:
        raise ValueError("draft and revision adapter configs differ")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = GPT(GPTConfig(**checkpoint["cfg"])).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    attach_role_lora(model, draft_config)
    load_role_adapter(
        model,
        draft_payload,
        draft_config,
        expected_role="draft",
        expected_base_checkpoint_sha256=bindings["init"]["sha256"],
    )
    load_role_adapter(
        model,
        revision_payload,
        revision_config,
        expected_role="revision",
        expected_base_checkpoint_sha256=bindings["init"]["sha256"],
    )
    tokenizer = Tokenizer.from_file(args.tokenizer)
    prompts = load_jsonl(args.prompts)
    started = time.time()
    transcripts = []
    with torch.inference_mode():
        for index, row in enumerate(prompts):
            draft_prompt = render_draft_prompt(row)
            set_active_role(model, "draft")
            draft, draft_input_tokens = bounded_generate(
                model,
                tokenizer,
                draft_prompt,
                device,
                max_new=args.max_new_draft,
                temperature=args.temperature,
            )
            revision_prompt = render_revision_prompt(row, draft)
            set_active_role(model, "revision")
            revision, revision_input_tokens = bounded_generate(
                model,
                tokenizer,
                revision_prompt,
                device,
                max_new=args.max_new_revision,
                temperature=args.temperature,
            )
            record = {
                "index": index,
                "id": row.get("id", f"row-{index:04d}"),
                "question": row.get("question") or row.get("prompt"),
                "draft_prompt": draft_prompt,
                "draft": draft,
                "revision_prompt": revision_prompt,
                "revision": revision,
                "draft_input_tokens": draft_input_tokens,
                "revision_input_tokens": revision_input_tokens,
            }
            transcripts.append(record)
            print(
                f"\n===== {record['id']} =====\nDRAFT:\n{draft}\nREVISION:\n{revision}",
                flush=True,
            )

    report = {
        "schema": INTERACTION_REPORT_SCHEMA,
        "status": "complete",
        "bindings": bindings,
        "base_step": checkpoint.get("step"),
        "adapter_config": draft_payload["adapter_config"],
        "generation": {
            "max_new_draft": args.max_new_draft,
            "max_new_revision": args.max_new_revision,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "runtime": {
            "device": device,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "duration_seconds": time.time() - started,
        },
        "transcripts": transcripts,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    print(f"[native-role-interact] wrote {output} sha256={sha256_file(output)}")


if __name__ == "__main__":
    main()
