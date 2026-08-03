#!/usr/bin/env python3
"""Measure the effective prompt and charged-target token mix of a JSONL corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. Give concise, verifiable reasoning "
    "and a clearly marked final answer."
)


class TokenMixAuditError(RuntimeError):
    """The corpus or tokenizer contract cannot be audited exactly."""


def question_response(row: dict[str, Any]) -> tuple[str, str] | None:
    question = row.get("question") or row.get("problem") or row.get("prompt")
    response = (
        row.get("response")
        or row.get("solution")
        or row.get("completion")
        or row.get("answer")
    )
    if not question or not response:
        return None
    return str(question), str(response)


def truncate_lengths(
    prompt_length: int,
    response_length: int,
    *,
    max_sequence_length: int,
    workspace_slots: int,
) -> tuple[int, int, bool, bool]:
    """Apply the trainer's exact prompt/response budget and include target EOS."""

    target_budget = max_sequence_length - workspace_slots
    response_budget_floor = min(256, max_sequence_length // 2)
    original_response = response_length
    if response_length > target_budget - 9:
        response_length = target_budget - 9
    prompt_budget = target_budget - response_length
    if prompt_budget < 8:
        response_length = min(response_length, response_budget_floor)
        prompt_budget = target_budget - response_length
    kept_prompt = min(prompt_length, prompt_budget)
    charged_target = response_length + 1
    return (
        kept_prompt,
        charged_target,
        response_length < original_response,
        kept_prompt < prompt_length,
    )


def audit(
    *,
    data: Path,
    tokenizer: Any,
    model_revision: str,
    max_sequence_length: int,
    workspace_slots: int,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    counters: Counter[str] = Counter()
    group_metrics: dict[str, Counter[str]] = defaultdict(Counter)
    with data.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            try:
                row = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise TokenMixAuditError("corpus contains malformed JSONL") from exc
            pair = question_response(row)
            if pair is None:
                counters["schema_rejected"] += 1
                continue
            question, response = pair
            group = str(row.get("training_group") or row.get("domain") or "unknown")
            rendered = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_length = len(tokenizer.encode(rendered, add_special_tokens=False))
            response_length = len(tokenizer.encode(response, add_special_tokens=False))
            kept_prompt, charged_target, response_cut, prompt_cut = truncate_lengths(
                prompt_length,
                response_length,
                max_sequence_length=max_sequence_length,
                workspace_slots=workspace_slots,
            )
            metrics = group_metrics[group]
            metrics["rows"] += 1
            metrics["raw_prompt_tokens"] += prompt_length
            metrics["raw_response_tokens"] += response_length
            metrics["kept_prompt_tokens"] += kept_prompt
            metrics["charged_target_tokens"] += charged_target
            metrics["response_truncated_rows"] += int(response_cut)
            metrics["prompt_truncated_rows"] += int(prompt_cut)
            if not response_cut and not prompt_cut:
                metrics["fully_untruncated_rows"] += 1
                metrics["fully_untruncated_target_tokens"] += charged_target
            counters["valid_rows"] += 1

    if not counters["valid_rows"]:
        raise TokenMixAuditError("corpus has no valid rows")
    total_target = sum(
        metrics["charged_target_tokens"] for metrics in group_metrics.values()
    )
    groups = {}
    for group, metrics in sorted(group_metrics.items()):
        values = dict(sorted(metrics.items()))
        values["charged_target_fraction"] = (
            metrics["charged_target_tokens"] / total_target
        )
        groups[group] = values
    return {
        "schema": "shohin-product-reasoning-token-mix-audit-v1",
        "status": "complete",
        "data": str(data.resolve()),
        "data_sha256": digest.hexdigest(),
        "tokenizer_name_or_path": str(tokenizer.name_or_path),
        "model_revision": model_revision,
        "max_sequence_length": max_sequence_length,
        "workspace_slots": workspace_slots,
        "counters": dict(sorted(counters.items())),
        "total_charged_target_tokens": total_target,
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--workspace-slots", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise TokenMixAuditError("refusing to replace token-mix report")
    if args.max_sequence_length <= args.workspace_slots + 16:
        raise TokenMixAuditError("sequence budget is too small")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_root,
        revision=args.model_revision,
        trust_remote_code=True,
    )
    report = audit(
        data=args.data,
        tokenizer=tokenizer,
        model_revision=args.model_revision,
        max_sequence_length=args.max_sequence_length,
        workspace_slots=args.workspace_slots,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
