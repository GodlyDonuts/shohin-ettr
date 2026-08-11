#!/usr/bin/env python3
"""Build VTE1 sets of semantically equivalent executable transactions."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from build_kcr1_branch_data import (
    KCR1DataError,
    atomic_json,
    atomic_lines,
    conservative_natural_correct,
    continuation_split,
    load_merged,
    render_prompt,
)
from build_ndr1_natural_revision_data import source_identity
from hf_product_reasoning_eval import (
    extract_boxed,
    extract_short_answer,
    match_math,
    match_short_answer,
)
from kcr1_branch_transducer import (
    CONTINUE,
    KEEP,
    RESTART,
    execute_transaction,
    kcr1_prompt,
    render_transaction,
)


SCHEMA = "shohin-vte1-equivalence-train-v1"
REPORT_SCHEMA = "shohin-vte1-equivalence-data-report-v1"
CORRECTION_DELIMITER = "\n\nCorrection:\n"


class VTE1DataError(RuntimeError):
    """VTE1 source, candidate, assessor, or token custody differs."""


def semantic_correct(source: dict[str, Any], value: str) -> bool:
    group = str(source.get("training_group", ""))
    if group == "code":
        return value == str(source.get("response", ""))
    if group in {"math", "science"}:
        gold = source.get("expected_answer_normalized")
        return isinstance(gold, str) and match_math(extract_boxed(value), gold)
    if group == "procedural":
        gold = source.get("answer")
        return isinstance(gold, str) and match_short_answer(
            extract_short_answer(value), gold
        )
    raise VTE1DataError(f"unsupported training group: {group}")


def verified_equivalence_set(
    source: dict[str, Any], draft: str, verified: str
) -> list[str]:
    """Enumerate the frozen complete transaction equivalence set."""

    if not draft or not verified or not semantic_correct(source, verified):
        raise VTE1DataError("VTE1 draft or verified target differs")
    candidates: list[str] = []
    if draft == verified:
        candidates.append(render_transaction(KEEP))
    if verified.startswith(draft) and len(verified) > len(draft):
        candidates.append(render_transaction(CONTINUE, verified[len(draft) :]))
    candidates.append(render_transaction(RESTART, verified))
    if (
        source.get("training_group") != "code"
        and draft != verified
        and not verified.startswith(draft)
    ):
        candidates.append(render_transaction(CONTINUE, CORRECTION_DELIMITER + verified))

    unique = list(dict.fromkeys(candidates))
    if not unique:
        raise VTE1DataError("VTE1 equivalence set is empty")
    for transaction in unique:
        executed = execute_transaction(draft, transaction)
        if not semantic_correct(source, executed):
            raise VTE1DataError("VTE1 candidate fails independent semantics")
    return unique


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise VTE1DataError(f"refusing existing output root: {args.output}")
    sources, drafts = load_merged(args)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    prompt_max = candidate_max = sequence_max = candidate_tokens = 0
    admitted_sources = 0
    for source, natural in zip(sources, drafts, strict=True):
        identity = source_identity(source)
        verified = str(source.get("response", ""))
        natural_draft = str(natural.get("completion", ""))
        if not verified or not natural_draft:
            raise VTE1DataError("VTE1 verified response or natural draft is empty")
        try:
            prefix, _ = continuation_split(verified)
        except KCR1DataError:
            counters["unsplittable_source"] += 1
            continue
        exhausted = natural.get("max_token_exhausted") is True
        natural_correct = conservative_natural_correct(source, natural_draft, exhausted)
        presentations = (
            ("verified_keep", verified, False),
            ("verified_continue", prefix, True),
            ("natural_owner", natural_draft, exhausted),
        )
        staged: list[tuple[dict[str, Any], int, list[int], list[str]]] = []
        for presentation, draft, draft_exhausted in presentations:
            transactions = verified_equivalence_set(source, draft, verified)
            prompt = kcr1_prompt(
                str(source["question"]),
                draft,
                exhausted=draft_exhausted,
                task=str(source["training_group"]),
            )
            rendered = render_prompt(tokenizer, prompt)
            prompt_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
            target_lengths = [
                len(tokenizer.encode(value, add_special_tokens=False)) + 1
                for value in transactions
            ]
            if any(
                prompt_tokens + target_tokens > args.max_sequence_length
                for target_tokens in target_lengths
            ):
                counters[f"overflow_{presentation}"] += 1
                staged = []
                break
            staged.append(
                (
                    {
                        "schema": SCHEMA,
                        "identity_sha256": hashlib.sha256(
                            f"vte1\0{identity}\0{presentation}".encode()
                        ).hexdigest(),
                        "source_identity_sha256": identity,
                        "training_group": source["training_group"],
                        "presentation": presentation,
                        "question": prompt,
                        "responses": transactions,
                        "candidate_count": len(transactions),
                        "draft_max_token_exhausted": draft_exhausted,
                        "natural_draft_verified": (
                            natural_correct if presentation == "natural_owner" else None
                        ),
                        "runtime_fields": ["question", "responses"],
                        "assessor_fields_visible_to_model": False,
                    },
                    prompt_tokens,
                    target_lengths,
                    transactions,
                )
            )
        if len(staged) != 3:
            continue
        admitted_sources += 1
        for row, prompt_tokens, target_lengths, transactions in staged:
            rows.append(row)
            prompt_max = max(prompt_max, prompt_tokens)
            candidate_max = max(candidate_max, *target_lengths)
            sequence_max = max(
                sequence_max, *(prompt_tokens + value for value in target_lengths)
            )
            candidate_tokens += sum(target_lengths)
            counters[f"presentation_{row['presentation']}"] += 1
            counters[f"candidate_count_{len(transactions)}"] += 1
            for transaction in transactions:
                action = transaction.partition("\n")[0]
                counters[f"candidate_action_{action}"] += 1

    if admitted_sources < int(0.9 * len(sources)) or len(rows) != 3 * admitted_sources:
        raise VTE1DataError("VTE1 three-state admission gate differs")
    if len({row["identity_sha256"] for row in rows}) != len(rows):
        raise VTE1DataError("VTE1 identity is duplicated")

    args.output.mkdir(parents=True)
    data_path = args.output / "train.jsonl"
    data_sha256 = atomic_lines(data_path, rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "source_rows": len(sources),
        "admitted_sources": admitted_sources,
        "presentations": len(rows),
        "equivalence_candidates": sum(row["candidate_count"] for row in rows),
        "candidate_tokens": candidate_tokens,
        "max_sequence_length": args.max_sequence_length,
        "token_maxima": {
            "prompt": prompt_max,
            "candidate": candidate_max,
            "sequence": sequence_max,
        },
        "zero_truncation": sequence_max <= args.max_sequence_length,
        "counters": dict(sorted(counters.items())),
        "holdout_used": False,
        "runtime_fields": ["question", "responses"],
        "assessor_fields_visible_to_model": False,
        "output": {
            "path": str(data_path.resolve()),
            "rows": len(rows),
            "sha256": data_sha256,
        },
    }
    atomic_json(args.output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--draft-report", type=Path, action="append", required=True)
    parser.add_argument("--adapter-checkpoint-sha256", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    report = build(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
