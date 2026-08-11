#!/usr/bin/env python3
"""Build source-local KEEP/CONTINUE/RESTART episodes from NDR1 custody."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from build_ndr1_natural_revision_data import (
    DRAFT_REPORT_SCHEMA,
    DRAFT_SCHEMA,
    SOURCE_REPORT_SCHEMA,
    load_lines,
    sha256_file,
    source_identity,
)
from hf_product_reasoning_eval import (
    extract_boxed,
    extract_short_answer,
    has_explicit_final_answer,
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


SCHEMA = "shohin-kcr1-branch-train-v1"
REPORT_SCHEMA = "shohin-kcr1-branch-data-report-v1"


class KCR1DataError(RuntimeError):
    """KCR1 source, draft, transaction, or token custody differs."""


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise KCR1DataError(f"refusing existing output: {path}")
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


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise KCR1DataError(f"refusing existing report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def continuation_split(response: str) -> tuple[str, str]:
    """Choose a deterministic semantic boundary while preserving exact bytes."""

    if len(response) < 8:
        raise KCR1DataError("verified response is too short to split")
    target = int(len(response) * 0.6)
    candidates = {
        match.end()
        for match in re.finditer(r"(?:\n+|(?<=[.!?])\s+)", response)
        if 0.4 <= match.end() / len(response) <= 0.8
    }
    if not candidates:
        candidates = {
            match.end()
            for match in re.finditer(r"\s+", response)
            if 0.4 <= match.end() / len(response) <= 0.8
        }
    if not candidates:
        raise KCR1DataError("verified response has no safe continuation boundary")
    cut = min(candidates, key=lambda value: (abs(value - target), value))
    prefix, suffix = response[:cut], response[cut:]
    if not prefix or not suffix or prefix + suffix != response:
        raise KCR1DataError("continuation split is not byte exact")
    return prefix, suffix


def conservative_natural_correct(
    source: dict[str, Any], draft: str, exhausted: bool
) -> bool:
    """Certify only explicit, non-exhausted non-code final answers."""

    if exhausted or not has_explicit_final_answer(draft):
        return False
    group = str(source.get("training_group", ""))
    if group == "code":
        return False
    if group in {"math", "science"}:
        gold = source.get("expected_answer_normalized")
        return isinstance(gold, str) and match_math(extract_boxed(draft), gold)
    if group == "procedural":
        gold = source.get("answer")
        return isinstance(gold, str) and match_short_answer(
            extract_short_answer(draft), gold
        )
    raise KCR1DataError(f"unsupported training group: {group}")


def load_merged(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    source_sha256 = sha256_file(args.source)
    if (
        source_report.get("schema") != SOURCE_REPORT_SCHEMA
        or source_report.get("status") != "complete"
        or source_report.get("output_sha256") != source_sha256
        or int(source_report.get("max_sequence_length", -1)) != 1536
    ):
        raise KCR1DataError("KCR1 source report differs")
    sources = load_lines(args.source)
    identities = [source_identity(row) for row in sources]
    if len(identities) != len(set(identities)):
        raise KCR1DataError("KCR1 source identity is duplicated")

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.draft_report]
    shard_count = len(reports)
    if shard_count < 2 or {report.get("shard_index") for report in reports} != set(
        range(shard_count)
    ):
        raise KCR1DataError("KCR1 draft shard coverage differs")
    draft_by_index: dict[int, dict[str, Any]] = {}
    for report_path, report in zip(args.draft_report, reports, strict=True):
        if (
            report.get("schema") != DRAFT_REPORT_SCHEMA
            or report.get("status") != "complete"
            or report.get("source_sha256") != source_sha256
            or report.get("shard_count") != shard_count
            or report.get("adapter_checkpoint_sha256")
            != args.adapter_checkpoint_sha256
            or report.get("max_new_tokens") != 768
            or report.get("seed") != 2026080919
        ):
            raise KCR1DataError(f"KCR1 draft report differs: {report_path}")
        draft_path = Path(str(report.get("output", "")))
        if report.get("output_sha256") != sha256_file(draft_path):
            raise KCR1DataError("KCR1 draft output hash differs")
        for draft in load_lines(draft_path):
            index = int(draft.get("source_index", -1))
            if draft.get("schema") != DRAFT_SCHEMA or index in draft_by_index:
                raise KCR1DataError("KCR1 draft schema/index differs")
            draft_by_index[index] = draft
    if set(draft_by_index) != set(range(len(sources))):
        raise KCR1DataError("KCR1 merged draft coverage differs")
    for index, (source, identity) in enumerate(zip(sources, identities, strict=True)):
        draft = draft_by_index[index]
        if (
            draft.get("source_identity_sha256") != identity
            or draft.get("training_group") != source.get("training_group")
        ):
            raise KCR1DataError("KCR1 source/draft binding differs")
    return sources, [draft_by_index[index] for index in range(len(sources))]


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise KCR1DataError(f"refusing existing output root: {args.output}")
    sources, drafts = load_merged(args)

    from transformers import AutoTokenizer
    from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    prompt_token_max = target_token_max = charged_target_tokens = 0
    admitted_sources = 0
    for source, natural in zip(sources, drafts, strict=True):
        identity = source_identity(source)
        response = str(source.get("response", ""))
        natural_draft = str(natural.get("completion", ""))
        if not response or not natural_draft:
            raise KCR1DataError("KCR1 response or natural draft is empty")
        try:
            prefix, suffix = continuation_split(response)
        except KCR1DataError:
            counters["unsplittable_source"] += 1
            continue
        natural_exhausted = natural.get("max_token_exhausted") is True
        natural_correct = conservative_natural_correct(
            source, natural_draft, natural_exhausted
        )
        natural_action = KEEP if natural_correct else RESTART
        natural_payload = "" if natural_correct else response
        presentations = (
            ("verified_keep", response, False, KEEP, ""),
            ("verified_continue", prefix, True, CONTINUE, suffix),
            (
                "natural_owner",
                natural_draft,
                natural_exhausted,
                natural_action,
                natural_payload,
            ),
        )
        staged: list[tuple[dict[str, Any], int, int]] = []
        for presentation, draft, exhausted, action, payload in presentations:
            transaction = render_transaction(action, payload)
            executed = execute_transaction(draft, transaction)
            expected_execution = draft if action == KEEP else response
            if executed != expected_execution:
                raise KCR1DataError("KCR1 transaction does not reconstruct target")
            prompt = kcr1_prompt(
                str(source["question"]),
                draft,
                exhausted=exhausted,
                task=str(source["training_group"]),
            )
            rendered = render_reasoning_messages(
                tokenizer,
                [
                    {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                enable_thinking=False,
            )
            prompt_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
            target_tokens = len(tokenizer.encode(transaction, add_special_tokens=False)) + 1
            if prompt_tokens + target_tokens > args.max_sequence_length:
                counters[f"overflow_{presentation}"] += 1
                staged = []
                break
            staged.append(
                (
                    {
                        "schema": SCHEMA,
                        "identity_sha256": hashlib.sha256(
                            f"kcr1\0{identity}\0{presentation}".encode()
                        ).hexdigest(),
                        "source_identity_sha256": identity,
                        "training_group": source["training_group"],
                        "presentation": presentation,
                        "question": prompt,
                        "response": transaction,
                        "action": action,
                        "executed_target_sha256": hashlib.sha256(
                            expected_execution.encode()
                        ).hexdigest(),
                        "draft_max_token_exhausted": exhausted,
                        "natural_draft_verified": (
                            natural_correct if presentation == "natural_owner" else None
                        ),
                        "runtime_fields": ["question"],
                        "assessor_fields_visible_to_model": False,
                    },
                    prompt_tokens,
                    target_tokens,
                )
            )
        if len(staged) != 3:
            continue
        admitted_sources += 1
        for row, prompt_tokens, target_tokens in staged:
            rows.append(row)
            counters[f"action_{row['action']}"] += 1
            counters[f"presentation_{row['presentation']}"] += 1
            if row["presentation"] == "natural_owner":
                counters[f"natural_action_{row['action']}"] += 1
                counters[
                    "natural_exhausted" if row["draft_max_token_exhausted"] else "natural_stopped"
                ] += 1
            prompt_token_max = max(prompt_token_max, prompt_tokens)
            target_token_max = max(target_token_max, target_tokens)
            charged_target_tokens += target_tokens

    if admitted_sources < int(0.9 * len(sources)) or len(rows) != 3 * admitted_sources:
        raise KCR1DataError("KCR1 three-state admission gate differs")
    if len({row["identity_sha256"] for row in rows}) != len(rows):
        raise KCR1DataError("KCR1 presentation identity is duplicated")

    args.output.mkdir(parents=True)
    train_path = args.output / "train.jsonl"
    train_sha256 = atomic_lines(train_path, rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "source_report_sha256": sha256_file(args.source_report),
        "adapter_checkpoint_sha256": args.adapter_checkpoint_sha256,
        "draft_reports": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.draft_report
        ],
        "source_rows": len(sources),
        "admitted_sources": admitted_sources,
        "presentations": len(rows),
        "presentations_per_source": 3,
        "transaction_roundtrip_rows": len(rows),
        "scan_counters": dict(counters),
        "charged_target_tokens": charged_target_tokens,
        "prompt_token_max": prompt_token_max,
        "target_token_max": target_token_max,
        "max_sequence_length": args.max_sequence_length,
        "zero_truncation": True,
        "runtime_fields": ["question"],
        "assessor_fields_visible_to_model": False,
        "holdout_used": False,
        "output": {
            "path": str(train_path.resolve()),
            "sha256": train_sha256,
            "rows": len(rows),
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
