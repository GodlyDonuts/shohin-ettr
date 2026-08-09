#!/usr/bin/env python3
"""Merge natural B1 drafts into matched aligned/shuffled NDR1 curricula."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ttr1_revision import internal_revision_prompt


SCHEMA = "shohin-ndr1-natural-revision-train-v1"
REPORT_SCHEMA = "shohin-ndr1-natural-revision-data-report-v1"
DRAFT_SCHEMA = "shohin-ndr1-natural-drafts-v1"
DRAFT_REPORT_SCHEMA = "shohin-ndr1-natural-draft-report-v1"
SOURCE_REPORT_SCHEMA = "shohin-token-balanced-reasoning-mix-v1"


class NDR1DataError(RuntimeError):
    """NDR1 source, draft, matching, or sequence custody differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(row: dict[str, Any]) -> str:
    question = str(row.get("question", "")).strip()
    if not question:
        raise NDR1DataError("NDR1 source question is empty")
    return hashlib.sha256(" ".join(question.casefold().split()).encode()).hexdigest()


def load_lines(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise NDR1DataError(f"empty NDR1 input: {path}")
    return rows


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists() or path.is_symlink():
        raise NDR1DataError(f"refusing existing output: {path}")
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
        raise NDR1DataError(f"refusing existing report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def donor_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["training_group"])].append(row)
    donors: dict[str, str] = {}
    for group, members in by_group.items():
        if len(members) < 2:
            raise NDR1DataError(f"NDR1 donor group is singleton: {group}")
        ordered = sorted(
            members,
            key=lambda row: (len(str(row["draft"])), row["source_identity_sha256"]),
        )
        for index, row in enumerate(ordered):
            donor = ordered[(index + 1) % len(ordered)]
            donors[row["source_identity_sha256"]] = donor["source_identity_sha256"]
    return donors


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise NDR1DataError(f"refusing existing output root: {args.output}")
    source_report = json.loads(args.source_report.read_text())
    source_sha256 = sha256_file(args.source)
    if (
        source_report.get("schema") != SOURCE_REPORT_SCHEMA
        or source_report.get("status") != "complete"
        or source_report.get("output_sha256") != source_sha256
        or int(source_report.get("max_sequence_length", -1)) != 1536
    ):
        raise NDR1DataError("NDR1 source report differs")
    sources = load_lines(args.source)
    source_ids = [source_identity(row) for row in sources]
    if len(source_ids) != len(set(source_ids)):
        raise NDR1DataError("NDR1 source identity is duplicated")

    reports = [json.loads(path.read_text()) for path in args.draft_report]
    shard_count = len(reports)
    if shard_count < 2 or {report.get("shard_index") for report in reports} != set(
        range(shard_count)
    ):
        raise NDR1DataError("NDR1 draft shard coverage differs")
    draft_by_index: dict[int, dict[str, Any]] = {}
    for path, report in zip(args.draft_report, reports, strict=True):
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
            raise NDR1DataError(f"NDR1 draft report differs: {path}")
        draft_path = Path(str(report.get("output", "")))
        if report.get("output_sha256") != sha256_file(draft_path):
            raise NDR1DataError("NDR1 draft output hash differs")
        for draft in load_lines(draft_path):
            index = int(draft.get("source_index", -1))
            if draft.get("schema") != DRAFT_SCHEMA or index in draft_by_index:
                raise NDR1DataError("NDR1 draft schema/index differs")
            draft_by_index[index] = draft
    if set(draft_by_index) != set(range(len(sources))):
        raise NDR1DataError("NDR1 merged draft coverage differs")

    merged: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        draft = draft_by_index[index]
        identity = source_ids[index]
        if (
            draft.get("source_identity_sha256") != identity
            or draft.get("training_group") != source.get("training_group")
        ):
            raise NDR1DataError("NDR1 draft/source binding differs")
        merged.append(
            {
                "source_identity_sha256": identity,
                "training_group": source["training_group"],
                "question": source["question"],
                "response": source["response"],
                "draft": draft["completion"],
                "draft_generated_tokens": draft["generated_tokens"],
                "draft_max_token_exhausted": draft["max_token_exhausted"],
            }
        )

    from transformers import AutoTokenizer
    from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    donors = donor_map(merged)
    by_identity = {row["source_identity_sha256"]: row for row in merged}
    aligned_rows: list[dict[str, Any]] = []
    shuffled_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    target_tokens = prompt_token_max = target_token_max = 0
    donor_char_deltas: list[int] = []
    for row in merged:
        donor = by_identity[donors[row["source_identity_sha256"]]]
        task = "mbpp" if row["training_group"] == "code" else row["training_group"]
        aligned_prompt = internal_revision_prompt(row["question"], row["draft"], task)
        shuffled_prompt = internal_revision_prompt(row["question"], donor["draft"], task)
        prompt_lengths = []
        for prompt in (aligned_prompt, shuffled_prompt):
            rendered = render_reasoning_messages(
                tokenizer,
                [
                    {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                enable_thinking=False,
            )
            prompt_lengths.append(len(tokenizer.encode(rendered, add_special_tokens=False)))
        response_tokens = len(tokenizer.encode(row["response"], add_special_tokens=False)) + 1
        if any(length + response_tokens > 4096 for length in prompt_lengths):
            counters["sequence_overflow_rejected"] += 1
            continue
        common = {
            "schema": SCHEMA,
            "source_identity_sha256": row["source_identity_sha256"],
            "training_group": row["training_group"],
            "response": row["response"],
            "target_kind": "verified_full_solution",
            "internal_draft_visible": True,
            "external_candidate_text_visible": False,
            "draft_max_token_exhausted": row["draft_max_token_exhausted"],
        }
        aligned_rows.append(
            {
                **common,
                "identity_sha256": hashlib.sha256(
                    f"ndr1-aligned\0{row['source_identity_sha256']}".encode()
                ).hexdigest(),
                "question": aligned_prompt,
                "draft_control": "aligned_natural_b1",
            }
        )
        shuffled_rows.append(
            {
                **common,
                "identity_sha256": hashlib.sha256(
                    f"ndr1-shuffled\0{row['source_identity_sha256']}".encode()
                ).hexdigest(),
                "question": shuffled_prompt,
                "draft_control": "within_domain_near_length_shuffle",
                "draft_donor_identity_sha256": donor["source_identity_sha256"],
            }
        )
        target_tokens += response_tokens
        target_token_max = max(target_token_max, response_tokens)
        prompt_token_max = max(prompt_token_max, *prompt_lengths)
        donor_char_deltas.append(abs(len(row["draft"]) - len(donor["draft"])))

    if (
        len(aligned_rows) < int(0.9 * len(sources))
        or len(aligned_rows) != len(shuffled_rows)
        or [row["response"] for row in aligned_rows]
        != [row["response"] for row in shuffled_rows]
        or any(
            aligned["source_identity_sha256"]
            == shuffled["draft_donor_identity_sha256"]
            for aligned, shuffled in zip(aligned_rows, shuffled_rows, strict=True)
        )
    ):
        raise NDR1DataError("NDR1 matched admission gate differs")

    args.output.mkdir(parents=True)
    paths = {
        "aligned": args.output / "train_aligned.jsonl",
        "shuffled": args.output / "train_shuffled.jsonl",
    }
    hashes = {
        "aligned": atomic_lines(paths["aligned"], aligned_rows),
        "shuffled": atomic_lines(paths["shuffled"], shuffled_rows),
    }
    deltas = sorted(donor_char_deltas)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "source": str(args.source.resolve()),
        "source_sha256": source_sha256,
        "source_report_sha256": sha256_file(args.source_report),
        "adapter_checkpoint_sha256": args.adapter_checkpoint_sha256,
        "draft_reports": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.draft_report
        ],
        "source_rows": len(sources),
        "admitted_rows_per_arm": len(aligned_rows),
        "charged_target_tokens_per_arm": target_tokens,
        "target_multiset_exactly_matched": True,
        "zero_source_donor_identity_matches": True,
        "natural_drafts_only": True,
        "synthetic_faults_used": False,
        "clean_copy_presentations_used": False,
        "draft_exhausted_rows_per_arm": sum(
            int(row["draft_max_token_exhausted"]) for row in aligned_rows
        ),
        "group_counts_per_arm": dict(Counter(row["training_group"] for row in aligned_rows)),
        "scan_counters": dict(counters),
        "prompt_token_max": prompt_token_max,
        "target_token_max": target_token_max,
        "donor_character_delta_p95": deltas[min(len(deltas) - 1, int(0.95 * len(deltas)))],
        "outputs": {
            arm: {"path": str(path.resolve()), "sha256": hashes[arm], "rows": len(aligned_rows)}
            for arm, path in paths.items()
        },
        "max_sequence_length": 4096,
        "holdout_used": False,
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
    parser.add_argument("--output", type=Path, required=True)
    report = build(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
