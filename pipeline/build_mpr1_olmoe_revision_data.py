#!/usr/bin/env python3
"""Build exact aligned/shuffled OLMoE-owned MPR1 revision curricula."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "shohin-mpr1-revision-train-v1"
REPORT_SCHEMA = "shohin-mpr1-revision-data-report-v1"
SOURCE_SCHEMA = "shohin-idr1-revision-data-report-v1"
DRAFT_MARKER = "\n\nInternal draft:\n"
TAIL_MARKERS = (
    "\n\nReturn a complete corrected solution with the exact final answer in "
    "\\boxed{}.\n\nOriginal problem:\n",
    "\n\nReturn only executable Python code, without Markdown fences."
    "\n\nOriginal problem:\n",
)


class MPR1DataError(RuntimeError):
    """MPR1 source binding, draft matching, or sequence custody differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_lines(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not rows:
        raise MPR1DataError(f"empty MPR1 input: {path}")
    return rows


def split_draft(prompt: str) -> tuple[str, str, str]:
    matches = [marker for marker in TAIL_MARKERS if prompt.count(marker) == 1]
    if prompt.count(DRAFT_MARKER) != 1 or len(matches) != 1:
        raise MPR1DataError("MPR1 revision prompt markers differ")
    prefix, remainder = prompt.split(DRAFT_MARKER, 1)
    marker = matches[0]
    draft, suffix = remainder.split(marker, 1)
    if not prefix.strip() or not draft.strip() or not suffix.strip():
        raise MPR1DataError("MPR1 source, draft, or repeated source is empty")
    return prefix + DRAFT_MARKER, draft, marker + suffix


def donor_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    result: dict[str, str] = {}
    for task, members in grouped.items():
        if len(members) < 2:
            raise MPR1DataError(f"singleton MPR1 task: {task}")
        ordered = sorted(members, key=lambda row: (row["draft_tokens"], row["source_id"]))
        for index, row in enumerate(ordered):
            candidates = []
            if index:
                candidates.append(ordered[index - 1])
            if index + 1 < len(ordered):
                candidates.append(ordered[index + 1])
            donor = min(
                candidates,
                key=lambda candidate: (
                    abs(candidate["draft_tokens"] - row["draft_tokens"]),
                    candidate["source_id"],
                ),
            )
            result[row["source_id"]] = donor["source_id"]
    return result


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
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
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise MPR1DataError(f"refusing existing output root: {args.output}")
    source_report = json.loads(args.source_report.read_text())
    source_hash = sha256_file(args.source)
    expected = source_report.get("outputs", {}).get("train", {})
    if (
        source_report.get("schema") != SOURCE_SCHEMA
        or source_report.get("status") != "complete"
        or expected.get("sha256") != source_hash
        or Path(str(expected.get("path", ""))).resolve() != args.source.resolve()
        or source_report.get("internal_draft_visible") is not True
    ):
        raise MPR1DataError("MPR1 source report differs")

    pair_rows = load_lines(args.pairs)
    task_by_source = {
        str(row["identity_sha256"]): str(row["task"])
        for row in pair_rows
    }
    source_rows = load_lines(args.source)

    from transformers import AutoTokenizer
    from hf_product_reasoning_train import PRODUCT_SYSTEM_PROMPT, render_reasoning_messages

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    parsed = []
    for row in source_rows:
        source_id = str(row.get("source_identity_sha256", ""))
        task = task_by_source.get(source_id)
        if not task:
            raise MPR1DataError("MPR1 source task is absent")
        prefix, draft, suffix = split_draft(str(row["question"]))
        parsed.append(
            {
                "row": row,
                "source_id": source_id,
                "task": task,
                "prefix": prefix,
                "draft": draft,
                "suffix": suffix,
                "draft_tokens": len(tokenizer.encode(draft, add_special_tokens=False)),
            }
        )
    by_source: dict[str, dict[str, Any]] = {}
    for row in parsed:
        existing = by_source.setdefault(row["source_id"], row)
        if (
            existing["task"] != row["task"]
            or existing["draft"] != row["draft"]
            or existing["prefix"] != row["prefix"]
            or existing["suffix"] != row["suffix"]
        ):
            raise MPR1DataError("MPR1 repeated source binding differs")
    donors = donor_map(list(by_source.values()))
    aligned_rows, shuffled_rows = [], []
    counters: Counter[str] = Counter()
    token_maxima = Counter()
    donor_deltas = []
    target_tokens = 0
    for parsed_row in parsed:
        row = parsed_row["row"]
        donor = by_source[donors[parsed_row["source_id"]]]
        aligned_prompt = str(row["question"])
        shuffled_prompt = parsed_row["prefix"] + donor["draft"] + parsed_row["suffix"]
        prompts = []
        for prompt in (aligned_prompt, shuffled_prompt):
            rendered = render_reasoning_messages(
                tokenizer,
                [
                    {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                enable_thinking=False,
            )
            prompts.append(len(tokenizer.encode(rendered, add_special_tokens=False)))
        response_tokens = len(tokenizer.encode(str(row["response"]), add_special_tokens=False)) + 1
        if any(length + response_tokens > 4096 for length in prompts):
            counters["matched_sequence_overflow_rejected"] += 1
            continue
        common = {
            key: value
            for key, value in row.items()
            if key not in {"identity_sha256", "question", "schema"}
        }
        common.update(
            {
                "schema": SCHEMA,
                "task": parsed_row["task"],
                "complete_source_retained": True,
                "complete_draft_retained": True,
                "complete_target_retained": True,
            }
        )
        presentation_id = str(row.get("identity_sha256", ""))
        if not presentation_id:
            raise MPR1DataError("MPR1 presentation identity is absent")
        aligned_rows.append(
            {
                **common,
                "identity_sha256": hashlib.sha256(
                    f"mpr1-aligned\0{presentation_id}".encode()
                ).hexdigest(),
                "question": aligned_prompt,
                "draft_control": "aligned_exact_olmoe",
            }
        )
        shuffled_rows.append(
            {
                **common,
                "identity_sha256": hashlib.sha256(
                    f"mpr1-shuffled\0{presentation_id}".encode()
                ).hexdigest(),
                "question": shuffled_prompt,
                "draft_control": "same_task_nearest_token_length",
                "draft_donor_identity_sha256": donor["source_id"],
            }
        )
        target_tokens += response_tokens
        token_maxima["prompt"] = max(token_maxima["prompt"], *prompts)
        token_maxima["target"] = max(token_maxima["target"], response_tokens)
        token_maxima["total"] = max(
            token_maxima["total"], max(prompts) + response_tokens
        )
        donor_deltas.append(abs(parsed_row["draft_tokens"] - donor["draft_tokens"]))

    if (
        len(aligned_rows) < int(0.95 * len(source_rows))
        or len(aligned_rows) != len(shuffled_rows)
        or [row["response"] for row in aligned_rows]
        != [row["response"] for row in shuffled_rows]
        or any(
            a["source_identity_sha256"] == s["draft_donor_identity_sha256"]
            for a, s in zip(aligned_rows, shuffled_rows, strict=True)
        )
    ):
        raise MPR1DataError("MPR1 matched admission differs")

    args.output.mkdir(parents=True)
    outputs = {}
    for arm, rows in (("aligned", aligned_rows), ("shuffled", shuffled_rows)):
        path = args.output / f"train_{arm}.jsonl"
        outputs[arm] = {
            "path": str(path.resolve()),
            "sha256": atomic_lines(path, rows),
            "rows": len(rows),
        }
    deltas = sorted(donor_deltas)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "model_root": str(args.model_root.resolve()),
        "source": str(args.source.resolve()),
        "source_sha256": source_hash,
        "source_report_sha256": sha256_file(args.source_report),
        "pairs": str(args.pairs.resolve()),
        "pairs_sha256": sha256_file(args.pairs),
        "source_rows": len(source_rows),
        "unique_source_identities": len(by_source),
        "admitted_rows_per_arm": len(aligned_rows),
        "rejected_rows": len(source_rows) - len(aligned_rows),
        "charged_target_tokens_per_arm": target_tokens,
        "target_multiset_exactly_matched": True,
        "zero_source_donor_identity_matches": True,
        "same_task_donors": True,
        "olmoe_owned_drafts": True,
        "max_sequence_length": 4096,
        "complete_retention": True,
        "holdout_used": False,
        "task_counts_per_arm": dict(Counter(row["task"] for row in aligned_rows)),
        "scan_counters": dict(counters),
        "maximum_tokens": dict(token_maxima),
        "donor_token_delta_p95": deltas[min(len(deltas) - 1, int(0.95 * len(deltas)))],
        "outputs": outputs,
    }
    atomic_json(args.output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(build(parser.parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
