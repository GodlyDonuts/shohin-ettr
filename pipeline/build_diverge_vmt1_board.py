#!/usr/bin/env python3
"""Build the balanced verified-trajectory matching board for DIVERGE-VMT1."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable


CANDIDATE_SCHEMA = "shohin-product-rollout-candidate-v1"
SCHEMA = "shohin-diverge-vmt1-board-v1"


class VMTBoardError(RuntimeError):
    """The candidate population cannot satisfy the frozen VMT1 contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise VMTBoardError(f"refusing existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            encoded = (json.dumps(row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise VMTBoardError(f"refusing existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_pairs(source: Path) -> tuple[list[dict[str, Any]], Counter[str]]:
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    counters: Counter[str] = Counter()
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            counters["candidate_rows"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VMTBoardError(
                    "candidate source contains malformed JSONL"
                ) from exc
            required = (
                "identity_sha256",
                "question",
                "completion",
                "correct",
                "sample_index",
                "training_group",
                "task",
            )
            if any(key not in row for key in required):
                raise VMTBoardError("candidate row schema differs")
            identity = str(row["identity_sha256"])
            sample_index = int(row["sample_index"])
            if sample_index not in (0, 1):
                continue
            if sample_index in grouped[identity]:
                raise VMTBoardError("candidate sample index repeats within a prompt")
            grouped[identity][sample_index] = row

    pairs: list[dict[str, Any]] = []
    for identity, candidates in grouped.items():
        counters["prompt_identities"] += 1
        if set(candidates) != {0, 1}:
            counters["missing_pair"] += 1
            continue
        first, second = candidates[0], candidates[1]
        invariant_fields = ("question", "training_group", "task")
        if any(first[field] != second[field] for field in invariant_fields):
            raise VMTBoardError("candidate invariant fields differ within a prompt")
        outcomes = (bool(first["correct"]), bool(second["correct"]))
        if outcomes[0] == outcomes[1]:
            counters["same_outcome"] += 1
            continue
        completions = (
            str(first.get("completion") or "").strip(),
            str(second.get("completion") or "").strip(),
        )
        if not all(completions):
            counters["empty_completion"] += 1
            continue
        if completions[0] == completions[1]:
            counters["duplicate_completion"] += 1
            continue
        if bool(first.get("max_token_exhausted")) or bool(
            second.get("max_token_exhausted")
        ):
            counters["exhausted_completion"] += 1
            continue
        group = str(first["training_group"])
        if group not in {"math", "science"}:
            counters["unsupported_group"] += 1
            continue
        pairs.append(
            {
                "schema": SCHEMA,
                "identity_sha256": identity,
                "question": str(first["question"]).strip(),
                "responses": list(completions),
                "correct": list(outcomes),
                "correct_index": 0 if outcomes[0] else 1,
                "sample_indices": [0, 1],
                "task": str(first["task"]),
                "training_group": group,
                "predictions": [first.get("prediction"), second.get("prediction")],
                "source_row_seeds": [first.get("row_seed"), second.get("row_seed")],
            }
        )
        counters["structurally_admissible"] += 1
    return pairs, counters


def build_board(
    source: Path,
    output: Path,
    report_output: Path,
    *,
    tokenizer: Any,
    render_prompt: Callable[[Any, str], str],
    per_cell: int,
    max_sequence_length: int,
    workspace_slots: int,
    seed: int,
) -> dict[str, Any]:
    if not source.is_file():
        raise VMTBoardError(f"candidate source is missing: {source}")
    if per_cell <= 0 or max_sequence_length <= workspace_slots + 32:
        raise VMTBoardError("board dimensions differ")
    pairs, counters = _load_pairs(source)
    cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    token_lengths: list[int] = []
    for pair in pairs:
        prompt_text = render_prompt(tokenizer, pair["question"])
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        response_tokens = [
            tokenizer.encode(response, add_special_tokens=False)
            for response in pair["responses"]
        ]
        totals = [
            len(prompt_tokens) + len(response) + 1 + workspace_slots
            for response in response_tokens
        ]
        if max(totals) > max_sequence_length:
            counters["context_rejected"] += 1
            continue
        if min(map(len, response_tokens)) < 8:
            counters["short_response_rejected"] += 1
            continue
        pair["token_accounting"] = {
            "prompt_tokens": len(prompt_tokens),
            "response_tokens": list(map(len, response_tokens)),
            "maximum_total_tokens": max(totals),
            "workspace_slots": workspace_slots,
        }
        cells[(pair["training_group"], pair["correct_index"])].append(pair)
        token_lengths.append(max(totals))
        counters["tokenizer_admissible"] += 1

    required_cells = [
        (group, correct) for group in ("math", "science") for correct in (0, 1)
    ]
    selected: list[dict[str, Any]] = []
    cell_counts: dict[str, int] = {}
    for group, correct_index in required_cells:
        rows = sorted(
            cells[(group, correct_index)],
            key=lambda row: hashlib.sha256(
                f"{seed}\0{group}\0{correct_index}\0{row['identity_sha256']}".encode()
            ).hexdigest(),
        )
        if len(rows) < per_cell:
            raise VMTBoardError(
                f"cell {group}/{correct_index} has {len(rows)} rows below {per_cell}"
            )
        selected.extend(rows[:per_cell])
        cell_counts[f"{group}/correct_{correct_index}"] = per_cell
    selected.sort(
        key=lambda row: hashlib.sha256(
            f"{seed}\0output\0{row['identity_sha256']}".encode()
        ).hexdigest()
    )
    identities = [row["identity_sha256"] for row in selected]
    if len(set(identities)) != len(identities):
        raise VMTBoardError("selected board contains duplicate prompts")
    if any(sum(map(bool, row["correct"])) != 1 for row in selected):
        raise VMTBoardError("selected board does not have one correct trajectory")

    output_sha256 = _atomic_jsonl(output, selected)
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "seed": seed,
        "source": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "output": str(output.resolve()),
        "output_sha256": output_sha256,
        "rows": len(selected),
        "per_cell": per_cell,
        "cell_counts": cell_counts,
        "max_sequence_length": max_sequence_length,
        "workspace_slots": workspace_slots,
        "selected_maximum_total_tokens": max(
            row["token_accounting"]["maximum_total_tokens"] for row in selected
        ),
        "population_maximum_total_tokens": max(token_lengths),
        "counters": dict(sorted(counters.items())),
    }
    _atomic_json(report_output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--per-cell", type=int, default=4)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--workspace-slots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026080602)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from hf_product_reasoning_train import (
        PRODUCT_SYSTEM_PROMPT,
        render_reasoning_messages,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def render_prompt(current_tokenizer: Any, question: str) -> str:
        return render_reasoning_messages(
            current_tokenizer,
            [
                {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            enable_thinking=False,
        )

    report = build_board(
        args.source,
        args.output,
        args.report,
        tokenizer=tokenizer,
        render_prompt=render_prompt,
        per_cell=args.per_cell,
        max_sequence_length=args.max_sequence_length,
        workspace_slots=args.workspace_slots,
        seed=args.seed,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
