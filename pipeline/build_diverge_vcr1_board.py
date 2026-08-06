"""Build identity-disjoint, tokenizer-exact temporal-correction boards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from diverge_vcr1_data import tokenize_correction_example


PAIR_SCHEMA = "shohin-product-verifier-preference-pairs-v1"
BOARD_SCHEMA = "shohin-diverge-vcr1-pair-board-v1"
REPORT_SCHEMA = "shohin-diverge-vcr1-board-report-v1"


class VCR1BoardError(RuntimeError):
    """The temporal-correction board cannot satisfy its contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank(seed: int, label: str, identity: str) -> str:
    return hashlib.sha256(f"{seed}\0{label}\0{identity}".encode()).hexdigest()


def _normalized_question(value: str) -> str:
    return " ".join(value.casefold().split())


def _excluded_questions(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("question", "problem", "prompt", "text", "input"):
                if row.get(key):
                    excluded.add(_normalized_question(str(row[key])))
                    break
    return excluded


def _read_unique_pairs(path: Path) -> tuple[list[dict[str, Any]], int]:
    by_identity: dict[str, dict[str, Any]] = {}
    total = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            if row.get("schema") != PAIR_SCHEMA:
                raise VCR1BoardError("preference pair schema differs")
            required = (
                "identity_sha256",
                "pair_rank_sha256",
                "training_group",
                "question",
                "chosen",
                "rejected",
            )
            if any(not row.get(key) for key in required):
                raise VCR1BoardError("preference pair is incomplete")
            identity = str(row["identity_sha256"])
            if len(identity) != 64 or len(str(row["pair_rank_sha256"])) != 64:
                raise VCR1BoardError("preference identity hash differs")
            if row["chosen"] == row["rejected"]:
                raise VCR1BoardError("preference responses are identical")
            incumbent = by_identity.get(identity)
            if incumbent is None or str(row["pair_rank_sha256"]) < str(
                incumbent["pair_rank_sha256"]
            ):
                by_identity[identity] = row
    if not by_identity:
        raise VCR1BoardError("preference source is empty")
    return list(by_identity.values()), total


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise VCR1BoardError(f"refusing to replace board: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise VCR1BoardError(f"refusing to replace report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if _sha256_file(args.source) != args.source_sha256:
        raise VCR1BoardError("preference source hash differs")
    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if not getattr(tokenizer, "is_fast", False):
        raise VCR1BoardError("VCR1 requires exact fast-tokenizer offsets")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    pairs, source_rows = _read_unique_pairs(args.source)
    excluded = _excluded_questions(args.exclude_eval)
    admitted: dict[tuple[str, str], list[dict[str, Any]]] = {}
    rejected_overlap = 0
    rejected_length = 0
    max_positions = 0
    for row in pairs:
        question = str(row["question"]).strip()
        if _normalized_question(question) in excluded:
            rejected_overlap += 1
            continue
        identity = str(row["identity_sha256"])
        split_value = int(_rank(args.seed, "split", identity)[:16], 16) / 2**64
        split = "development" if split_value < args.development_fraction else "train"
        wrong = tokenize_correction_example(
            tokenizer,
            question,
            str(row["rejected"]),
            str(row["chosen"]),
            max_sequence_length=args.max_sequence_length,
            workspace_slots=args.workspace_slots,
        )
        correct = tokenize_correction_example(
            tokenizer,
            question,
            str(row["chosen"]),
            str(row["chosen"]),
            max_sequence_length=args.max_sequence_length,
            workspace_slots=args.workspace_slots,
        )
        if wrong is None or correct is None:
            rejected_length += 1
            continue
        max_positions = max(
            max_positions,
            len(wrong.prompt_ids) + args.workspace_slots + len(wrong.response_ids),
            len(correct.prompt_ids) + args.workspace_slots + len(correct.response_ids),
        )
        group = str(row["training_group"])
        if group not in {"math", "science"}:
            raise VCR1BoardError("preference training group differs")
        admitted.setdefault((split, group), []).append(
            {
                "schema": BOARD_SCHEMA,
                "split": split,
                "training_group": group,
                "identity_sha256": identity,
                "source_pair_rank_sha256": str(row["pair_rank_sha256"]),
                "question": question,
                "wrong_draft": str(row["rejected"]).strip(),
                "correct_draft": str(row["chosen"]).strip(),
                "target": str(row["chosen"]).strip(),
                "wrong_positions": len(wrong.prompt_ids)
                + args.workspace_slots
                + len(wrong.response_ids),
                "correct_positions": len(correct.prompt_ids)
                + args.workspace_slots
                + len(correct.response_ids),
                "target_tokens": len(wrong.response_ids),
            }
        )

    selected: dict[str, list[dict[str, Any]]] = {"train": [], "development": []}
    available: dict[str, int] = {}
    for split in ("train", "development"):
        cap = args.train_per_group if split == "train" else args.development_per_group
        minimum = (
            args.minimum_train_per_group
            if split == "train"
            else args.minimum_development_per_group
        )
        for group in ("math", "science"):
            rows = admitted.get((split, group), [])
            rows.sort(
                key=lambda row: _rank(
                    args.seed,
                    f"{split}-{group}",
                    str(row["identity_sha256"]),
                )
            )
            available[f"{split}_{group}"] = len(rows)
            if len(rows) < minimum:
                raise VCR1BoardError(
                    f"insufficient {split} {group} rows: {len(rows)} < {minimum}"
                )
            selected[split].extend(rows[:cap])
        selected[split].sort(
            key=lambda row: _rank(args.seed, split, str(row["identity_sha256"]))
        )

    train_ids = {str(row["identity_sha256"]) for row in selected["train"]}
    development_ids = {str(row["identity_sha256"]) for row in selected["development"]}
    if train_ids & development_ids:
        raise VCR1BoardError("identity crossed train/development split")
    _atomic_jsonl(args.train_output, selected["train"])
    _atomic_jsonl(args.development_output, selected["development"])
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "source": str(args.source.resolve()),
        "source_sha256": args.source_sha256,
        "source_rows": source_rows,
        "source_unique_identities": len(pairs),
        "excluded_eval_files": [str(path.resolve()) for path in args.exclude_eval],
        "excluded_eval_questions": len(excluded),
        "rejected_exact_eval_overlap": rejected_overlap,
        "rejected_length": rejected_length,
        "available": available,
        "selected_train_rows": len(selected["train"]),
        "selected_development_rows": len(selected["development"]),
        "train_group_counts": {
            group: sum(row["training_group"] == group for row in selected["train"])
            for group in ("math", "science")
        },
        "development_group_counts": {
            group: sum(
                row["training_group"] == group for row in selected["development"]
            )
            for group in ("math", "science")
        },
        "max_sequence_length": args.max_sequence_length,
        "workspace_slots": args.workspace_slots,
        "max_selected_positions": max_positions,
        "selected_truncations": 0,
        "identity_overlap": 0,
        "train_output": str(args.train_output.resolve()),
        "train_output_sha256": _sha256_file(args.train_output),
        "development_output": str(args.development_output.resolve()),
        "development_output_sha256": _sha256_file(args.development_output),
        "seed": args.seed,
    }
    _atomic_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--development-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--exclude-eval", type=Path, action="append", default=[])
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--workspace-slots", type=int, default=8)
    parser.add_argument("--train-per-group", type=int, default=1400)
    parser.add_argument("--development-per-group", type=int, default=160)
    parser.add_argument("--minimum-train-per-group", type=int, default=400)
    parser.add_argument("--minimum-development-per-group", type=int, default=40)
    parser.add_argument("--development-fraction", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=2026080603)
    args = parser.parse_args()
    positive = (
        args.max_sequence_length,
        args.workspace_slots,
        args.train_per_group,
        args.development_per_group,
        args.minimum_train_per_group,
        args.minimum_development_per_group,
    )
    if any(value <= 0 for value in positive):
        parser.error("VCR1 board dimensions must be positive")
    if not 0.0 < args.development_fraction < 1.0:
        parser.error("development fraction must be between zero and one")
    return args


def main() -> int:
    report = build(parse_args())
    print(
        "[vcr1-board] "
        f"train={report['selected_train_rows']} "
        f"development={report['selected_development_rows']} "
        f"max_positions={report['max_selected_positions']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
