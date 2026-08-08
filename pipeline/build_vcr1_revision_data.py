#!/usr/bin/env python3
"""Build source-disjoint verifier-supervised VCR1 revision data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from hf_pcj1_pairwise_judge import assigned_split

PAIR_SCHEMA = "shohin-cvg1-whole-lineage-pairs-v1"
TRAIN_SCHEMA = "shohin-vcr1-revision-train-v1"
EVAL_SCHEMA = "shohin-vcr1-revision-eval-v1"
REPORT_SCHEMA = "shohin-vcr1-revision-data-report-v1"
SPLITS = ("train", "development", "holdout")
TASKS = ("math500", "bbh_logic", "mbpp")
SPLIT_SEED = 2026080811
ORDER_SEED = 2026080813


class VCR1DataError(RuntimeError):
    """VCR1 source data or partitioning differs from the frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise VCR1DataError(f"refusing existing VCR1 output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
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
        raise VCR1DataError(f"refusing existing VCR1 report: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise VCR1DataError(f"missing VCR1 source: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise VCR1DataError(f"empty VCR1 source: {path}")
    return rows


def load_source_banks(paths: list[Path]) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in _load_jsonl(path):
            identity = row.get("identity_sha256")
            task = row.get("task")
            if not isinstance(identity, str) or len(identity) != 64:
                raise VCR1DataError("VCR1 source identity is invalid")
            if task not in TASKS:
                raise VCR1DataError("VCR1 source task differs")
            if identity in sources:
                raise VCR1DataError("VCR1 source identity is duplicated")
            sources[identity] = row
    return sources


def source_task_prompt(row: dict[str, Any]) -> str:
    task = str(row["task"])
    if task == "mbpp":
        tests = "\n".join(str(item) for item in row.get("test_list", ()))
        return (
            "Write Python code that solves the task and passes every test. Return "
            "only executable Python code, without Markdown fences.\n\nTask:\n"
            f"{row['text']}\n\nTests:\n{tests}"
        )
    question = str(row["question"])
    if task == "bbh_logic":
        return (
            f"{question}\n\nReason carefully, then put only the exact requested "
            "answer or option label inside \\boxed{{}}."
        )
    return question


def revision_prompt(
    task_prompt: str,
    candidate_a: str,
    candidate_b: str,
) -> str:
    return (
        "Produce one corrected solution to the problem. The two attempts below "
        "may each contain valid steps or errors. Check them against the problem, "
        "repair any mistakes, and return a complete final solution rather than a "
        "verdict about the candidates.\n\n"
        f"Problem:\n{task_prompt}\n\nCandidate A:\n{candidate_a}\n\n"
        f"Candidate B:\n{candidate_b}\n\n"
        "Solve the original problem now. Preserve executable-code-only format for "
        f"code tasks.\n\nOriginal problem:\n{task_prompt}"
    )


def _order(identity: str, presentation: int) -> tuple[int, int]:
    digest = hashlib.sha256(
        f"{ORDER_SEED}\0{identity}\0{presentation}".encode()
    ).digest()
    return (0, 1) if digest[0] & 1 == 0 else (1, 0)


def canonical_target(source: dict[str, Any]) -> str:
    if source["task"] == "mbpp":
        target = source.get("code")
    else:
        target = source.get("answer")
    if not isinstance(target, str) or not target.strip():
        raise VCR1DataError("VCR1 source target is empty")
    return target.strip()


def training_target(pair: dict[str, Any], source: dict[str, Any]) -> tuple[str, str]:
    candidates = pair["candidates"]
    correct = [bool(candidate["correct"]) for candidate in candidates]
    if correct[0] != correct[1]:
        return str(candidates[int(correct[1])]["completion"]), "verified_candidate"
    if all(correct):
        chosen = min(
            candidates, key=lambda item: (len(item["completion"]), item["lineage"])
        )
        return str(chosen["completion"]), "shortest_verified_candidate"
    return canonical_target(source), "source_verified_repair"


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise VCR1DataError(f"refusing existing VCR1 output root: {args.output}")
    args.output.mkdir(parents=True)
    pairs = _load_jsonl(args.pairs)
    sources = load_source_banks(args.banks)
    pair_ids = {str(row.get("identity_sha256")) for row in pairs}
    if len(pair_ids) != len(pairs) or pair_ids != set(sources):
        raise VCR1DataError("VCR1 pair/source identity coverage differs")

    train_rows: list[dict[str, Any]] = []
    evaluation: dict[str, list[dict[str, Any]]] = {
        "development": [],
        "holdout": [],
    }
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    target_counts: Counter[str] = Counter()
    for pair in pairs:
        if pair.get("schema") != PAIR_SCHEMA:
            raise VCR1DataError("VCR1 pair schema differs")
        identity = str(pair["identity_sha256"])
        source = sources[identity]
        if pair.get("task") != source.get("task"):
            raise VCR1DataError("VCR1 task binding differs")
        if str(pair.get("question")) not in source_task_prompt(source):
            raise VCR1DataError("VCR1 question/source binding differs")
        split = assigned_split(identity, args.split_seed)
        counts[split]["pairs"] += 1
        counts[split][str(pair["task"])] += 1
        counts[split][str(pair["outcome_class"])] += 1
        target, target_kind = training_target(pair, source)
        if split == "train":
            presentations = (
                4 if pair["outcome_class"] in ("base_only", "expert_only") else 1
            )
            for presentation in range(presentations):
                order = _order(identity, presentation)
                candidates = pair["candidates"]
                train_rows.append(
                    {
                        "schema": TRAIN_SCHEMA,
                        "identity_sha256": hashlib.sha256(
                            f"{identity}\0{presentation}".encode()
                        ).hexdigest(),
                        "source_identity_sha256": identity,
                        "outcome_class": pair["outcome_class"],
                        "presentation": presentation,
                        "question": revision_prompt(
                            source_task_prompt(source),
                            str(candidates[order[0]]["completion"]),
                            str(candidates[order[1]]["completion"]),
                        ),
                        "response": target,
                        "target_kind": target_kind,
                    }
                )
                target_counts[target_kind] += 1
            continue
        order = _order(identity, 0)
        candidates = pair["candidates"]
        evaluation[split].append(
            {
                "schema": EVAL_SCHEMA,
                "identity_sha256": identity,
                "split": split,
                "task": pair["task"],
                "outcome_class": pair["outcome_class"],
                "question": revision_prompt(
                    source_task_prompt(source),
                    str(candidates[order[0]]["completion"]),
                    str(candidates[order[1]]["completion"]),
                ),
                "candidates": pair["candidates"],
                "assessor": source,
                "runtime_fields": ["question"],
            }
        )

    if not train_rows or any(not evaluation[split] for split in evaluation):
        raise VCR1DataError("VCR1 output split is empty")
    paths = {
        "train": args.output / "train.jsonl",
        "development": args.output / "development.jsonl",
        "holdout": args.output / "holdout.jsonl",
    }
    hashes = {
        "train": _atomic_lines(paths["train"], train_rows),
        "development": _atomic_lines(paths["development"], evaluation["development"]),
        "holdout": _atomic_lines(paths["holdout"], evaluation["holdout"]),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "pairs": str(args.pairs.resolve()),
        "pairs_sha256": sha256_file(args.pairs),
        "banks": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.banks
        ],
        "split_seed": args.split_seed,
        "order_seed": ORDER_SEED,
        "counts": {split: dict(counts[split]) for split in SPLITS},
        "target_counts": dict(target_counts),
        "outputs": {
            split: {
                "path": str(paths[split].resolve()),
                "sha256": hashes[split],
                "rows": len(train_rows) if split == "train" else len(evaluation[split]),
            }
            for split in paths
        },
        "runtime_fields": ["question"],
        "assessor_fields_visible_to_model": False,
    }
    _atomic_json(args.output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument(
        "--bank", dest="banks", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    report = build(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
