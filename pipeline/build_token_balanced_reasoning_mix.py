#!/usr/bin/env python3
"""Build a deterministic, tokenizer-exact reasoning mix from multiple corpora."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable

try:
    from pipeline.audit_product_reasoning_token_mix import (
        question_response,
        render_reasoning_prompt,
        truncate_lengths,
    )
except ModuleNotFoundError:  # Direct execution with pipeline/ on PYTHONPATH.
    from audit_product_reasoning_token_mix import (
        question_response,
        render_reasoning_prompt,
        truncate_lengths,
    )


SCHEMA = "shohin-token-balanced-reasoning-mix-v1"
WORD = re.compile(r"\w+")


class TokenBalancedMixError(RuntimeError):
    """The requested exact-token corpus cannot be built safely."""


@dataclass(frozen=True)
class Candidate:
    row: dict[str, Any]
    identity: str
    group: str
    charged_tokens: int
    source_index: int
    priority: str
    quality_rank: int


@dataclass(frozen=True)
class EvalOverlapFilter:
    exact_questions: frozenset[str]
    unique_ngrams: frozenset[tuple[str, ...]]
    ngram_size: int
    references: tuple[dict[str, Any], ...]


def parse_weights(value: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in value.split(","):
        if "=" not in item:
            raise argparse.ArgumentTypeError("weights must use group=fraction entries")
        group, raw_weight = item.split("=", 1)
        group = group.strip()
        if not group or group in weights:
            raise argparse.ArgumentTypeError(
                "weight groups must be unique and nonempty"
            )
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "weight fractions must be numeric"
            ) from exc
        if not 0.0 < weight <= 1.0:
            raise argparse.ArgumentTypeError("weight fractions must be in (0, 1]")
        weights[group] = weight
    if not weights or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise argparse.ArgumentTypeError("weight fractions must sum to one")
    return weights


def _normalized_question(question: str) -> str:
    normalized = re.sub(r"\s+", " ", question).strip().casefold()
    if not normalized:
        raise TokenBalancedMixError("row has an empty normalized question")
    return normalized


def _normalized_overlap_question(question: str) -> str:
    normalized = " ".join(WORD.findall(question.casefold()))
    if not normalized:
        raise TokenBalancedMixError("row has an empty normalized question")
    return normalized


def _reference_question(row: dict[str, Any]) -> str:
    for key in ("assessor", "internal_draft"):
        payload = row.get(key)
        if isinstance(payload, dict) and str(payload.get("question", "")).strip():
            return str(payload["question"]).strip()
    question = str(row.get("question", "")).strip()
    if question:
        return question
    raise TokenBalancedMixError("evaluation reference row has no source question")


def _word_ngrams(question: str, size: int) -> set[tuple[str, ...]]:
    words = WORD.findall(question.casefold())
    return {
        tuple(words[index : index + size])
        for index in range(len(words) - size + 1)
    }


def _build_eval_overlap_filter(
    paths: Iterable[Path], ngram_size: int
) -> EvalOverlapFilter | None:
    paths = list(paths)
    if not paths:
        return None
    if ngram_size <= 0:
        raise TokenBalancedMixError("evaluation n-gram size must be positive")
    exact_questions: set[str] = set()
    unique_ngrams: set[tuple[str, ...]] = set()
    references: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise TokenBalancedMixError(
                f"evaluation reference does not exist: {path}"
            )
        frequencies: Counter[tuple[str, ...]] = Counter()
        normalized_questions: set[str] = set()
        rows = 0
        duplicate_questions = 0
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                rows += 1
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise TokenBalancedMixError(
                        f"malformed evaluation reference JSONL in {path}"
                    ) from exc
                question = _reference_question(row)
                normalized = _normalized_overlap_question(question)
                if normalized in normalized_questions:
                    duplicate_questions += 1
                normalized_questions.add(normalized)
                frequencies.update(_word_ngrams(question, ngram_size))
        if not rows:
            raise TokenBalancedMixError(f"evaluation reference is empty: {path}")
        split_unique = {gram for gram, count in frequencies.items() if count == 1}
        exact_questions.update(normalized_questions)
        unique_ngrams.update(split_unique)
        references.append(
            {
                "path": str(path.resolve()),
                "sha256": _source_sha256(path),
                "rows": rows,
                "normalized_questions": len(normalized_questions),
                "duplicate_normalized_questions": duplicate_questions,
                "all_word_ngrams": len(frequencies),
                "unique_word_ngrams": len(split_unique),
            }
        )
    return EvalOverlapFilter(
        exact_questions=frozenset(exact_questions),
        unique_ngrams=frozenset(unique_ngrams),
        ngram_size=ngram_size,
        references=tuple(references),
    )


def _quality_rank(row: dict[str, Any]) -> int:
    verification = str(row.get("verification") or "")
    if verification == "execution_verified_source_tests":
        return 4
    if verification in {
        "expected_answer_match_v1",
        "execution_verified",
    }:
        return 3
    if row.get("answer") or row.get("expected_answer_normalized"):
        return 2
    return 1


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _token_lengths(tokenizer: Any, question: str, response: str) -> tuple[int, int]:
    rendered = render_reasoning_prompt(tokenizer, question)
    return (
        len(tokenizer.encode(rendered, add_special_tokens=False)),
        len(tokenizer.encode(response, add_special_tokens=False)),
    )


def _read_candidates(
    sources: Iterable[Path],
    *,
    tokenizer: Any,
    weights: dict[str, float],
    total_target_tokens: int,
    max_sequence_length: int,
    workspace_slots: int,
    seed: int,
    eval_overlap_filter: EvalOverlapFilter | None,
) -> tuple[dict[str, list[Candidate]], dict[str, Any]]:
    counters: Counter[str] = Counter()
    source_reports: list[dict[str, Any]] = []
    by_identity: dict[str, Candidate] = {}
    for source_index, path in enumerate(sources):
        if not path.is_file():
            raise TokenBalancedMixError(f"source does not exist: {path}")
        source_counter: Counter[str] = Counter()
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                counters["raw_rows"] += 1
                source_counter["raw_rows"] += 1
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise TokenBalancedMixError(f"malformed JSONL in {path}") from exc
                pair = question_response(row)
                if pair is None:
                    counters["schema_rejected"] += 1
                    source_counter["schema_rejected"] += 1
                    continue
                question, response = pair
                group = str(row.get("training_group") or row.get("domain") or "")
                if group not in weights:
                    counters["unrequested_group"] += 1
                    source_counter["unrequested_group"] += 1
                    continue
                normalized = _normalized_question(question)
                if (
                    eval_overlap_filter is not None
                    and _normalized_overlap_question(question)
                    in eval_overlap_filter.exact_questions
                ):
                    counters["eval_exact_overlap_rejected"] += 1
                    source_counter["eval_exact_overlap_rejected"] += 1
                    continue
                if (
                    eval_overlap_filter is not None
                    and _word_ngrams(question, eval_overlap_filter.ngram_size)
                    & eval_overlap_filter.unique_ngrams
                ):
                    counters["eval_unique_ngram_overlap_rejected"] += 1
                    source_counter["eval_unique_ngram_overlap_rejected"] += 1
                    continue
                prompt_length, response_length = _token_lengths(
                    tokenizer, question, response
                )
                _, charged_tokens, response_cut, prompt_cut = truncate_lengths(
                    prompt_length,
                    response_length,
                    max_sequence_length=max_sequence_length,
                    workspace_slots=workspace_slots,
                )
                if response_cut:
                    counters["response_truncated_rejected"] += 1
                    source_counter["response_truncated_rejected"] += 1
                    continue
                if prompt_cut:
                    counters["prompt_truncated_rejected"] += 1
                    source_counter["prompt_truncated_rejected"] += 1
                    continue
                identity = hashlib.sha256(normalized.encode()).hexdigest()
                priority = hashlib.sha256(
                    f"{seed}\0{group}\0{identity}".encode()
                ).hexdigest()
                candidate = Candidate(
                    row={**row, "training_group": group},
                    identity=identity,
                    group=group,
                    charged_tokens=charged_tokens,
                    source_index=source_index,
                    priority=priority,
                    quality_rank=_quality_rank(row),
                )
                previous = by_identity.get(identity)
                if previous is None:
                    by_identity[identity] = candidate
                else:
                    counters["duplicate_questions"] += 1
                    source_counter["duplicate_questions"] += 1
                    replacement_key = (
                        candidate.quality_rank,
                        -candidate.charged_tokens,
                        -candidate.source_index,
                    )
                    previous_key = (
                        previous.quality_rank,
                        -previous.charged_tokens,
                        -previous.source_index,
                    )
                    if replacement_key > previous_key:
                        by_identity[identity] = candidate
                        counters["duplicate_replacements"] += 1
                        source_counter["duplicate_replacements"] += 1
                counters["valid_rows"] += 1
                source_counter["valid_rows"] += 1
        source_reports.append(
            {
                "path": str(path.resolve()),
                "sha256": _source_sha256(path),
                "counters": dict(sorted(source_counter.items())),
            }
        )

    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in by_identity.values():
        grouped[candidate.group].append(candidate)
    availability = {}
    for group, weight in weights.items():
        candidates = grouped[group]
        candidates.sort(
            key=lambda candidate: (-candidate.quality_rank, candidate.priority)
        )
        available_tokens = sum(candidate.charged_tokens for candidate in candidates)
        required_tokens = math.ceil(total_target_tokens * weight)
        availability[group] = {
            "rows": len(candidates),
            "charged_tokens": available_tokens,
            "required_tokens": required_tokens,
        }
        if available_tokens < required_tokens:
            raise TokenBalancedMixError(
                f"group {group!r} has {available_tokens} charged tokens, "
                f"below required {required_tokens}"
            )
    return dict(grouped), {
        "counters": dict(sorted(counters.items())),
        "sources": source_reports,
        "availability": availability,
    }


def build_token_balanced_mix(
    sources: list[Path],
    output: Path,
    report_path: Path,
    *,
    tokenizer: Any,
    model_revision: str,
    weights: dict[str, float],
    total_target_tokens: int,
    max_sequence_length: int,
    workspace_slots: int,
    seed: int,
    eval_references: list[Path] | None = None,
    eval_ngram_size: int = 13,
) -> dict[str, Any]:
    if not sources:
        raise TokenBalancedMixError("at least one source is required")
    if total_target_tokens <= 0:
        raise TokenBalancedMixError("total target tokens must be positive")
    if output.exists() or report_path.exists():
        raise TokenBalancedMixError("refusing to replace an existing output")
    eval_overlap_filter = _build_eval_overlap_filter(
        eval_references or [], eval_ngram_size
    )
    grouped, scan = _read_candidates(
        sources,
        tokenizer=tokenizer,
        weights=weights,
        total_target_tokens=total_target_tokens,
        max_sequence_length=max_sequence_length,
        workspace_slots=workspace_slots,
        seed=seed,
        eval_overlap_filter=eval_overlap_filter,
    )
    selected: list[Candidate] = []
    selected_metrics = {}
    for group, weight in weights.items():
        target = math.ceil(total_target_tokens * weight)
        charged = 0
        group_selected = []
        for candidate in grouped[group]:
            if charged >= target:
                break
            group_selected.append(candidate)
            charged += candidate.charged_tokens
        if charged < target:
            raise TokenBalancedMixError(f"group {group!r} selection underfilled")
        selected.extend(group_selected)
        selected_metrics[group] = {
            "rows": len(group_selected),
            "charged_target_tokens": charged,
            "target_charged_tokens": target,
        }
    selected.sort(
        key=lambda candidate: hashlib.sha256(
            f"{seed}\0output\0{candidate.identity}".encode()
        ).hexdigest()
    )
    identities = [candidate.identity for candidate in selected]
    if len(identities) != len(set(identities)):
        raise TokenBalancedMixError("selected output contains duplicate questions")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for candidate in selected:
            encoded = (json.dumps(candidate.row, sort_keys=True) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
    os.replace(temporary, output)
    actual_total = sum(
        metrics["charged_target_tokens"] for metrics in selected_metrics.values()
    )
    for group, metrics in selected_metrics.items():
        metrics["charged_target_fraction"] = (
            metrics["charged_target_tokens"] / actual_total
        )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "model_revision": model_revision,
        "tokenizer_name_or_path": str(tokenizer.name_or_path),
        "max_sequence_length": max_sequence_length,
        "workspace_slots": workspace_slots,
        "weights": dict(sorted(weights.items())),
        "requested_total_target_tokens": total_target_tokens,
        "actual_total_charged_target_tokens": actual_total,
        "selected_rows": len(selected),
        "selected_groups": dict(sorted(selected_metrics.items())),
        "duplicate_questions": 0,
        "response_truncated_rows": 0,
        "prompt_truncated_rows": 0,
        "seed": seed,
        "sources": scan["sources"],
        "scan_counters": scan["counters"],
        "availability": scan["availability"],
        "output": str(output.resolve()),
        "output_sha256": digest.hexdigest(),
    }
    if eval_overlap_filter is not None:
        report["eval_overlap_filter"] = {
            "algorithm": (
                "exact normalized source question plus union of word n-grams "
                "that occur in exactly one row within each reference split"
            ),
            "ngram_size": eval_overlap_filter.ngram_size,
            "exact_normalized_questions": len(eval_overlap_filter.exact_questions),
            "unique_word_ngrams": len(eval_overlap_filter.unique_ngrams),
            "references": list(eval_overlap_filter.references),
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_tmp = report_path.with_name(f".{report_path.name}.tmp.{os.getpid()}")
    report_tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(report_tmp, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--weights", required=True, type=parse_weights)
    parser.add_argument("--total-target-tokens", required=True, type=int)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--workspace-slots", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--eval-reference", action="append", type=Path, default=[])
    parser.add_argument("--eval-ngram-size", type=int, default=13)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_root,
        revision=args.model_revision,
        trust_remote_code=True,
    )
    report = build_token_balanced_mix(
        args.source,
        args.output,
        args.report,
        tokenizer=tokenizer,
        model_revision=args.model_revision,
        weights=args.weights,
        total_target_tokens=args.total_target_tokens,
        max_sequence_length=args.max_sequence_length,
        workspace_slots=args.workspace_slots,
        seed=args.seed,
        eval_references=args.eval_reference,
        eval_ngram_size=args.eval_ngram_size,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
