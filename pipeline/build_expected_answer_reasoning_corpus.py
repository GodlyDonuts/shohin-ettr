"""Admit one reasoning trace per prompt using a published expected answer."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from build_openthoughts_consensus_corpus import extract_final_answer, normalize_answer
from build_product_reasoning_seed_corpus import (
    ProductCorpusError,
    admissible,
    load_eval_contamination,
    prompt_sha256,
    quality_score,
    word_ngrams,
)


def parse_requirements(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        field, separator, expected = value.partition("=")
        if not separator or not field or not expected:
            raise ProductCorpusError(f"invalid required field equality: {value}")
        result[field] = expected
    return result


def expected_normalized(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return extract_final_answer(text) or normalize_answer(text)


def select_rows(
    rows: Iterable[dict[str, Any]],
    *,
    dataset_id: str,
    prompt_field: str,
    response_field: str,
    answer_field: str,
    domain: str,
    source_field: str | None,
    license_field: str | None,
    metadata_fields: list[str],
    requirements: dict[str, str],
    eval_exact: set[str],
    eval_ngrams: set[str],
    maximum_rows: int,
    seed: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    winners: dict[str, dict[str, Any]] = {}
    scores: dict[str, tuple[int, int, int, str]] = {}
    counters: Counter[str] = Counter()
    for raw in rows:
        counters["raw"] += 1
        if any(str(raw.get(field)) != expected for field, expected in requirements.items()):
            counters["required_field_rejected"] += 1
            continue
        question = raw.get(prompt_field)
        response = raw.get(response_field)
        expected = expected_normalized(raw.get(answer_field))
        if not question or not response or expected is None:
            counters["schema_or_expected_answer_rejected"] += 1
            continue
        question = str(question).strip()
        response = str(response).strip()
        predicted = extract_final_answer(response)
        if predicted is None:
            counters["response_answer_not_extracted"] += 1
            continue
        if predicted != expected:
            counters["response_answer_mismatch"] += 1
            continue
        identity = prompt_sha256(question)
        if identity in eval_exact:
            counters["eval_exact_rejected"] += 1
            continue
        if word_ngrams(question) & eval_ngrams:
            counters["eval_13gram_rejected"] += 1
            continue
        source = str(raw.get(source_field) or dataset_id) if source_field else dataset_id
        row = {
            "question": question,
            "response": response,
            "domain": domain,
            "source": source,
            "license": raw.get(license_field) if license_field else None,
            "difficulty": raw.get("difficulty"),
        }
        reason = admissible(row)
        if reason:
            counters[reason] += 1
            continue
        candidate = {
            **row,
            "prompt_sha256": identity,
            "training_group": domain,
            "verification": "expected_answer_match_v1",
            "expected_answer_normalized": expected,
            "source_metadata": {
                field: raw.get(field) for field in metadata_fields if field in raw
            },
        }
        score = quality_score(candidate)
        if identity not in winners or score > scores[identity]:
            if identity in winners:
                counters["duplicate_prompt_replaced"] += 1
            winners[identity] = candidate
            scores[identity] = score
        else:
            counters["duplicate_prompt_dropped"] += 1
    ordered = sorted(
        winners.values(),
        key=lambda row: hashlib.sha256(
            f"{seed}\0{row['prompt_sha256']}".encode()
        ).hexdigest(),
    )
    selected = ordered[:maximum_rows]
    counters["unique_answer_matched"] = len(winners)
    counters["selected"] = len(selected)
    return selected, counters


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise ProductCorpusError(f"refusing to replace corpus: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as handle:
            for row in rows:
                encoded = json.dumps(row, sort_keys=True, ensure_ascii=False).encode() + b"\n"
                digest.update(encoded)
                handle.write(encoded)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import load_dataset

    eval_exact, eval_ngrams = load_eval_contamination(args.eval)
    dataset = load_dataset(
        args.dataset_id,
        args.config,
        split=args.split,
        revision=args.dataset_revision,
        streaming=True,
    )
    requirements = parse_requirements(args.require_field_equals)
    selected, counters = select_rows(
        dataset,
        dataset_id=args.dataset_id,
        prompt_field=args.prompt_field,
        response_field=args.response_field,
        answer_field=args.answer_field,
        domain=args.domain,
        source_field=args.source_field,
        license_field=args.license_field,
        metadata_fields=args.metadata_field,
        requirements=requirements,
        eval_exact=eval_exact,
        eval_ngrams=eval_ngrams,
        maximum_rows=args.maximum_rows,
        seed=args.seed,
    )
    output_sha256 = atomic_write_jsonl(args.output, selected)
    report = {
        "schema": "shohin-expected-answer-reasoning-corpus-v1",
        "status": "complete",
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "config": args.config,
        "split": args.split,
        "fields": {
            "prompt": args.prompt_field,
            "response": args.response_field,
            "answer": args.answer_field,
            "source": args.source_field,
            "license": args.license_field,
            "metadata": args.metadata_field,
        },
        "required_field_equals": requirements,
        "domain": args.domain,
        "seed": args.seed,
        "maximum_rows": args.maximum_rows,
        "eval_paths": [str(path.resolve()) for path in args.eval],
        "eval_prompt_count": len(eval_exact),
        "eval_13gram_count": len(eval_ngrams),
        "counters": dict(sorted(counters.items())),
        "rows": len(selected),
        "output": str(args.output.resolve()),
        "output_sha256": output_sha256,
    }
    if args.report.exists():
        raise ProductCorpusError(f"refusing to replace report: {args.report}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--prompt-field", required=True)
    parser.add_argument("--response-field", required=True)
    parser.add_argument("--answer-field", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--source-field")
    parser.add_argument("--license-field")
    parser.add_argument("--metadata-field", action="append", default=[])
    parser.add_argument("--require-field-equals", action="append", default=[])
    parser.add_argument("--eval", type=Path, action="append", required=True)
    parser.add_argument("--maximum-rows", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.maximum_rows <= 0:
        parser.error("maximum rows must be positive")
    return args


def main() -> int:
    report = run(parse_args())
    print(
        f"[expected-answer-corpus] rows={report['rows']} "
        f"sha256={report['output_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
