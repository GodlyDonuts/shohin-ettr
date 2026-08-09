#!/usr/bin/env python3
"""Build an OLMoE-tokenized, development-disjoint broad owner curriculum."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from ttr1_revision import internal_revision_prompt, tokenize_with_draft_mask


SCHEMA = "shohin-obr1-broad-owner-train-v1"
REPORT_SCHEMA = "shohin-obr1-broad-owner-data-report-v1"
SOURCE_SCHEMA = "shohin-token-balanced-reasoning-mix-v1"
DEVELOPMENT_SCHEMA = "shohin-idr1-revision-data-report-v1"
WORD = re.compile(r"\w+")


class OBR1DataError(RuntimeError):
    """The OBR1 source, overlap, tokenizer, or retention contract differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_question(text: str) -> str:
    return " ".join(WORD.findall(str(text).lower()))


def grams(text: str, width: int) -> set[str]:
    words = WORD.findall(str(text).lower())
    if len(words) < width:
        return {" ".join(words)} if words else set()
    return {
        " ".join(words[index : index + width])
        for index in range(len(words) - width + 1)
    }


def unique_development_grams(
    questions: list[str], width: int
) -> tuple[set[str], int]:
    frequencies: Counter[str] = Counter()
    for question in questions:
        frequencies.update(grams(question, width))
    unique = {gram for gram, count in frequencies.items() if count == 1}
    return unique, len(frequencies) - len(unique)


def development_question(row: dict[str, Any]) -> str:
    internal = row.get("internal_draft")
    assessor = row.get("assessor")
    for payload in (internal, assessor):
        if isinstance(payload, dict) and str(payload.get("question", "")).strip():
            return str(payload["question"]).strip()
    raise OBR1DataError("development source question is absent")


def overlap_kind(
    question: str,
    exact: set[str],
    ngrams: set[str],
    width: int,
) -> str | None:
    normalized = normalized_question(question)
    if not normalized:
        return "empty"
    if normalized in exact:
        return "exact"
    if grams(question, width) & ngrams:
        return "ngram"
    return None


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


def _bound_report(
    report_path: Path,
    expected_schema: str,
    data_path: Path,
    output_key: str | None,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text())
    if report.get("schema") != expected_schema or report.get("status") != "complete":
        raise OBR1DataError(f"bound report differs: {report_path}")
    if output_key is None:
        expected_path = report.get("output")
        expected_sha = report.get("output_sha256")
    else:
        expected = report.get("outputs", {}).get(output_key, {})
        expected_path = expected.get("path")
        expected_sha = expected.get("sha256")
    if (
        Path(str(expected_path or "")).resolve() != data_path.resolve()
        or expected_sha != sha256_file(data_path)
    ):
        raise OBR1DataError(f"bound data differs: {data_path}")
    return report


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.output.is_symlink():
        raise OBR1DataError(f"refusing existing output root: {args.output}")
    source_report = _bound_report(
        args.source_report, SOURCE_SCHEMA, args.source, None
    )
    development_report = _bound_report(
        args.development_report,
        DEVELOPMENT_SCHEMA,
        args.development,
        "development",
    )
    if development_report.get("holdout_used") is True:
        raise OBR1DataError("holdout was used to build OBR1")

    development_rows = [
        json.loads(line) for line in args.development.read_text().splitlines() if line
    ]
    development_questions = [development_question(row) for row in development_rows]
    exact = {normalized_question(question) for question in development_questions}
    ngrams, repeated_ngram_count = unique_development_grams(
        development_questions, args.ngram
    )
    if not exact or not ngrams:
        raise OBR1DataError("development overlap boundary is empty")

    from transformers import AutoTokenizer
    from hf_product_reasoning_train import (
        PRODUCT_SYSTEM_PROMPT,
        render_reasoning_messages,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_root, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    admitted: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    groups: Counter[str] = Counter()
    charged: Counter[str] = Counter()
    maxima: Counter[str] = Counter()
    identities: set[str] = set()
    seen_questions: set[str] = set()
    with args.source.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question = str(row.get("question", "")).strip()
            response = str(row.get("response", "")).strip()
            group = str(row.get("training_group", "")).strip()
            if not question or not response or not group:
                drops["malformed"] += 1
                continue
            normalized = normalized_question(question)
            if normalized in seen_questions:
                drops["duplicate_question"] += 1
                continue
            kind = overlap_kind(question, exact, ngrams, args.ngram)
            if kind:
                drops[f"development_{kind}_overlap"] += 1
                continue
            task = "mbpp" if group == "code" else "broad_reasoning"
            prompt = internal_revision_prompt(
                question,
                "No prior draft is available.",
                task,
            )
            rendered = render_reasoning_messages(
                tokenizer,
                [
                    {"role": "system", "content": PRODUCT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                enable_thinking=False,
            )
            prompt_ids, draft_mask, _ = tokenize_with_draft_mask(tokenizer, rendered)
            response_ids = tokenizer.encode(response, add_special_tokens=False)
            response_ids.append(tokenizer.eos_token_id)
            total = len(prompt_ids) + len(response_ids)
            if total > args.max_sequence_length:
                drops["olmoe_overflow"] += 1
                continue
            identity = hashlib.sha256(normalized.encode()).hexdigest()
            if identity in identities:
                raise OBR1DataError("OBR1 source identity collision")
            identities.add(identity)
            seen_questions.add(normalized)
            groups[group] += 1
            charged[group] += len(response_ids)
            maxima["prompt"] = max(maxima["prompt"], len(prompt_ids))
            maxima["draft"] = max(maxima["draft"], sum(1 - value for value in draft_mask))
            maxima["target"] = max(maxima["target"], len(response_ids))
            maxima["total"] = max(maxima["total"], total)
            admitted.append(
                {
                    "schema": SCHEMA,
                    "identity_sha256": identity,
                    "source_line": line_number,
                    "source": str(row.get("source", "")),
                    "training_group": group,
                    "task": task,
                    "question": prompt,
                    "response": response,
                    "draft_control": "source_only_full_model_mask",
                    "verification": str(row.get("verification", "")),
                }
            )
    total_charged = sum(charged.values())
    if len(admitted) < 40_000 or total_charged < 12_000_000:
        raise OBR1DataError("OBR1 decontaminated broad corpus is too small")

    args.output.mkdir(parents=True)
    data = args.output / "train.jsonl"
    data_sha = atomic_lines(data, admitted)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "model_root": str(args.model_root.resolve()),
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "source_report_sha256": sha256_file(args.source_report),
        "development": str(args.development.resolve()),
        "development_sha256": sha256_file(args.development),
        "development_report_sha256": sha256_file(args.development_report),
        "holdout_used": False,
        "ngram": args.ngram,
        "development_unique_ngram_count": len(ngrams),
        "development_repeated_boilerplate_ngram_count": repeated_ngram_count,
        "max_sequence_length": args.max_sequence_length,
        "complete_retention": True,
        "source_rows": int(source_report["selected_rows"]),
        "admitted_rows": len(admitted),
        "dropped": dict(drops),
        "training_group_rows": dict(groups),
        "training_group_target_tokens": dict(charged),
        "charged_target_tokens": total_charged,
        "charged_target_fractions": {
            key: value / total_charged for key, value in charged.items()
        },
        "maximum_tokens": dict(maxima),
        "zero_exact_development_overlap": drops["development_exact_overlap"] >= 0,
        "zero_ngram_development_overlap": drops["development_ngram_overlap"] >= 0,
        "outputs": {
            "train": {
                "path": str(data.resolve()),
                "sha256": data_sha,
                "rows": len(admitted),
            }
        },
    }
    # The counters record removed collisions; admitted rows have zero collisions by construction.
    report["zero_exact_development_overlap"] = True
    report["zero_ngram_development_overlap"] = True
    atomic_json(args.output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--development-report", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ngram", type=int, default=13)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    args = parser.parse_args()
    if args.ngram <= 0 or args.max_sequence_length <= 0:
        parser.error("OBR1 dimensions must be positive")
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
