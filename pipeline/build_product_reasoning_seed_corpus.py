"""Build a deduplicated multi-domain reasoning seed from OpenThoughts3.

The upstream corpus contains many annotations per question.  This builder
keeps one deterministic high-quality trace per normalized prompt, applies
basic degeneration and benchmark-overlap filters, and publishes a frozen
JSONL plus an auditable report.  Teacher traces remain explicitly marked as
unverified; later solver/execution gates can promote them independently.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable


class ProductCorpusError(RuntimeError):
    """The product-reasoning corpus contract was violated."""


ALLOWED_DOMAINS = ("math", "code", "science")


def normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def prompt_sha256(question: str) -> str:
    return hashlib.sha256(normalize_text(question).encode()).hexdigest()


def word_ngrams(text: str, width: int = 13) -> set[str]:
    words = re.findall(r"[a-z0-9]+", normalize_text(text))
    return {" ".join(words[index : index + width]) for index in range(len(words) - width + 1)}


def _messages(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("from") or item.get("role")
        content = item.get("value") or item.get("content")
        if role and content:
            result.append({"role": str(role).lower(), "content": str(content)})
    return result


def extract_openthoughts_row(row: dict[str, Any]) -> dict[str, Any] | None:
    messages = _messages(row.get("conversations") or row.get("messages"))
    questions = [
        message["content"]
        for message in messages
        if message["role"] in {"human", "user"}
    ]
    responses = [
        message["content"]
        for message in messages
        if message["role"] in {"assistant", "gpt"}
    ]
    question = questions[0] if questions else row.get("question") or row.get("prompt")
    response = responses[-1] if responses else row.get("response") or row.get("output")
    domain = str(row.get("domain") or row.get("_domain") or "").lower()
    if domain not in ALLOWED_DOMAINS or not question or not response:
        return None
    return {
        "question": str(question).strip(),
        "response": str(response).strip(),
        "domain": domain,
        "source": str(row.get("source") or "open-thoughts/OpenThoughts3-1.2M"),
        "license": row.get("license"),
        "difficulty": row.get("difficulty"),
    }


def degeneration_ratio(text: str) -> float:
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return 0.0
    return 1.0 - len(set(lines)) / len(lines)


def quality_score(row: dict[str, Any]) -> tuple[int, int, int, str]:
    response = row["response"]
    domain = row["domain"]
    final_marker = int(
        r"\boxed" in response
        or bool(re.search(r"(?:final answer|answer\s*:)", response, re.IGNORECASE))
        or (domain == "code" and "```" in response)
    )
    reasoning_marker = int(
        "<think>" in response
        or bool(re.search(r"\b(?:therefore|because|step\s+1|analysis)\b", response, re.IGNORECASE))
    )
    target = 5000 if domain != "code" else 7000
    length_score = -abs(min(len(response), target * 2) - target)
    tie = hashlib.sha256(response.encode()).hexdigest()
    return final_marker, reasoning_marker, length_score, tie


def admissible(row: dict[str, Any]) -> str | None:
    question = row["question"]
    response = row["response"]
    if len(question) < 24:
        return "question_too_short"
    if len(question) > 40_000:
        return "question_too_long"
    if len(response) < 160:
        return "response_too_short"
    if len(response) > 80_000:
        return "response_too_long"
    if degeneration_ratio(response) > 0.35:
        return "repeated_lines"
    if normalize_text(question) in normalize_text(response) and len(response) < len(question) * 1.15:
        return "question_echo"
    return None


def load_eval_contamination(paths: Iterable[Path]) -> tuple[set[str], set[str]]:
    exact: set[str] = set()
    ngrams: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                question = row.get("question") or row.get("problem") or row.get("prompt")
                if not question:
                    continue
                normalized = normalize_text(str(question))
                exact.add(hashlib.sha256(normalized.encode()).hexdigest())
                ngrams.update(word_ngrams(normalized))
    return exact, ngrams


def select_rows(
    rows: Iterable[dict[str, Any]],
    caps: dict[str, int],
    seed: int,
    eval_exact: set[str],
    eval_ngrams: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    winners: dict[str, dict[str, Any]] = {}
    winner_scores: dict[str, tuple[int, int, int, str]] = {}
    counters: Counter[str] = Counter()
    for raw in rows:
        counters["raw"] += 1
        row = extract_openthoughts_row(raw)
        if row is None:
            counters["schema_or_domain_rejected"] += 1
            continue
        reason = admissible(row)
        if reason:
            counters[reason] += 1
            continue
        identity = prompt_sha256(row["question"])
        if identity in eval_exact:
            counters["eval_exact_rejected"] += 1
            continue
        if word_ngrams(row["question"]) & eval_ngrams:
            counters["eval_13gram_rejected"] += 1
            continue
        score = quality_score(row)
        if identity not in winners or score > winner_scores[identity]:
            if identity in winners:
                counters["duplicate_annotation_replaced"] += 1
            winners[identity] = row
            winner_scores[identity] = score
        else:
            counters["duplicate_annotation_dropped"] += 1

    selected: list[dict[str, Any]] = []
    available_by_domain: Counter[str] = Counter(row["domain"] for row in winners.values())
    for domain in ALLOWED_DOMAINS:
        candidates = [
            (identity, row)
            for identity, row in winners.items()
            if row["domain"] == domain
        ]
        candidates.sort(
            key=lambda item: hashlib.sha256(
                f"{seed}\0{domain}\0{item[0]}".encode()
            ).hexdigest()
        )
        for identity, row in candidates[: caps[domain]]:
            selected.append(
                {
                    **row,
                    "prompt_sha256": identity,
                    "training_group": domain,
                    "verification": "teacher_trace_unverified",
                }
            )
    selected.sort(
        key=lambda row: hashlib.sha256(
            f"{seed}\0{row['prompt_sha256']}".encode()
        ).hexdigest()
    )
    counters["unique_admissible"] = len(winners)
    counters["selected"] = len(selected)
    report = {
        "counters": dict(sorted(counters.items())),
        "available_by_domain": dict(sorted(available_by_domain.items())),
        "selected_by_domain": dict(
            sorted(Counter(row["domain"] for row in selected).items())
        ),
    }
    return selected, report


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise ProductCorpusError(f"refusing to replace corpus: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            line = json.dumps(row, sort_keys=True, ensure_ascii=False).encode() + b"\n"
            digest.update(line)
            handle.write(line)
    os.replace(temporary, path)
    return digest.hexdigest()


def parse_caps(values: list[str]) -> dict[str, int]:
    caps = {domain: 0 for domain in ALLOWED_DOMAINS}
    for value in values:
        domain, separator, raw_count = value.partition("=")
        if not separator or domain not in caps:
            raise ProductCorpusError(f"invalid domain cap: {value}")
        caps[domain] = int(raw_count)
    if any(count <= 0 for count in caps.values()):
        raise ProductCorpusError("every domain requires a positive cap")
    return caps


def run(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import load_dataset

    eval_exact, eval_ngrams = load_eval_contamination(args.eval)
    dataset = load_dataset(
        args.dataset_id,
        split=args.split,
        revision=args.dataset_revision,
        streaming=True,
    )
    selected, selection_report = select_rows(
        dataset,
        parse_caps(args.domain_cap),
        args.seed,
        eval_exact,
        eval_ngrams,
    )
    corpus_sha = _atomic_write_jsonl(args.output, selected)
    report = {
        "schema": "shohin-product-reasoning-seed-corpus-v1",
        "status": "complete",
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "split": args.split,
        "seed": args.seed,
        "eval_paths": [str(path.resolve()) for path in args.eval],
        "eval_prompt_count": len(eval_exact),
        "eval_13gram_count": len(eval_ngrams),
        "output": str(args.output.resolve()),
        "output_sha256": corpus_sha,
        "rows": len(selected),
        **selection_report,
    }
    if args.report.exists():
        raise ProductCorpusError(f"refusing to replace report: {args.report}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", default="open-thoughts/OpenThoughts3-1.2M")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--domain-cap", action="append", required=True)
    parser.add_argument("--eval", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(
        f"[product-corpus] rows={report['rows']} sha256={report['output_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
