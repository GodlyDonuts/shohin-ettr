"""Build high-confidence OpenThoughts supervision from annotation consensus.

OpenThoughts3 contains sixteen independent teacher traces per question but no
reference answer or execution metadata.  This builder uses exact normalized
agreement between extracted final answers as a conservative pseudo-verifier.
It deliberately excludes code: agreement between code strings does not prove
functional correctness.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable

from build_product_reasoning_seed_corpus import (
    ProductCorpusError,
    admissible,
    load_eval_contamination,
    normalize_text,
    prompt_sha256,
    quality_score,
    word_ngrams,
)


ALLOWED_DOMAINS = ("math", "science")
FINAL_RE = re.compile(
    r"(?is)(?:final\s+answer|answer)\s*(?:is|:|=)\s*([^\n<]{1,256})"
)


@dataclass
class QuestionVotes:
    question: str
    domain: str
    source: str
    difficulty: Any
    annotations: int = 0
    answers: Counter[str] = field(default_factory=Counter)


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


def extract_row(row: dict[str, Any]) -> dict[str, Any] | None:
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
        "difficulty": row.get("difficulty"),
    }


def boxed_answers(text: str) -> list[str]:
    answers: list[str] = []
    marker = r"\boxed{"
    offset = 0
    while True:
        start = text.find(marker, offset)
        if start < 0:
            return answers
        index = start + len(marker)
        depth = 1
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            content = text[start + len(marker) : index - 1].strip()
            if content:
                answers.append(content)
        offset = max(index, start + len(marker))


def normalize_answer(answer: str) -> str | None:
    value = answer.strip()
    value = re.sub(r"(?is)^\s*(?:is|:|=)\s*", "", value)
    value = value.replace("−", "-").replace("–", "-")
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    value = re.sub(r"\\(?:left|right)", "", value)
    value = value.strip(" \t\r\n$.,;:!`")
    value = re.sub(r"\s+", "", value).casefold()
    if not value or len(value) > 192:
        return None
    if value in {"unknown", "cannotbedetermined", "n/a", "none"}:
        return None
    return value


def extract_final_answer(response: str) -> str | None:
    boxed = boxed_answers(response)
    for candidate in reversed(boxed):
        normalized = normalize_answer(candidate)
        if normalized:
            return normalized
    matches = FINAL_RE.findall(response)
    for candidate in reversed(matches):
        normalized = normalize_answer(candidate)
        if normalized:
            return normalized
    return None


def consensus_answer(
    votes: QuestionVotes,
    minimum_extracted: int,
    minimum_votes: int,
    minimum_fraction: float,
    minimum_margin: int,
) -> tuple[str, int, int, float] | None:
    extracted = sum(votes.answers.values())
    if extracted < minimum_extracted or not votes.answers:
        return None
    ranked = votes.answers.most_common(2)
    answer, count = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    fraction = count / extracted
    if count < minimum_votes or fraction < minimum_fraction or count - runner_up < minimum_margin:
        return None
    return answer, count, runner_up, fraction


def collect_votes(
    rows: Iterable[dict[str, Any]],
    eval_exact: set[str],
    eval_ngrams: set[str],
) -> tuple[dict[str, QuestionVotes], Counter[str]]:
    questions: dict[str, QuestionVotes] = {}
    rejected: set[str] = set()
    counters: Counter[str] = Counter()
    for raw in rows:
        counters["raw"] += 1
        row = extract_row(raw)
        if row is None:
            counters["schema_or_domain_rejected"] += 1
            continue
        identity = prompt_sha256(row["question"])
        if identity in rejected:
            counters["question_previously_rejected"] += 1
            continue
        if identity not in questions:
            reason = admissible(row)
            if reason:
                rejected.add(identity)
                counters[reason] += 1
                continue
            if identity in eval_exact:
                rejected.add(identity)
                counters["eval_exact_rejected"] += 1
                continue
            if word_ngrams(row["question"]) & eval_ngrams:
                rejected.add(identity)
                counters["eval_13gram_rejected"] += 1
                continue
            questions[identity] = QuestionVotes(
                question=row["question"],
                domain=row["domain"],
                source=row["source"],
                difficulty=row["difficulty"],
            )
        votes = questions[identity]
        if votes.domain != row["domain"] or normalize_text(votes.question) != normalize_text(row["question"]):
            raise ProductCorpusError(f"inconsistent repeated prompt: {identity}")
        votes.annotations += 1
        answer = extract_final_answer(row["response"])
        if answer is None:
            counters["answer_not_extracted"] += 1
        else:
            votes.answers[answer] += 1
            counters["answer_extracted"] += 1
    counters["eligible_questions"] = len(questions)
    return questions, counters


def select_consensus_rows(
    rows: Iterable[dict[str, Any]],
    accepted: dict[str, tuple[str, int, int, float]],
    votes: dict[str, QuestionVotes],
) -> list[dict[str, Any]]:
    winners: dict[str, dict[str, Any]] = {}
    scores: dict[str, tuple[int, int, int, str]] = {}
    for raw in rows:
        row = extract_row(raw)
        if row is None:
            continue
        identity = prompt_sha256(row["question"])
        decision = accepted.get(identity)
        if decision is None or extract_final_answer(row["response"]) != decision[0]:
            continue
        score = quality_score(row)
        if identity not in winners or score > scores[identity]:
            winners[identity] = row
            scores[identity] = score
    selected: list[dict[str, Any]] = []
    for identity, row in winners.items():
        answer, count, runner_up, fraction = accepted[identity]
        question_votes = votes[identity]
        selected.append(
            {
                **row,
                "prompt_sha256": identity,
                "training_group": row["domain"],
                "verification": "annotation_consensus_v1",
                "consensus_answer": answer,
                "annotations_seen": question_votes.annotations,
                "answers_extracted": sum(question_votes.answers.values()),
                "consensus_votes": count,
                "runner_up_votes": runner_up,
                "consensus_fraction": fraction,
            }
        )
    selected.sort(key=lambda row: row["prompt_sha256"])
    return selected


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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

    def dataset() -> Iterable[dict[str, Any]]:
        return load_dataset(
            args.dataset_id,
            split=args.split,
            revision=args.dataset_revision,
            streaming=True,
        )

    votes, counters = collect_votes(dataset(), eval_exact, eval_ngrams)
    accepted: dict[str, tuple[str, int, int, float]] = {}
    rejection: Counter[str] = Counter()
    for identity, question_votes in votes.items():
        decision = consensus_answer(
            question_votes,
            args.minimum_extracted,
            args.minimum_votes,
            args.minimum_fraction,
            args.minimum_margin,
        )
        if decision is None:
            rejection["consensus_gate_rejected"] += 1
        else:
            accepted[identity] = decision
    selected = select_consensus_rows(dataset(), accepted, votes)
    if len(selected) != len(accepted):
        raise ProductCorpusError(
            f"second pass recovered {len(selected)} of {len(accepted)} accepted questions"
        )
    output_sha256 = atomic_write_jsonl(args.output, selected)
    report = {
        "schema": "shohin-openthoughts-consensus-corpus-v1",
        "status": "complete",
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "split": args.split,
        "thresholds": {
            "minimum_extracted": args.minimum_extracted,
            "minimum_votes": args.minimum_votes,
            "minimum_fraction": args.minimum_fraction,
            "minimum_margin": args.minimum_margin,
        },
        "eval_paths": [str(path.resolve()) for path in args.eval],
        "eval_prompt_count": len(eval_exact),
        "eval_13gram_count": len(eval_ngrams),
        "first_pass_counters": dict(sorted(counters.items())),
        "consensus_counters": dict(sorted(rejection.items())),
        "rows": len(selected),
        "rows_by_domain": dict(sorted(Counter(row["domain"] for row in selected).items())),
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
    parser.add_argument("--dataset-id", default="open-thoughts/OpenThoughts3-1.2M")
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-extracted", type=int, default=8)
    parser.add_argument("--minimum-votes", type=int, default=8)
    parser.add_argument("--minimum-fraction", type=float, default=0.60)
    parser.add_argument("--minimum-margin", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    print(
        f"[openthoughts-consensus] rows={report['rows']} "
        f"sha256={report['output_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
