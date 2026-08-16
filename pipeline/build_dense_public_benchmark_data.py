#!/usr/bin/env python3
"""Freeze source-disjoint public benchmark screens for dense Shohin pairs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "shohin-dense-public-benchmark-data-v1"
QUESTION_SCHEMA = "shohin-dense-public-benchmark-question-v1"
ASSESSOR_SCHEMA = "shohin-dense-public-benchmark-assessor-v1"
SCREEN_SEED = 2026081517
SCREEN_ROWS = 256
MMLU_REPO_REVISION = "f418b116db00b065c2aea046518d8fcf74d39872"
MMLU_DATA_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
IFEVAL_REPO_REVISION = "589e977488f21a336a3d3da9b96da91ddbcf935e"
MUSR_REPO_REVISION = "b1f4d4168a9cfc6760e8b74d728e4516023dfaa5"
MMLU_CHOICES = tuple("ABCDEFGHIJKLMNOP")


class DenseBenchmarkDataError(RuntimeError):
    """The benchmark source, prompt, identity, or split contract differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def text_digest(value: str) -> str:
    normalized = " ".join(value.split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def identity(benchmark: str, upstream_id: str, prompt: str) -> str:
    return hashlib.sha256(
        f"{benchmark}\0{upstream_id}\0{text_digest(prompt)}".encode()
    ).hexdigest()


def mmlu_format_example(row: dict[str, Any], include_answer: bool) -> str:
    prompt = f"Question:\n{row['question']}\nOptions:\n"
    options = [option for option in row["options"] if option != "N/A"]
    for index, option in enumerate(options):
        prompt += f"{MMLU_CHOICES[index]}. {option}\n"
    if include_answer:
        cot = str(row["cot_content"]).replace(
            "A: Let's think step by step.", "Answer: Let's think step by step."
        )
        return prompt + cot + "\n\n"
    return prompt + "Answer: Let's think step by step."


def mmlu_prompt(
    validation_by_category: dict[str, list[dict[str, Any]]],
    row: dict[str, Any],
) -> str:
    category = str(row["category"])
    initial = (
        "The following are multiple choice questions (with answers) about "
        f'{category}. Think step by step and then finish your answer with "the '
        'answer is (X)" where X is the correct letter choice.\n\n'
    )
    examples = validation_by_category.get(category, [])[:5]
    if len(examples) != 5:
        raise DenseBenchmarkDataError(f"MMLU-Pro lacks five shots for {category}")
    return (
        initial
        + "".join(mmlu_format_example(example, True) for example in examples)
        + mmlu_format_example(row, False)
    )


MUSR_DOMAINS = {
    "murder_mystery": {
        "file": "murder_mystery.json",
        "hint_before_question": False,
        "hint": (
            "Before selecting a choice, explain your reasoning step by step. The "
            "murderer needs to have a means (access to weapon), motive (reason to "
            "kill the victim), and opportunity (access to crime scene) in order to "
            "have killed the victim. Innocent suspects may have two of these "
            "proven, but not all three. An innocent suspect may be suspicious for "
            "some other reason, but they will not have all of motive, means, and "
            "opportunity established.\n\nIf you believe that both suspects have "
            "motive, means, and opportunity, you should make an educated guess pick "
            "the one for whom these are best established. If you believe that "
            "neither suspect has all three established, then choose the suspect "
            "where these are most clearly established."
        ),
    },
    "object_placements": {
        "file": "object_placements.json",
        "hint_before_question": True,
        "hint": (
            "Based on this story, we want to identify where someone believes that a "
            "certain object is at the end of the story. In order to do that, you "
            "need to read the story and keep track of where they think the object is "
            "at each point. When an object is moved, the person may observe its new "
            "location if they saw it move.\n\nTo see where an object ends up, they "
            "must be able to see the location that it moves to and not be too "
            "distracted by what they are doing. If they do not observe the object "
            "moving, then they will still believe it to be in the last location "
            "where they observed it."
        ),
    },
    "team_allocation": {
        "file": "team_allocation.json",
        "hint_before_question": False,
        "hint": (
            "The story should allow you to determine how good each person is at a "
            "skill. Roughly, each person is either great, acceptable, or bad at a "
            "task. We want to find an optimal assignment of people to tasks that "
            "uses their skills as well as possible. In addition, one task will have "
            "to have two people assigned to it. The effectiveness of their teamwork "
            "(great team, acceptable team, or bad team) also impacts the overall "
            "quality of the assignment.\n\nWhen two people need to work on a task "
            "and one is bad at it, they don't necessarily benefit from the other "
            "person being good, unless they work well together.\n\nWith different "
            "strengths, weaknesses, and interpersonal dynamics at play, you should "
            "allocate your team to find the single assignment to ensure that the "
            "tasks overall are completed as effectively as possible.\n\n"
        ),
    },
}


def musr_prompt(context: str, question: dict[str, Any], domain: str) -> str:
    spec = MUSR_DOMAINS[domain]
    choices = "\n".join(
        f"{index + 1} - {choice}" for index, choice in enumerate(question["choices"])
    )
    hint = str(spec["hint"])
    if spec["hint_before_question"]:
        middle = f"{hint}\n\n{question['question']}"
    else:
        middle = f"{question['question']}"
    suffix = (
        "You must pick one option. "
        + ("" if spec["hint_before_question"] else hint + " ")
        + "Explain your reasoning step by step before you answer. Finally, the last "
        'thing you generate should be "ANSWER: (your answer here, including the '
        'choice number)"'
    )
    return f"{context}\n\n{middle}\n\nPick one of the following choices:\n{choices}\n\n{suffix}"


def ranked_screen(
    rows: list[dict[str, Any]], count: int, stratum_key: str
) -> list[dict[str, Any]]:
    if count <= 0 or count > len(rows):
        raise DenseBenchmarkDataError("screen count is outside the benchmark")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[stratum_key])].append(row)
    quotas: dict[str, int] = {}
    fractional: list[tuple[float, str]] = []
    allocated = 0
    for name, group in grouped.items():
        exact = count * len(group) / len(rows)
        floor = min(len(group), math.floor(exact))
        quotas[name] = floor
        allocated += floor
        fractional.append((exact - floor, name))
    for _, name in sorted(fractional, key=lambda item: (-item[0], item[1])):
        if allocated == count:
            break
        if quotas[name] < len(grouped[name]):
            quotas[name] += 1
            allocated += 1
    if allocated != count:
        raise DenseBenchmarkDataError("screen quota allocation differs")
    selected: list[dict[str, Any]] = []
    for name, group in sorted(grouped.items()):
        ranked = sorted(
            group,
            key=lambda row: hashlib.sha256(
                f"{SCREEN_SEED}\0{row['identity_sha256']}".encode()
            ).hexdigest(),
        )
        selected.extend(ranked[: quotas[name]])
    return sorted(selected, key=lambda row: row["identity_sha256"])


def training_prompt_hashes(paths: Iterable[Path]) -> tuple[set[str], int]:
    hashes: set[str] = set()
    rows = 0
    fields = ("source_prompt", "question", "problem", "prompt", "text", "input")
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                value = next(
                    (row.get(field) for field in fields if row.get(field)), None
                )
                if value is None and isinstance(row.get("assessor"), dict):
                    assessor = row["assessor"]
                    value = next(
                        (
                            assessor.get(field)
                            for field in fields
                            if assessor.get(field)
                        ),
                        None,
                    )
                if value is not None:
                    hashes.add(text_digest(str(value)))
                rows += 1
    if rows == 0 or not hashes:
        raise DenseBenchmarkDataError("training source projection is empty")
    return hashes, rows


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise DenseBenchmarkDataError(f"refusing to replace {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            line = canonical_json(row) + b"\n"
            digest.update(line)
            handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise DenseBenchmarkDataError(f"refusing to replace {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_mmlu() -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("TIGER-Lab/MMLU-Pro", revision=MMLU_DATA_REVISION)
    test = list(dataset["test"])
    validation_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset["validation"]:
        validation_by_category[str(row["category"])].append(dict(row))
    if len(test) != 12032:
        raise DenseBenchmarkDataError("MMLU-Pro test cardinality differs")
    rows = []
    for source in test:
        source = dict(source)
        prompt = mmlu_prompt(validation_by_category, source)
        upstream_id = str(source["question_id"])
        ident = identity("mmlu_pro", upstream_id, prompt)
        rows.append(
            {
                "identity_sha256": ident,
                "benchmark": "mmlu_pro",
                "stratum": str(source["category"]),
                "upstream_id": upstream_id,
                "question": prompt,
                "response_mode": "general",
                "assessor": {
                    "answer": str(source["answer"]),
                    "category": str(source["category"]),
                    "question_id": source["question_id"],
                },
            }
        )
    return rows


def load_ifeval(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source = json.loads(line)
            prompt = str(source["prompt"])
            upstream_id = str(source["key"])
            ident = identity("ifeval", upstream_id, prompt)
            instruction_ids = list(source["instruction_id_list"])
            rows.append(
                {
                    "identity_sha256": ident,
                    "benchmark": "ifeval",
                    "stratum": str(len(instruction_ids)),
                    "upstream_id": upstream_id,
                    "question": prompt,
                    "response_mode": "general",
                    "assessor": {
                        "key": source["key"],
                        "instruction_id_list": instruction_ids,
                        "prompt": prompt,
                        "kwargs": source["kwargs"],
                    },
                }
            )
    if len(rows) != 541:
        raise DenseBenchmarkDataError("IFEval cardinality differs")
    return rows


def load_musr(root: Path) -> list[dict[str, Any]]:
    rows = []
    for domain, spec in MUSR_DOMAINS.items():
        path = root / "datasets" / str(spec["file"])
        source_rows = json.loads(path.read_text(encoding="utf-8"))
        for example_index, example in enumerate(source_rows):
            for question_index, question in enumerate(example["questions"]):
                upstream_id = f"{domain}:{example_index}:{question_index}"
                prompt = musr_prompt(str(example["context"]), question, domain)
                ident = identity("musr", upstream_id, prompt)
                rows.append(
                    {
                        "identity_sha256": ident,
                        "benchmark": "musr",
                        "stratum": domain,
                        "upstream_id": upstream_id,
                        "question": prompt,
                        "response_mode": "general",
                        "assessor": {
                            "answer": int(question["answer"]) + 1,
                            "choice_count": len(question["choices"]),
                            "domain": domain,
                        },
                    }
                )
    if len(rows) != 756:
        raise DenseBenchmarkDataError("MuSR flattened cardinality differs")
    return rows


def write_benchmark(
    output_root: Path,
    benchmark: str,
    rows: list[dict[str, Any]],
    training_hashes: set[str],
) -> dict[str, Any]:
    identities = [row["identity_sha256"] for row in rows]
    if len(set(identities)) != len(identities):
        raise DenseBenchmarkDataError(f"{benchmark} identities are duplicated")
    overlaps = [
        row["identity_sha256"]
        for row in rows
        if text_digest(row["question"]) in training_hashes
    ]
    if overlaps:
        raise DenseBenchmarkDataError(
            f"{benchmark} has {len(overlaps)} exact normalized training-source overlaps"
        )
    selected = ranked_screen(rows, SCREEN_ROWS, "stratum")
    directory = output_root / benchmark
    outputs = {}
    for split, materialized in (("full", rows), ("screen", selected)):
        questions = [
            {
                "schema": QUESTION_SCHEMA,
                "id": row["identity_sha256"],
                "benchmark": benchmark,
                "upstream_id": row["upstream_id"],
                "question": row["question"],
                "response_mode": row["response_mode"],
            }
            for row in materialized
        ]
        assessors = [
            {
                "schema": ASSESSOR_SCHEMA,
                "id": row["identity_sha256"],
                "benchmark": benchmark,
                "upstream_id": row["upstream_id"],
                "stratum": row["stratum"],
                "question_sha256": text_digest(row["question"]),
                "assessor": row["assessor"],
            }
            for row in materialized
        ]
        question_path = directory / f"{split}.questions.jsonl"
        assessor_path = directory / f"{split}.assessors.jsonl"
        outputs[split] = {
            "rows": len(materialized),
            "questions": str(question_path.resolve()),
            "questions_sha256": _atomic_lines(question_path, questions),
            "assessors": str(assessor_path.resolve()),
            "assessors_sha256": _atomic_lines(assessor_path, assessors),
            "strata": dict(Counter(row["stratum"] for row in materialized)),
        }
    return {
        "benchmark": benchmark,
        "full_rows": len(rows),
        "source_disjoint_method": "exact_normalized_prompt_sha256",
        "exact_training_source_overlaps": 0,
        "outputs": outputs,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists():
        raise DenseBenchmarkDataError("output root already exists")
    training_hashes, training_rows = training_prompt_hashes(args.training_source)
    benchmarks = {
        "mmlu_pro": load_mmlu(),
        "ifeval": load_ifeval(args.ifeval_input),
        "musr": load_musr(args.musr_root),
    }
    reports = {
        name: write_benchmark(args.output_root, name, rows, training_hashes)
        for name, rows in benchmarks.items()
    }
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "screen": {
            "label": "prospective_screen_not_full_benchmark",
            "rows_per_benchmark": SCREEN_ROWS,
            "seed": SCREEN_SEED,
        },
        "official_sources": {
            "mmlu_pro": {
                "repository_revision": MMLU_REPO_REVISION,
                "dataset_revision": MMLU_DATA_REVISION,
                "test_rows": 12032,
                "prompt": "official_five_shot_cot",
            },
            "ifeval": {
                "repository_revision": IFEVAL_REPO_REVISION,
                "input_sha256": sha256_file(args.ifeval_input),
                "rows": 541,
                "metrics": [
                    "strict_prompt",
                    "strict_instruction",
                    "loose_prompt",
                    "loose_instruction",
                ],
            },
            "musr": {
                "repository_revision": MUSR_REPO_REVISION,
                "rows": 756,
                "prompt": "official_cot_plus_zero_shot",
                "dataset_files": {
                    spec["file"]: sha256_file(
                        args.musr_root / "datasets" / spec["file"]
                    )
                    for spec in MUSR_DOMAINS.values()
                },
            },
        },
        "training_sources": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.training_source
        ],
        "training_source_rows": training_rows,
        "training_source_prompt_hashes": len(training_hashes),
        "benchmarks": reports,
        "model_visible_fields": ["question"],
        "assessor_visible_to_model": False,
    }
    _atomic_json(args.output_root / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ifeval-input", type=Path, required=True)
    parser.add_argument("--musr-root", type=Path, required=True)
    parser.add_argument("--training-source", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(
        json.dumps(
            {"status": report["status"], "benchmarks": list(report["benchmarks"])}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
