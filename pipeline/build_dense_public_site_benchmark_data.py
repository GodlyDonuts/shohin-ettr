#!/usr/bin/env python3
"""Materialize additional official website benchmark boards and hidden assessors."""

from __future__ import annotations

import argparse
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as parquet

QUESTION_SCHEMA = "shohin-dense-public-benchmark-question-v1"
ASSESSOR_SCHEMA = "shohin-dense-public-benchmark-assessor-v1"
REPORT_SCHEMA = "shohin-dense-public-site-data-v1"
LONGBENCH_ADMITTED_LENGTHS = {"8k", "16k", "32k", "64k"}
LIVEBENCH_RELEASE = "2024-11-25"
LIVEBENCH_REVISIONS = {
    "coding": "a958549fdd8aa57be0a3fafe7b205ffc160ed5f4",
    "data_analysis": "31b9661ff678df9958e2f7fa228427f4c858c1a1",
    "instruction_following": "0868379c4b5cf62aeacaf8be4f08fced815c81bb",
    "math": "bb66571c8ccf32d3df9e6f48b920d3770ff4aacb",
    "reasoning": "6fc6498a5dfba553f69f4413feabade1f1a2d384",
    "language": "3ada32a2e53d5e04e57fa503384cb85ce9116c40",
}


class SiteBenchmarkDataError(RuntimeError):
    """An upstream dataset, prompt, identity, or output differs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(value: str) -> str:
    return " ".join(value.split())


def identity(benchmark: str, upstream_id: str, prompt: str) -> str:
    return hashlib.sha256(
        f"{benchmark}\0{upstream_id}\0{hashlib.sha256(normalized_text(prompt).encode()).hexdigest()}".encode()
    ).hexdigest()


def lcb_prompt(row: dict[str, Any]) -> str:
    prompt = f"### Question:\n{row['question_content']}\n\n"
    if row.get("starter_code"):
        prompt += (
            "### Format: You will use the following starter code to write the solution "
            "to the problem and enclose your code within delimiters.\n"
            f"```python\n{row['starter_code']}\n```\n\n"
        )
    else:
        prompt += (
            "### Format: Read the inputs from stdin, solve the problem, and write the "
            "answer to stdout. Enclose your code in one Python code block.\n"
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    return prompt + "### Answer: (use the provided format with backticks)\n"


def load_humaneval(path: Path) -> list[dict[str, Any]]:
    source = parquet.read_table(path).to_pylist()
    if len(source) != 164:
        raise SiteBenchmarkDataError("HumanEval+ cardinality differs")
    rows = []
    for row in source:
        prompt = (
            "Complete the following Python function. Return only the complete executable "
            "Python code, without Markdown fences.\n\n" + str(row["prompt"])
        )
        rows.append(make_row("humaneval_plus", str(row["task_id"]), prompt, "code", "all", row))
    return rows


def load_mbpp(path: Path) -> list[dict[str, Any]]:
    source = parquet.read_table(path).to_pylist()
    if len(source) != 378:
        raise SiteBenchmarkDataError("MBPP+ cardinality differs")
    rows = []
    for row in source:
        prompt = (
            "Write a complete Python solution for the following task. Return only "
            "executable Python code, without Markdown fences.\n\n" + str(row["prompt"])
        )
        rows.append(make_row("mbpp_plus", str(row["task_id"]), prompt, "code", "all", row))
    return rows


def load_correctbench(path: Path) -> list[dict[str, Any]]:
    source = parquet.read_table(path).to_pylist()
    if len(source) != 739:
        raise SiteBenchmarkDataError("CorrectBench cardinality differs")
    return [
        make_row(
            "correctbench",
            str(index),
            str(row["question"]),
            "general",
            "all",
            {"answer": str(row["answer"])},
        )
        for index, row in enumerate(source)
    ]


def load_livecodebench(paths: list[Path]) -> list[dict[str, Any]]:
    source = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            source.extend(json.loads(line) for line in handle if line.strip())
    if len(source) != 1055:
        raise SiteBenchmarkDataError("LiveCodeBench release_v6 cardinality differs")
    rows = []
    for row in source:
        rows.append(
            make_row(
                "livecodebench",
                str(row["question_id"]),
                lcb_prompt(row),
                "code",
                str(row["difficulty"]),
                row,
            )
        )
    return rows


def load_longbench(path: Path) -> list[dict[str, Any]]:
    source = json.loads(path.read_text(encoding="utf-8"))
    if len(source) != 1500:
        raise SiteBenchmarkDataError("LongBench Pro cardinality differs")
    admitted = [row for row in source if row["token_length"] in LONGBENCH_ADMITTED_LENGTHS]
    if len(admitted) != 1000:
        raise SiteBenchmarkDataError("LongBench Pro <=64k subset cardinality differs")
    rows = []
    for row in admitted:
        prompt = f"{row['context']}\n\n\n\n{row['question_nonthinking']}"
        assessor = {key: value for key, value in row.items() if key != "context"}
        rows.append(
            make_row(
                "longbench_pro",
                str(row["id"]),
                prompt,
                "general",
                str(row["token_length"]),
                assessor,
            )
        )
    return rows


def json_safe(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def load_livebench() -> list[dict[str, Any]]:
    from datasets import load_dataset

    rows = []
    for category, revision in LIVEBENCH_REVISIONS.items():
        source = load_dataset(
            f"livebench/{category}", split="test", revision=revision
        )
        for raw in source:
            row = json_safe(dict(raw))
            released = str(row["livebench_release_date"])[:10]
            removed = str(row.get("livebench_removal_date", ""))[:10]
            if released > LIVEBENCH_RELEASE or (removed and removed <= LIVEBENCH_RELEASE):
                continue
            turns = row.get("turns")
            if not isinstance(turns, list) or len(turns) != 1 or not str(turns[0]).strip():
                raise SiteBenchmarkDataError("LiveBench 2024-11-25 turn geometry differs")
            rows.append(
                make_row(
                    "livebench",
                    str(row["question_id"]),
                    str(turns[0]),
                    "code" if category == "coding" else "general",
                    category,
                    row,
                )
            )
    if len(rows) != 1000:
        raise SiteBenchmarkDataError("LiveBench 2024-11-25 cardinality differs")
    return rows


def make_row(
    benchmark: str,
    upstream_id: str,
    prompt: str,
    response_mode: str,
    stratum: str,
    assessor: dict[str, Any],
) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "upstream_id": upstream_id,
        "id": identity(benchmark, upstream_id, prompt),
        "question": prompt,
        "response_mode": response_mode,
        "stratum": stratum,
        "assessor": assessor,
    }


def atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise SiteBenchmarkDataError(f"refusing to replace {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            encoded = json.dumps(
                row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode() + b"\n"
            digest.update(encoded)
            handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return digest.hexdigest()


def write_benchmark(root: Path, name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SiteBenchmarkDataError(f"{name} identities are duplicated")
    questions = [
        {
            "schema": QUESTION_SCHEMA,
            "id": row["id"],
            "benchmark": name,
            "upstream_id": row["upstream_id"],
            "question": row["question"],
            "response_mode": row["response_mode"],
        }
        for row in rows
    ]
    assessors = [
        {
            "schema": ASSESSOR_SCHEMA,
            "id": row["id"],
            "benchmark": name,
            "upstream_id": row["upstream_id"],
            "stratum": row["stratum"],
            "assessor": row["assessor"],
        }
        for row in rows
    ]
    directory = root / name
    question_path = directory / "full.questions.jsonl"
    assessor_path = directory / "full.assessors.jsonl"
    return {
        "rows": len(rows),
        "questions": str(question_path.resolve()),
        "questions_sha256": atomic_lines(question_path, questions),
        "assessors": str(assessor_path.resolve()),
        "assessors_sha256": atomic_lines(assessor_path, assessors),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise SiteBenchmarkDataError("refusing to replace site data report")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_root.exists():
        raise SiteBenchmarkDataError("site benchmark output root already exists")
    loaders = {
        "humaneval_plus": load_humaneval(args.humaneval_parquet),
        "mbpp_plus": load_mbpp(args.mbpp_parquet),
        "correctbench": load_correctbench(args.correctbench_parquet),
        "livebench": load_livebench(),
        "livecodebench": load_livecodebench(args.livecodebench_jsonl),
        "longbench_pro": load_longbench(args.longbench_json),
    }
    outputs = {
        name: write_benchmark(args.output_root, name, rows)
        for name, rows in loaders.items()
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "benchmarks": outputs,
        "source_files": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in [
                args.humaneval_parquet,
                args.mbpp_parquet,
                args.correctbench_parquet,
                *args.livecodebench_jsonl,
                args.longbench_json,
            ]
        ],
        "longbench_scope": "official_nonthinking_8k_to_64k_subset_not_full_8k_to_256k",
        "livebench_release": LIVEBENCH_RELEASE,
        "livebench_dataset_revisions": LIVEBENCH_REVISIONS,
        "assessors_visible_to_model": False,
    }
    atomic_json(args.output_root / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--humaneval-parquet", type=Path, required=True)
    parser.add_argument("--mbpp-parquet", type=Path, required=True)
    parser.add_argument("--correctbench-parquet", type=Path, required=True)
    parser.add_argument("--livecodebench-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--longbench-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    report = run(parse_args())
    print(json.dumps({name: row["rows"] for name, row in report["benchmarks"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
