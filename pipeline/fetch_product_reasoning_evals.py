#!/usr/bin/env python3
"""Fetch exact-revision product reasoning development boards.

The public benchmark rows are evaluator inputs only. Every output is written
atomically, the source revision is explicit, and reruns refuse replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.request import urlopen


AIME_REPOSITORY = "HuggingFaceH4/aime_2024"
AIME_REVISION = "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"
BBH_REPOSITORY = "suzgunmirac/BIG-Bench-Hard"
BBH_REVISION = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"
BBH_TASKS = (
    "boolean_expressions",
    "formal_fallacies",
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "web_of_lies",
)


class ProductEvalFetchError(RuntimeError):
    """A benchmark fetch or publication contract was violated."""


def _canonical_json(row: dict[str, Any]) -> bytes:
    return json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode() + b"\n"


def _publish_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise ProductEvalFetchError(f"refusing to replace benchmark board: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in rows:
            payload = _canonical_json(row)
            handle.write(payload)
            digest.update(payload)
    os.replace(temporary, path)
    return digest.hexdigest()


def normalize_aime(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        problem = str(row.get("problem", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not problem or not answer:
            continue
        normalized.append(
            {
                "id": int(row["id"]),
                "problem": problem,
                "answer": answer,
                "year": str(row.get("year", "2024")),
                "source": AIME_REPOSITORY,
                "source_revision": AIME_REVISION,
            }
        )
    normalized.sort(key=lambda item: item["id"])
    if len(normalized) != 30:
        raise ProductEvalFetchError("AIME-2024 does not contain exactly 30 rows")
    return normalized


def normalize_bbh(
    task: str, examples: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    normalized = []
    for index, row in enumerate(examples):
        prompt = str(row.get("input", "")).strip()
        target = str(row.get("target", "")).strip()
        if not prompt or not target:
            continue
        normalized.append(
            {
                "id": f"{task}:{index:04d}",
                "task": task,
                "input": prompt,
                "target": target,
                "source": BBH_REPOSITORY,
                "source_revision": BBH_REVISION,
            }
        )
    if len(normalized) < 100:
        raise ProductEvalFetchError(f"BBH task {task} has too few rows")
    return normalized


def fetch(args: argparse.Namespace) -> dict[str, Any]:
    from datasets import load_dataset

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ProductEvalFetchError("benchmark output directory is not empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aime_source = load_dataset(
        AIME_REPOSITORY,
        split="train",
        revision=AIME_REVISION,
        cache_dir=str(args.cache_dir),
    )
    aime = normalize_aime([dict(row) for row in aime_source])

    bbh: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for task in BBH_TASKS:
        url = (
            "https://raw.githubusercontent.com/"
            f"{BBH_REPOSITORY}/{BBH_REVISION}/bbh/{task}.json"
        )
        with urlopen(url, timeout=120) as response:
            payload = response.read()
        source_hashes[task] = hashlib.sha256(payload).hexdigest()
        document = json.loads(payload)
        bbh.extend(normalize_bbh(task, list(document["examples"])))

    aime_path = args.output_dir / "aime2024.jsonl"
    bbh_path = args.output_dir / "bbh_logic.jsonl"
    report_path = args.output_dir / "product_eval_boards.report.json"
    report = {
        "schema": "shohin-product-reasoning-eval-boards-v1",
        "aime": {
            "repository": AIME_REPOSITORY,
            "revision": AIME_REVISION,
            "rows": len(aime),
            "output": str(aime_path.resolve()),
            "sha256": _publish_jsonl(aime_path, aime),
        },
        "bbh_logic": {
            "repository": BBH_REPOSITORY,
            "revision": BBH_REVISION,
            "tasks": list(BBH_TASKS),
            "source_sha256": source_hashes,
            "rows": len(bbh),
            "output": str(bbh_path.resolve()),
            "sha256": _publish_jsonl(bbh_path, bbh),
        },
    }
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    report = fetch(parse_args())
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
