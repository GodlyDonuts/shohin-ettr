#!/usr/bin/env python3
"""Fetch and normalize the exact official GPQA-Diamond evaluation board."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any
from urllib.request import urlopen
import zipfile


GPQA_REPOSITORY = "idavidrein/gpqa"
GPQA_REVISION = "56686c06f5e19865c153de0fdb11be3890014df7"
GPQA_ARCHIVE_URL = (
    "https://raw.githubusercontent.com/"
    f"{GPQA_REPOSITORY}/{GPQA_REVISION}/dataset.zip"
)
GPQA_ARCHIVE_SHA256 = "461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc"
GPQA_ARCHIVE_PASSWORD = b"deserted-untie-orchid"
GPQA_MEMBER = "dataset/gpqa_diamond.csv"
CHOICE_LABELS = "ABCD"


class GPQAFetchError(RuntimeError):
    """The GPQA board fetch or publication contract was violated."""


def _canonical_json(row: dict[str, Any]) -> bytes:
    return json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode() + b"\n"


def normalize_gpqa(rows: list[dict[str, str]], seed: int) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        record_id = row.get("Record ID", "").strip()
        question = row.get("Question", "").strip()
        correct = row.get("Correct Answer", "").strip()
        incorrect = [
            row.get(f"Incorrect Answer {index}", "").strip()
            for index in range(1, 4)
        ]
        answers = [correct, *incorrect]
        if not record_id or not question or any(not answer for answer in answers):
            raise GPQAFetchError("GPQA row is missing a required public field")
        if correct in incorrect:
            raise GPQAFetchError(
                f"GPQA row {record_id} duplicates the correct answer as a distractor"
            )
        ordered = sorted(
            answers,
            key=lambda answer: hashlib.sha256(
                f"{seed}\0{record_id}\0{answer}".encode()
            ).hexdigest(),
        )
        answer_index = ordered.index(correct)
        normalized.append(
            {
                "id": record_id,
                "question": question,
                "choices": [
                    {"label": label, "text": answer}
                    for label, answer in zip(CHOICE_LABELS, ordered, strict=True)
                ],
                "answer": CHOICE_LABELS[answer_index],
                "domain": row.get("High-level domain", "").strip(),
                "subdomain": row.get("Subdomain", "").strip(),
                "source": GPQA_REPOSITORY,
                "source_revision": GPQA_REVISION,
            }
        )
    normalized.sort(key=lambda item: item["id"])
    if len(normalized) != 198:
        raise GPQAFetchError("GPQA-Diamond does not contain exactly 198 rows")
    if len({row["id"] for row in normalized}) != len(normalized):
        raise GPQAFetchError("GPQA-Diamond contains duplicate record identifiers")
    return normalized


def fetch(output: Path, report_path: Path, seed: int) -> dict[str, Any]:
    if output.exists() or report_path.exists():
        raise GPQAFetchError("refusing to replace GPQA output")
    with urlopen(GPQA_ARCHIVE_URL, timeout=120) as response:
        archive = response.read()
    archive_sha256 = hashlib.sha256(archive).hexdigest()
    if archive_sha256 != GPQA_ARCHIVE_SHA256:
        raise GPQAFetchError("GPQA archive hash differs")
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        source = bundle.read(GPQA_MEMBER, pwd=GPQA_ARCHIVE_PASSWORD)
    rows = list(csv.DictReader(io.StringIO(source.decode("utf-8"))))
    normalized = normalize_gpqa(rows, seed)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    output_digest = hashlib.sha256()
    with temporary.open("wb") as handle:
        for row in normalized:
            payload = _canonical_json(row)
            handle.write(payload)
            output_digest.update(payload)
    os.replace(temporary, output)
    report = {
        "schema": "shohin-gpqa-diamond-eval-board-v1",
        "status": "complete",
        "repository": GPQA_REPOSITORY,
        "revision": GPQA_REVISION,
        "archive_url": GPQA_ARCHIVE_URL,
        "archive_sha256": archive_sha256,
        "member": GPQA_MEMBER,
        "member_sha256": hashlib.sha256(source).hexdigest(),
        "choice_seed": seed,
        "answer_distribution": dict(
            sorted(Counter(row["answer"] for row in normalized).items())
        ),
        "duplicate_distractor_rows": sum(
            len({choice["text"] for choice in row["choices"]}) < 4
            for row in normalized
        ),
        "rows": len(normalized),
        "output": str(output.resolve()),
        "output_sha256": output_digest.hexdigest(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_report, report_path)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = fetch(args.output, args.report, args.seed)
    print(
        f"[gpqa-fetch] rows={report['rows']} sha256={report['output_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
