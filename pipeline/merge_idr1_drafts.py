#!/usr/bin/env python3
"""Verify and merge the frozen IDR1 internal-draft shards."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROLLOUT_SCHEMA = "shohin-hf-product-reasoning-rollouts-v1"
MERGED_SCHEMA = "shohin-idr1-internal-drafts-v1"
RECEIPT_SCHEMA = "shohin-idr1-internal-drafts-receipt-v1"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
ADAPTER_SHA256 = "854a7cc44fbc2b54418f4e5bd09b7efeed0da44fc9ce217b0bb6b1997b722971"
SEED = 2026080818
SHARD_SIZE = 512


class IDR1DraftError(RuntimeError):
    """The IDR1 draft corpus differs from its frozen contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise IDR1DraftError(f"missing IDR1 input: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise IDR1DraftError(f"empty IDR1 input: {path}")
    return rows


def _atomic_lines(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise IDR1DraftError(f"refusing existing IDR1 output: {path}")
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
        raise IDR1DraftError(f"refusing existing IDR1 receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _question(row: dict[str, Any]) -> str:
    value = row.get("question") if row.get("question") is not None else row.get("text")
    if not isinstance(value, str) or not value.strip():
        raise IDR1DraftError("source question is empty")
    return value


def merge(
    reports: list[Path],
    banks: list[Path],
    output: Path,
    receipt_path: Path,
    *,
    model_revision: str = MODEL_REVISION,
    adapter_sha256: str | None = ADAPTER_SHA256,
) -> dict[str, Any]:
    if not model_revision or (
        adapter_sha256 is not None and len(adapter_sha256) != 64
    ):
        raise IDR1DraftError("IDR1 model provenance is invalid")
    if len(reports) != 17 or len(banks) != 3:
        raise IDR1DraftError("IDR1 requires exactly 17 reports and three banks")
    bank_rows = {sha256_file(path): _load_jsonl(path) for path in banks}
    if sorted(len(rows) for rows in bank_rows.values()) != [200, 4096, 4096]:
        raise IDR1DraftError("IDR1 bank geometry differs")
    expected_ids: set[str] = set()
    for rows in bank_rows.values():
        for row in rows:
            identity = row.get("identity_sha256")
            if not isinstance(identity, str) or len(identity) != 64:
                raise IDR1DraftError("source identity is invalid")
            if identity in expected_ids:
                raise IDR1DraftError("source identity is duplicated")
            expected_ids.add(identity)

    merged_by_id: dict[str, dict[str, Any]] = {}
    seen_slices: set[tuple[str, int, int]] = set()
    input_receipts: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for report_path in reports:
        if not report_path.is_file():
            raise IDR1DraftError(f"missing shard report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        required = {
            "schema": ROLLOUT_SCHEMA,
            "status": "complete",
            "model_revision": model_revision,
            "adapter_checkpoint_sha256": adapter_sha256,
            "samples": 1,
            "generation_mode": "greedy",
            "prompt_batch_size": 4,
            "finalize_exhausted": False,
            "enable_thinking": False,
            "bare_prompt_style": "reasoning",
            "seed": SEED,
            "max_new_tokens": 768,
        }
        for field, value in required.items():
            if report.get(field) != value:
                raise IDR1DraftError(f"draft report field differs: {field}")
        data_hash = report.get("data_sha256")
        rows = bank_rows.get(data_hash)
        if rows is None:
            raise IDR1DraftError("draft report references an unknown bank")
        skip, count = report.get("skip"), report.get("count")
        if not isinstance(skip, int) or not isinstance(count, int):
            raise IDR1DraftError("draft slice geometry is invalid")
        if count not in (200, SHARD_SIZE) or skip < 0 or skip + count > len(rows):
            raise IDR1DraftError("draft slice is outside its bank")
        if len(rows) == 4096 and (count != SHARD_SIZE or skip % SHARD_SIZE):
            raise IDR1DraftError("large-bank draft shard differs")
        if len(rows) == 200 and (skip, count) != (0, 200):
            raise IDR1DraftError("code draft shard differs")
        slice_key = (str(data_hash), skip, count)
        if slice_key in seen_slices:
            raise IDR1DraftError("draft slice is duplicated")
        seen_slices.add(slice_key)

        candidates_path = Path(str(report.get("candidates_output", "")))
        positives_path = Path(str(report.get("positives_output", "")))
        if sha256_file(candidates_path) != report.get("candidates_sha256"):
            raise IDR1DraftError("candidate hash differs")
        if sha256_file(positives_path) != report.get("positives_sha256"):
            raise IDR1DraftError("positive hash differs")
        candidates = _load_jsonl(candidates_path)
        source_slice = rows[skip : skip + count]
        if len(candidates) != count:
            raise IDR1DraftError("candidate cardinality differs")
        if report.get("counters", {}).get("prompts") != count:
            raise IDR1DraftError("report prompt count differs")
        for source, candidate in zip(source_slice, candidates, strict=True):
            identity = str(source["identity_sha256"])
            if candidate.get("schema") != ROLLOUT_SCHEMA:
                raise IDR1DraftError("candidate schema differs")
            if candidate.get("identity_sha256") != identity:
                raise IDR1DraftError("candidate/source identity order differs")
            if candidate.get("task") != source.get("task"):
                raise IDR1DraftError("candidate/source task differs")
            if candidate.get("question") != _question(source):
                raise IDR1DraftError("candidate/source question differs")
            completion = candidate.get("completion")
            if not isinstance(completion, str):
                raise IDR1DraftError("internal draft is not text")
            if candidate.get("sample_index") != 0:
                raise IDR1DraftError("internal draft sample index differs")
            if identity in merged_by_id:
                raise IDR1DraftError("internal draft identity is duplicated")
            merged_by_id[identity] = {
                "schema": MERGED_SCHEMA,
                "identity_sha256": identity,
                "task": source["task"],
                "question": candidate["question"],
                "completion": completion,
                "correct": bool(candidate.get("correct")),
                "prediction": candidate.get("prediction"),
                "generated_tokens": candidate.get("generated_tokens"),
                "max_token_exhausted": bool(candidate.get("max_token_exhausted")),
                "source_report_sha256": sha256_file(report_path),
            }
            counters["rows"] += 1
            counters["correct"] += int(bool(candidate.get("correct")))
            counters["max_token_exhausted"] += int(
                bool(candidate.get("max_token_exhausted"))
            )
            counters["empty_drafts"] += int(not completion.strip())
            counters[str(source["task"])] += 1
        input_receipts.append(
            {
                "report": str(report_path.resolve()),
                "report_sha256": sha256_file(report_path),
                "data_sha256": data_hash,
                "skip": skip,
                "count": count,
                "candidates_sha256": report["candidates_sha256"],
                "positives_sha256": report["positives_sha256"],
            }
        )

    if set(merged_by_id) != expected_ids or len(merged_by_id) != 8392:
        raise IDR1DraftError("internal draft coverage is incomplete")
    ordered: list[dict[str, Any]] = []
    for path in banks:
        for source in _load_jsonl(path):
            ordered.append(merged_by_id[str(source["identity_sha256"])])
    output_sha256 = _atomic_lines(output, ordered)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "complete",
        "model_revision": model_revision,
        "adapter_checkpoint_sha256": adapter_sha256,
        "generation": {
            "mode": "greedy",
            "prompt_batch_size": 4,
            "samples": 1,
            "seed": SEED,
            "max_new_tokens": 768,
            "enable_thinking": False,
        },
        "banks": [
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "rows": len(_load_jsonl(path)),
            }
            for path in banks
        ],
        "inputs": sorted(
            input_receipts, key=lambda item: (item["data_sha256"], item["skip"])
        ),
        "counters": dict(sorted(counters.items())),
        "output": str(output.resolve()),
        "output_sha256": output_sha256,
        "unique_identities": len(merged_by_id),
        "exact_bank_coverage": True,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", dest="reports", action="append", type=Path, required=True
    )
    parser.add_argument(
        "--bank", dest="banks", action="append", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--adapter-sha256", default=ADAPTER_SHA256)
    parser.add_argument(
        "--base-model-draft",
        action="store_true",
        help="Require source reports generated without an adapter checkpoint.",
    )
    args = parser.parse_args()
    report = merge(
        args.reports,
        args.banks,
        args.output,
        args.receipt,
        model_revision=args.model_revision,
        adapter_sha256=None if args.base_model_draft else args.adapter_sha256,
    )
    print(
        json.dumps(
            {"rows": report["unique_identities"], "sha256": report["output_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
