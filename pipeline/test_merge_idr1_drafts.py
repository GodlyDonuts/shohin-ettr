from __future__ import annotations

import hashlib
import json
from pathlib import Path

from merge_idr1_drafts import (
    ADAPTER_SHA256,
    MODEL_REVISION,
    ROLLOUT_SCHEMA,
    SEED,
    merge,
)


def _write_lines(path: Path, rows: list[dict]) -> str:
    data = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_frozen_idr1_draft_constants() -> None:
    assert len(ADAPTER_SHA256) == 64
    assert MODEL_REVISION == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert SEED == 2026080818


def test_merge_accepts_exact_8392_row_geometry(tmp_path: Path) -> None:
    banks: list[Path] = []
    reports: list[Path] = []
    identity_index = 0
    for domain, rows_count, task in (
        ("math", 4096, "math500"),
        ("science", 4096, "bbh_logic"),
        ("code", 200, "mbpp"),
    ):
        rows = []
        for _ in range(rows_count):
            identity = f"{identity_index:064x}"
            identity_index += 1
            row = {"identity_sha256": identity, "task": task}
            if task == "mbpp":
                row["text"] = f"code question {identity}"
            else:
                row["question"] = f"question {identity}"
            rows.append(row)
        bank = tmp_path / f"{domain}.jsonl"
        bank_hash = _write_lines(bank, rows)
        banks.append(bank)
        shard_count = 1 if rows_count == 200 else 8
        for shard in range(shard_count):
            skip = 0 if rows_count == 200 else shard * 512
            count = rows_count if rows_count == 200 else 512
            candidates = []
            for source in rows[skip : skip + count]:
                question = source.get("question", source.get("text"))
                candidates.append(
                    {
                        "schema": ROLLOUT_SCHEMA,
                        "identity_sha256": source["identity_sha256"],
                        "task": task,
                        "question": question,
                        "sample_index": 0,
                        "completion": f"draft {source['identity_sha256']}",
                        "correct": False,
                        "prediction": None,
                        "generated_tokens": 4,
                        "max_token_exhausted": False,
                    }
                )
            candidate_path = tmp_path / f"{domain}-{shard}.candidates.jsonl"
            positive_path = tmp_path / f"{domain}-{shard}.positives.jsonl"
            candidate_hash = _write_lines(candidate_path, candidates)
            positive_hash = _write_lines(positive_path, [])
            report = {
                "schema": ROLLOUT_SCHEMA,
                "status": "complete",
                "model_revision": MODEL_REVISION,
                "adapter_checkpoint_sha256": ADAPTER_SHA256,
                "samples": 1,
                "generation_mode": "greedy",
                "prompt_batch_size": 4,
                "finalize_exhausted": False,
                "enable_thinking": False,
                "bare_prompt_style": "reasoning",
                "seed": SEED,
                "max_new_tokens": 768,
                "data_sha256": bank_hash,
                "skip": skip,
                "count": count,
                "counters": {"prompts": count},
                "candidates_output": str(candidate_path),
                "candidates_sha256": candidate_hash,
                "positives_output": str(positive_path),
                "positives_sha256": positive_hash,
            }
            report_path = tmp_path / f"{domain}-{shard}.report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            reports.append(report_path)

    output = tmp_path / "merged.jsonl"
    receipt_path = tmp_path / "receipt.json"
    receipt = merge(reports, banks, output, receipt_path)
    assert receipt["unique_identities"] == 8392
    assert receipt["exact_bank_coverage"]
    assert receipt["counters"]["rows"] == 8392
    assert len(output.read_text(encoding="utf-8").splitlines()) == 8392
