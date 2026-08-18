from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.analyze_mixtral_revision_format import (
    ARMS,
    CANDIDATE_SCHEMA,
    SCORE_SCHEMA,
    FormatAnalysisError,
    analyze,
    canonical_sha256,
    sha256_file,
)

TASKS = ("bbh_logic", "math500", "mbpp")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    identities = [f"{index + 1:064x}" for index in range(len(TASKS))]
    outcomes = []
    for identity, task in zip(identities, TASKS, strict=True):
        outcomes.append(
            {
                "identity_sha256": identity,
                "task": task,
                "correct": {
                    "unchanged": True,
                    "self_refinement": task != "mbpp",
                    "revision": task != "mbpp",
                },
            }
        )
    score_path = tmp_path / "score.json"
    write_json(
        score_path,
        {
            "schema": SCORE_SCHEMA,
            "status": "complete",
            "rows": len(TASKS),
            "outcomes": outcomes,
        },
    )

    candidates_root = tmp_path / "candidates"
    receipts = []
    for arm in ARMS:
        rows = []
        for identity, task in zip(identities, TASKS, strict=True):
            if arm == "revision":
                completion = "\\boxed{answer}"
            elif task == "mbpp":
                completion = "```python\ndef solve():\n    return 1\n```"
            else:
                completion = "A reasoned answer"
            rows.append(
                {
                    "schema": CANDIDATE_SCHEMA,
                    "arm": arm,
                    "identity_sha256": identity,
                    "task": task,
                    "completion": completion,
                    "generated_tokens": len(completion.split()),
                    "max_token_exhausted": False,
                }
            )
        shard_root = candidates_root / arm / "shard_00"
        shard_root.mkdir(parents=True)
        candidate_path = shard_root / "candidates.jsonl"
        candidate_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
        write_json(shard_root / "report.json", {"status": "complete"})
        receipts.append(
            {
                "path": f"{arm}/shard_00/candidates.jsonl",
                "rows": len(rows),
                "sha256": sha256_file(candidate_path),
            }
        )
    return score_path, candidates_root, canonical_sha256(receipts)


def run_fixture(tmp_path: Path) -> dict[str, object]:
    score_path, candidates_root, receipts_sha256 = fixture(tmp_path)
    return analyze(
        score_path,
        candidates_root,
        expected_score_sha256=sha256_file(score_path),
        expected_candidate_receipts_sha256=receipts_sha256,
        expected_rows=3,
        expected_task_counts={task: 1 for task in TASKS},
        expected_shards=1,
    )


def test_analyze_replays_identity_join_and_format_collapse(tmp_path: Path) -> None:
    report = run_fixture(tmp_path)
    assert report["status"] == "complete"
    assert report["source"]["ordered_identity_replay"] == "pass"
    assert report["metrics"]["revision"]["all"]["boxed_completions"] == 3
    assert report["metrics"]["revision"]["mbpp"]["correct"] == 0
    assert report["metrics"]["unchanged"]["mbpp"]["code_fenced_completions"] == 1


def test_analyze_rejects_candidate_projection_tamper(tmp_path: Path) -> None:
    score_path, candidates_root, receipts_sha256 = fixture(tmp_path)
    candidate_path = candidates_root / "revision" / "shard_00" / "candidates.jsonl"
    candidate_path.write_text(candidate_path.read_text().replace("math500", "mbpp", 1))
    with pytest.raises(FormatAnalysisError, match="candidate row differs"):
        analyze(
            score_path,
            candidates_root,
            expected_score_sha256=sha256_file(score_path),
            expected_candidate_receipts_sha256=receipts_sha256,
            expected_rows=3,
            expected_task_counts={task: 1 for task in TASKS},
            expected_shards=1,
        )


def test_analyze_rejects_score_hash_tamper(tmp_path: Path) -> None:
    score_path, candidates_root, receipts_sha256 = fixture(tmp_path)
    with pytest.raises(FormatAnalysisError, match="score SHA-256 differs"):
        analyze(
            score_path,
            candidates_root,
            expected_score_sha256="0" * 64,
            expected_candidate_receipts_sha256=receipts_sha256,
            expected_rows=3,
            expected_task_counts={task: 1 for task in TASKS},
            expected_shards=1,
        )
