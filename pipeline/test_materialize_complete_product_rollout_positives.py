import hashlib
import json
from pathlib import Path

import pytest

from materialize_complete_product_rollout_positives import (
    CompleteRolloutPositiveError,
    materialize_complete_positives,
)


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for row in rows:
            encoded = (json.dumps(row) + "\n").encode()
            handle.write(encoded)
            digest.update(encoded)
    return digest.hexdigest()


def _candidate(
    identity: str,
    sample: int,
    *,
    correct: bool,
    exhausted: bool,
) -> dict:
    draft = f"reasoning {identity} {sample}. Final answer: 2"
    finalization = "Final answer: 2" if exhausted else None
    completion = f"{draft}\n\n{finalization}" if finalization else draft
    return {
        "schema": "shohin-hf-product-reasoning-rollouts-v1",
        "identity_sha256": identity,
        "question": f"Question {identity}",
        "training_group": "math",
        "task": "math500",
        "gold": "2",
        "sample_index": sample,
        "correct": correct,
        "explicit_final_answer": True,
        "draft_max_token_exhausted": exhausted,
        "draft_generated_tokens": 20 + sample,
        "generated_tokens": 20 + sample,
        "draft_completion": draft,
        "completion": completion,
        "finalization": finalization,
    }


def _report(path: Path, candidates: Path, digest: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "shohin-product-rollout-aggregate-v1",
                "status": "complete",
                "admitted": True,
                "candidates_output": str(candidates),
                "candidates_sha256": digest,
                "contract": {"adapter_checkpoint": "/model/checkpoint.pt"},
            }
        )
    )


def test_materializes_only_complete_autonomous_positive(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    digest = _write_jsonl(
        candidates,
        [
            _candidate("a", 0, correct=True, exhausted=True),
            _candidate("a", 1, correct=False, exhausted=False),
            _candidate("b", 0, correct=True, exhausted=False),
            _candidate("b", 1, correct=True, exhausted=False),
        ],
    )
    aggregate = tmp_path / "aggregate.json"
    _report(aggregate, candidates, digest)

    report = materialize_complete_positives(
        aggregate,
        tmp_path / "positives.jsonl",
        tmp_path / "report.json",
        min_positive_total=1,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "positives.jsonl").read_text().splitlines()
    ]
    assert [row["source_identity_sha256"] for row in rows] == ["b"]
    assert rows[0]["chosen_sample_index"] == 0
    assert report["positive_prompts"] == 1
    assert report["counters"]["prompts_without_complete_positive"] == 1


def test_rejects_candidate_hash_mismatch(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(candidates, [_candidate("a", 0, correct=True, exhausted=False)])
    aggregate = tmp_path / "aggregate.json"
    _report(aggregate, candidates, "0" * 64)
    with pytest.raises(CompleteRolloutPositiveError, match="hash differs"):
        materialize_complete_positives(
            aggregate,
            tmp_path / "positives.jsonl",
            tmp_path / "report.json",
            min_positive_total=1,
        )


def test_rejects_below_minimum(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    digest = _write_jsonl(
        candidates, [_candidate("a", 0, correct=True, exhausted=True)]
    )
    aggregate = tmp_path / "aggregate.json"
    _report(aggregate, candidates, digest)
    with pytest.raises(CompleteRolloutPositiveError, match="below minimum"):
        materialize_complete_positives(
            aggregate,
            tmp_path / "positives.jsonl",
            tmp_path / "report.json",
            min_positive_total=1,
        )
