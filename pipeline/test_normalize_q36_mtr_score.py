from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from normalize_q36_mtr_score import ARM_SCHEMA, normalize
from q36_mtr_contract import MODEL_REVISION
from score_q36_mtr import SCORE_SCHEMA, build_publication_analysis


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_normalize_q36_score_emits_five_bound_arms(tmp_path: Path) -> None:
    data_sha = _sha("data")
    identity_sha = _sha("identities")
    run_id = "q36-test"
    metrics = {
        arm: {
            "overall": {"correct": 500, "total": 1289},
            "math500": {"correct": 200, "total": 500},
            "bbh_logic": {"correct": 200, "total": 500},
            "mbpp": {"correct": 100, "total": 289},
        }
        for arm in (
            "learned_commit",
            "revision",
            "unchanged",
            "self_refinement",
            "draft_hidden",
        )
    }
    score = {
        "schema": SCORE_SCHEMA,
        "status": "complete",
        "run_id": run_id,
        "model_revision": MODEL_REVISION,
        "rows": 1289,
        "outcome_rows": 1289,
        "identity_order_sha256": identity_sha,
        "metrics": metrics,
        "retention": {
            "revision_correct": {"retained": 490, "total": 500},
            "unchanged_correct": {"retained": 490, "total": 500},
        },
        "order_consistency": {"consistent": 1289, "total": 1289},
        "publication_analysis": build_publication_analysis(
            [
                {
                    "identity_sha256": hashlib.sha256(
                        f"normalize-publication-{index}".encode()
                    ).hexdigest(),
                    "task": ("math500", "bbh_logic", "mbpp")[index % 3],
                    "correct": {arm: index < 500 for arm in metrics},
                }
                for index in range(1_289)
            ]
        ),
        "empty_completion_counts": {arm: 0 for arm in metrics},
        "capability_policy_rejection_counts": {arm: 0 for arm in metrics},
        "malformed_completion_counts": {arm: 0 for arm in metrics},
        "generation_truncation_counts": {arm: 0 for arm in metrics},
        "commit_malformed": 0,
        "commit_prompt_truncated": 0,
        "commit_training_prompt_truncated": 0,
        "assessor_semantic_reads": 1,
        "score_consumption_state": "consumed",
        "input_hashes": {"development_data_sha256": data_sha},
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    score_path = tmp_path / "score.json"
    score_path.write_text(json.dumps(score) + "\n")
    custody = {
        "schema": "shohin-q36-mtr-precompute-custody-v1",
        "status": "complete",
        "run_id": run_id,
        "model_revision": MODEL_REVISION,
        "identity_order_sha256": identity_sha,
        "data_sha256": data_sha,
        "runtime_sha256": _sha("runtime"),
    }
    custody_path = tmp_path / "custody.json"
    custody_path.write_text(json.dumps(custody) + "\n")
    output = tmp_path / "arms"
    receipt = normalize(
        argparse.Namespace(
            score_report=score_path,
            precompute_custody=custody_path,
            output=output,
        )
    )
    assert set(receipt["arms"]) == {
        "learned_commit",
        "trained_revision",
        "unchanged",
        "self_refinement",
        "draft_hidden",
    }
    for path in output.iterdir():
        report = json.loads(path.read_text())
        assert report["schema"] == ARM_SCHEMA
        assert report["candidate_count"] == 1289
        assert report["malformed_count"] == 0
        if report["arm"] == "learned_commit":
            assert report["publication_analysis"]["status"] == (
                "descriptive_non_gating"
            )
