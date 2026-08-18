import hashlib
import inspect
import json
from pathlib import Path

import pytest

import train_apply_mixtral_8x22b_commit as module


def _candidate(identity: str, arm: str, task: str = "math500") -> dict:
    return {
        "schema": module.CANDIDATE_SCHEMA,
        "arm": arm,
        "identity_sha256": identity,
        "task": task,
        "completion": "The answer is 7.",
        "generated_tokens": 5,
        "max_token_exhausted": False,
    }


def test_feature_projection_cannot_accept_task_or_benchmark_route() -> None:
    assert list(inspect.signature(module.candidate_features).parameters) == [
        "source_prompt",
        "arm",
        "candidate",
    ]
    candidate = _candidate("a" * 64, "revision")
    first = module.candidate_features(
        "Write a function returning 7.", "revision", candidate
    )
    candidate["task"] = "mbpp"
    second = module.candidate_features(
        "Write a function returning 7.", "revision", candidate
    )
    assert first == second


def test_fit_and_selection_are_deterministic() -> None:
    rows = []
    for index in range(12):
        candidates = [_candidate(f"{index:064x}", arm) for arm in module.ARMS]
        rows.append(
            {
                "features": [
                    module.candidate_features(f"Question {index}", arm, candidate)
                    for arm, candidate in zip(module.ARMS, candidates, strict=True)
                ],
                "correct": [index % 3 == arm_index for arm_index in range(3)],
            }
        )
    first = module.fit(rows)
    second = module.fit(rows)
    assert first == second
    assert module.select_arm(first, rows[0]["features"]) == module.select_arm(
        second, rows[0]["features"]
    )


def test_validation_shards_use_64_row_prefix_and_63_row_tail(tmp_path: Path) -> None:
    sources = [
        {
            "identity_sha256": f"{index:064x}",
            "task": "math500",
            "source_prompt": f"Question {index}",
        }
        for index in range(module.VALIDATION_ROWS)
    ]
    root = tmp_path / "candidates"
    for arm in module.ARMS:
        for shard_index in range(module.VALIDATION_SHARDS):
            start = shard_index * 64
            end = min(module.VALIDATION_ROWS, start + 64)
            shard = root / arm / f"shard_{shard_index:02d}"
            shard.mkdir(parents=True)
            candidate_path = shard / "candidates.jsonl"
            candidate_path.write_text(
                "".join(
                    json.dumps(_candidate(f"{index:064x}", arm), sort_keys=True) + "\n"
                    for index in range(start, end)
                ),
                encoding="utf-8",
            )
            report = {
                "schema": module.EVALUATION_SCHEMA,
                "status": "complete",
                "arm": arm,
                "split": "external_validation_confirmation",
                "source_sha256": module.VALIDATION_SOURCE_SHA256,
                "shard_index": shard_index,
                "shard_count": module.VALIDATION_SHARDS,
                "row_start": start,
                "row_end": end,
                "full_row_count": module.VALIDATION_ROWS,
                "candidates_sha256": hashlib.sha256(
                    candidate_path.read_bytes()
                ).hexdigest(),
                "assessor_access_count": 0,
                "development_labels_read": 0,
                "sealed_access": {"holdout": 0, "product": 0, "public": 0},
            }
            (shard / "report.json").write_text(json.dumps(report), encoding="utf-8")
    candidates, receipts = module.load_candidates(
        root,
        sources,
        shards=module.VALIDATION_SHARDS,
        expected_source_sha256=module.VALIDATION_SOURCE_SHA256,
        expected_split="external_validation_confirmation",
    )
    assert all(len(candidates[arm]) == module.VALIDATION_ROWS for arm in module.ARMS)
    assert all(len(receipts[arm]) == module.VALIDATION_SHARDS for arm in module.ARMS)


def test_model_loader_rejects_task_feature_claim(tmp_path: Path) -> None:
    model = {
        "schema": module.MODEL_SCHEMA,
        "status": "complete",
        "feature_contract": "task_label_free_hashed_source_and_complete_trajectory_v1",
        "feature_dimension": module.FEATURE_DIMENSION,
        "learning_rate": module.LEARNING_RATE,
        "epochs": module.EPOCHS,
        "commit_margin": module.COMMIT_MARGIN,
        "seed": module.SEED,
        "arms": list(module.ARMS),
        "screen_source_sha256": module.SCREEN_SOURCE_SHA256,
        "screen_score_sha256": module.SCREEN_SCORE_SHA256,
        "screen_rows": module.SCREEN_ROWS,
        "validation_labels_read": 0,
        "task_label_used_as_feature": False,
        "nonzero_weights": [[1, 0.25]],
    }
    path = tmp_path / "model.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    loaded, weights = module._load_model(path)
    assert loaded["task_label_used_as_feature"] is False
    assert weights == {1: 0.25}
    model["task_label_used_as_feature"] = True
    path.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(module.MixtralCommitError, match="model contract"):
        module._load_model(path)


def test_metrics_bind_retention_wins_losses_and_domains() -> None:
    rows = [
        {"correct": [True, False, False], "task": "math500"},
        {"correct": [False, True, False], "task": "mbpp"},
        {"correct": [True, True, True], "task": "bbh_logic"},
    ]
    metrics = module._metrics(rows, [0, 1, 2])
    assert metrics["correct"] == 3
    assert metrics["unchanged_correct_retention"] == 1.0
    assert metrics["wins_over_unchanged"] == 1
    assert metrics["losses_from_unchanged"] == 0
    assert metrics["domains"]["mbpp"] == {"correct": 1, "total": 1}
