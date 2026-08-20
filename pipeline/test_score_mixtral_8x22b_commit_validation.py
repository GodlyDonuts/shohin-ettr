import hashlib
import inspect
import json
from pathlib import Path

import pytest

import score_mixtral_8x22b_commit_validation as score
import train_apply_mixtral_8x22b_commit as commit


def _write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def _candidate(identity: str, arm: str, completion: str) -> dict:
    return {
        "schema": commit.CANDIDATE_SCHEMA,
        "arm": arm,
        "identity_sha256": identity,
        "task": "math500",
        "completion": completion,
        "generated_tokens": 4,
        "max_token_exhausted": False,
    }


def test_paired_report_preserves_direction_and_exact_test() -> None:
    left = {"a": True, "b": True, "c": False, "d": True}
    right = {"a": False, "b": True, "c": True, "d": False}
    assert score._paired(left, right) == {
        "left_only_correct": 2,
        "right_only_correct": 1,
        "net_correct": 1,
        "mcnemar_exact_two_sided_p": 1.0,
    }


def test_application_replays_selector_and_rejects_completion_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(score, "ROWS", 2)
    identities = ("1" * 64, "2" * 64)
    sources = [
        {
            "identity_sha256": identity,
            "task": "math500",
            "source_prompt": f"Question {index}",
        }
        for index, identity in enumerate(identities)
    ]
    candidates = {
        arm: [
            _candidate(identity, arm, f"{arm} answer {index}")
            for index, identity in enumerate(identities)
        ]
        for arm in commit.ARMS
    }
    model = {
        "schema": commit.MODEL_SCHEMA,
        "status": "complete",
        "feature_contract": "task_label_free_hashed_source_and_complete_trajectory_v1",
        "feature_dimension": commit.FEATURE_DIMENSION,
        "learning_rate": commit.LEARNING_RATE,
        "epochs": commit.EPOCHS,
        "commit_margin": commit.COMMIT_MARGIN,
        "seed": commit.SEED,
        "arms": list(commit.ARMS),
        "screen_source_sha256": commit.SCREEN_SOURCE_SHA256,
        "screen_score_sha256": commit.SCREEN_SCORE_SHA256,
        "screen_rows": commit.SCREEN_ROWS,
        "validation_labels_read": 0,
        "task_label_used_as_feature": False,
        "nonzero_weights": [[1, 0.25]],
    }
    model_path = tmp_path / "model.json"
    model_path.write_text(json.dumps(model), encoding="utf-8")
    model_sha = commit.sha256_file(model_path)
    monkeypatch.setattr(score, "COMMIT_MODEL_SHA256", model_sha)
    _, weights = commit._load_model(model_path)
    selected_rows = []
    selections = []
    counts: dict[str, int] = {}
    for index, source in enumerate(sources):
        features = [
            commit.candidate_features(
                source["source_prompt"], arm, candidates[arm][index]
            )
            for arm in commit.ARMS
        ]
        selected_index, probabilities = commit.select_arm(weights, features)
        arm = commit.ARMS[selected_index]
        candidate = candidates[arm][index]
        counts[arm] = counts.get(arm, 0) + 1
        selected_rows.append(
            {
                "schema": commit.CANDIDATE_OUTPUT_SCHEMA,
                "arm": "selective_commit",
                "selected_arm": arm,
                "identity_sha256": source["identity_sha256"],
                "task": source["task"],
                "completion": candidate["completion"],
                "generated_tokens": candidate["generated_tokens"],
                "max_token_exhausted": candidate["max_token_exhausted"],
                "model_sha256": model_sha,
            }
        )
        selections.append(
            {
                "schema": commit.SELECTION_SCHEMA,
                "split": "external_validation_confirmation",
                "identity_sha256": source["identity_sha256"],
                "selected_arm": arm,
                "probabilities": probabilities,
                "model_sha256": model_sha,
            }
        )
    output = tmp_path / "candidates.jsonl"
    selection_path = tmp_path / "selections.jsonl"
    output_sha = _write_jsonl(output, selected_rows)
    selections_sha = _write_jsonl(selection_path, selections)
    candidate_receipts = {arm: [arm] for arm in commit.ARMS}
    report_path = tmp_path / "report.json"
    report = {
        "schema": commit.APPLICATION_REPORT_SCHEMA,
        "status": "complete",
        "rows": 2,
        "source_sha256": commit.VALIDATION_SOURCE_SHA256,
        "candidate_report_sha256s": candidate_receipts,
        "model_sha256": model_sha,
        "output": str(output.resolve()),
        "output_sha256": output_sha,
        "selections": str(selection_path.resolve()),
        "selections_sha256": selections_sha,
        "selection_counts": dict(sorted(counts.items())),
        "assessor_access_count": 0,
        "validation_labels_read": 0,
        "task_label_used_as_feature": False,
        "sealed_access": {"holdout": 0, "product": 0, "public": 0},
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "model": model_path,
            "application_report": report_path,
            "commit_candidates": output,
            "selections": selection_path,
        },
    )()
    observed, _ = score.validate_application(
        args, sources, candidates, candidate_receipts
    )
    assert observed == {
        source["identity_sha256"]: row["selected_arm"]
        for source, row in zip(sources, selections, strict=True)
    }

    selected_rows[0]["completion"] = "tampered"
    report["output_sha256"] = _write_jsonl(output, selected_rows)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(score.MixtralCommitScoreError, match="commit candidate"):
        score.validate_application(args, sources, candidates, candidate_receipts)


def test_scorer_opens_assessor_only_after_four_arm_validation() -> None:
    source = inspect.getsource(score.run)
    assert source.index("validate_application(") < source.index("load_assessors(")
    assert 'outcomes["selective_commit"][identity] = outcomes[' in source


def test_apply_and_score_jobs_are_cpu_only_and_nonrequeueing() -> None:
    root = Path(__file__).with_name("jobs")
    apply_job = (root / "mixtral_8x22b_commit_apply.sbatch").read_text()
    score_job = (root / "mixtral_8x22b_commit_validation_score.sbatch").read_text()
    for source in (apply_job, score_job):
        assert "#SBATCH --no-requeue" in source
        assert "--gres" not in source
        assert "q36_verify_runtime" in source
    assert "train_apply_mixtral_8x22b_commit.py" in apply_job
    assert "score_mixtral_8x22b_commit_validation.py" in score_job
