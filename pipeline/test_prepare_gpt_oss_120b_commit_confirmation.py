from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

import prepare_gpt_oss_120b_commit_confirmation as module


def _identity(prefix: str, index: int) -> str:
    return hashlib.sha256(f"{prefix}-{index}".encode()).hexdigest()


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _question(identity: str) -> dict:
    return {
        "schema": module.QUESTION_SCHEMA,
        "id": identity,
        "benchmark": module.BENCHMARK,
        "upstream_id": identity[:12],
        "question": f"question {identity}",
        "response_mode": "general",
    }


def _prior(identity: str) -> dict:
    return {
        "schema": module.SOURCE_SCHEMA,
        "split": "external_validation",
        "identity_sha256": identity,
        "task": "math500",
        "source_prompt": "old question",
    }


def _fixture(tmp_path: Path) -> argparse.Namespace:
    questions = [_question(_identity("mmlu", index)) for index in range(12_032)]
    excluded = questions[:256]
    prior = [_prior(_identity("prior", index)) for index in range(1_279)]
    question_path = tmp_path / "questions.jsonl"
    excluded_path = tmp_path / "excluded.jsonl"
    prior_path = tmp_path / "prior.jsonl"
    prior_confirmation_path = tmp_path / "prior-confirmation.jsonl"
    _write(question_path, questions)
    _write(excluded_path, excluded)
    _write(prior_path, prior)
    _write(
        prior_confirmation_path,
        [
            {
                **_prior(row["id"]),
                "task": module.BENCHMARK,
            }
            for row in questions[256:512]
        ],
    )
    output = tmp_path / "output"
    return argparse.Namespace(
        questions=question_path,
        excluded_questions=excluded_path,
        prior_q36_source=prior_path,
        prior_confirmation_source=prior_confirmation_path,
        selection_seed=2026082002,
        source_output=output / "source.jsonl",
        receipt=output / "receipt.json",
    )


def test_preparation_is_deterministic_disjoint_and_label_free(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    receipt = module.prepare(args)
    rows = [json.loads(line) for line in args.source_output.read_text().splitlines()]
    excluded = {
        row["id"]
        for row in map(json.loads, args.excluded_questions.read_text().splitlines())
    }
    prior = {
        row["identity_sha256"]
        for row in map(json.loads, args.prior_q36_source.read_text().splitlines())
    }
    prior_confirmation = {
        row["identity_sha256"]
        for row in map(
            json.loads, args.prior_confirmation_source.read_text().splitlines()
        )
    }
    identities = {row["identity_sha256"] for row in rows}
    assert len(rows) == 256
    assert not identities & excluded
    assert not identities & prior
    assert not identities & prior_confirmation
    assert {row["task"] for row in rows} == {"mmlu_pro"}
    assert receipt["assessor_access_count"] == 0
    assert receipt["selection_seed"] == 2026082002
    assert receipt["prior_confirmation_identity_overlap"] == 0


def test_preparation_rejects_label_bearing_question(tmp_path: Path) -> None:
    args = _fixture(tmp_path)
    rows = [json.loads(line) for line in args.questions.read_text().splitlines()]
    rows[0]["correct"] = True
    _write(args.questions, rows)
    with pytest.raises(module.ConfirmationPreparationError, match="question differs"):
        module.prepare(args)


def test_question_reader_preserves_unicode_line_separator_inside_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "questions.jsonl"
    row = _question(_identity("unicode-line-separator", 0))
    row["question"] = "first paragraph\u2028second paragraph"
    path.write_text(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    assert module._questions(path) == [row]


def test_confirmation_launcher_uses_thirteen_independent_single_h100_jobs() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (
        root / "train/jobs/submit_gpt_oss_120b_commit_confirmation.sh"
    ).read_text()
    evaluator = (root / "train/jobs/gpt_oss_120b_evaluate.sbatch").read_text()
    selector = (root / "train/jobs/q36_mtr_cross_host_commit.sbatch").read_text()
    assert "for arm in unchanged revision" in launcher
    assert "for shard in 0 1 2 3" in launcher
    assert '--dependency="afterok:$eval_dependency"' in launcher
    assert "independent_single_h100_jobs" in launcher
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in evaluator
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in selector
    assert "--array" not in launcher
    assert "0.703125" in launcher
    assert "REVISION_RELIABILITY_VETO" in launcher


def test_preparation_job_accepts_preexisting_shared_logs_directory() -> None:
    root = Path(__file__).resolve().parents[1]
    wrapper = (
        root / "pipeline/jobs/prepare_gpt_oss_120b_commit_confirmation.sbatch"
    ).read_text()
    assert 'mkdir -m 700 "$OUTPUT_ROOT"\n' in wrapper
    assert "mkdir -p logs\n" in wrapper
    assert 'mkdir -m 700 "$OUTPUT_ROOT" logs' not in wrapper
