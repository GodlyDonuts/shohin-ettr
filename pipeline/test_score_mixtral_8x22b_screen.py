"""Tests for the cross-family 141B-A39B score boundary."""

from pathlib import Path

import score_mixtral_8x22b_screen as score
import score_mixtral_8x22b_validation as validation


def test_mixtral_score_wrapper_binds_host_and_candidate_schema() -> None:
    assert score.CANDIDATE_SCHEMA == "shohin-mixtral-8x22b-fixed-draft-candidate-v1"
    assert score.REPORT_SCHEMA == "shohin-mixtral-8x22b-fixed-draft-screen-score-v1"
    assert score.TOTAL_PARAMETERS == 141_000_000_000
    assert score.ACTIVE_PARAMETERS == 39_000_000_000


def test_mixtral_score_job_is_one_shot_and_initializes_local_tmp() -> None:
    source = (
        Path(__file__)
        .with_name("jobs")
        .joinpath("mixtral_8x22b_score.sbatch")
        .read_text()
    )
    assert "#SBATCH --no-requeue" in source
    assert "score_mixtral_8x22b_screen.py" in source
    assert "q36_init_local_tmp" in source
    assert "trap q36_cleanup_local_tmp EXIT" in source
    assert 'chmod a-w "$OUTPUT" "$SANDBOX_RECEIPT"' in source


def test_validation_score_wrapper_binds_promoted_geometry() -> None:
    assert validation.CANDIDATE_SCHEMA == score.CANDIDATE_SCHEMA
    assert (
        validation.REPORT_SCHEMA
        == "shohin-mixtral-8x22b-fixed-draft-validation-score-v1"
    )
    assert validation.ROWS == 1023
    assert validation.SHARDS == 16
    source = (
        Path(__file__)
        .with_name("jobs")
        .joinpath("mixtral_8x22b_validation_score.sbatch")
        .read_text()
    )
    assert "for shard in $(seq -w 0 15)" in source
    assert "score_mixtral_8x22b_validation.py" in source
    assert "#SBATCH --no-requeue" in source
