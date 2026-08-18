from pathlib import Path

import score_gpt_oss_120b_screen as score


def test_score_wrapper_binds_the_openai_moe_host() -> None:
    assert score.CANDIDATE_SCHEMA == "shohin-gpt-oss-120b-fixed-draft-candidate-v1"
    assert score.REPORT_SCHEMA == "shohin-gpt-oss-120b-fixed-draft-screen-score-v1"
    assert score.HOST == "openai/gpt-oss-120b"
    assert score.TOTAL_PARAMETERS == 117_000_000_000
    assert score.ACTIVE_PARAMETERS == 5_100_000_000


def test_score_job_uses_the_existing_qualified_cpu_sandbox() -> None:
    source = (
        Path(__file__).parents[0].joinpath("jobs/gpt_oss_120b_score.sbatch").read_text()
    )
    assert "score_gpt_oss_120b_screen.py" in source
    assert "--no-requeue" in source
    assert "pcf1_code_sandbox" not in source
    assert "--sandbox-receipt" in source
