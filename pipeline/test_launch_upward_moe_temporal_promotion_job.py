"""Static contract tests for the automatic larger-MoE temporal launch job."""

from pathlib import Path

JOB = Path(__file__).with_name("jobs") / "launch_upward_moe_temporal_promotion.sbatch"


def test_job_is_cpu_only_nonrequeue_and_receipt_bound() -> None:
    source = JOB.read_text(encoding="utf-8")
    assert "#SBATCH --partition=normal" in source
    assert "#SBATCH --no-requeue" in source
    assert "#SBATCH --gres" not in source
    assert '[[ -f "$PROMOTION" && ! -L "$PROMOTION" ]]' in source
    assert '[[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]]' in source
    assert '[[ ! -e "$AUTOMATION_ROOT" && ! -L "$AUTOMATION_ROOT" ]]' in source


def test_job_invokes_one_submit_and_freezes_automation_receipts() -> None:
    source = JOB.read_text(encoding="utf-8")
    assert "launch_upward_moe_temporal_promotion.py" in source
    assert source.count("--submit") == 1
    assert "trap finalize EXIT" in source
    assert 'chmod -R a-w "$AUTOMATION_ROOT"' in source
    assert "RUNTIME_MANIFEST_SHA256" in source
