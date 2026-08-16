"""Static job-contract tests for larger-MoE temporal promotion."""

from pathlib import Path

JOB = Path(__file__).with_name("jobs") / "select_upward_moe_temporal_promotion.sbatch"


def test_job_is_cpu_only_dependency_safe_and_write_once() -> None:
    source = JOB.read_text(encoding="utf-8")
    assert "#SBATCH --partition=normal" in source
    assert "#SBATCH --no-requeue" in source
    assert "#SBATCH --gres" not in source
    assert '[[ -f "$score" && ! -L "$score" ]]' in source
    assert '[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]' in source
    assert 'chmod a-w "$OUTPUT"' in source


def test_job_binds_both_larger_hosts_and_immutable_runtime() -> None:
    source = JOB.read_text(encoding="utf-8")
    assert "RUNTIME_MANIFEST_SHA256" in source
    assert "q36_verify_runtime" in source
    assert '--score "$SUPER_SCORE"' in source
    assert '--score "$MIXTRAL_SCORE"' in source
    assert "select_upward_moe_temporal_promotion.py" in source
