#!/usr/bin/env python3
"""Static scheduler-contract checks for the automatic IDR scale decision."""

from pathlib import Path


ROOT = Path(__file__).parent / "jobs"


def test_compare_job_runs_frozen_scorer_on_all_four_reports():
    text = (ROOT / "compare_idr_scale.sbatch").read_text()
    assert "pipeline/compare_idr_scale.py" in text
    assert text.count("/merged/report.json") == 4
    assert "test ! -e \"$COMPARISON_OUTPUT\"" in text


def test_compare_dispatch_waits_for_all_four_merge_jobs():
    text = (ROOT / "dispatch_idr_scale_compare.sbatch").read_text()
    assert 'len(set(jobs)) != 4' in text
    assert '--dependency=afterok:"$dependency"' in text
    assert "compare_idr_scale.sbatch" in text


def test_training_dispatch_conditionally_releases_comparison():
    text = (ROOT / "dispatch_idr_scale_train.sbatch").read_text()
    assert 'if [[ -n "${COMPARISON_OUTPUT:-}" ]]' in text
    assert "dispatch_idr_scale_compare.sbatch" in text
    assert '"compare_dispatch_job": compare or None' in text
