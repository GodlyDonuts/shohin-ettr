from pathlib import Path

SCRIPT = Path(__file__).with_name("submit_gpt_oss_120b_screen.sh")


def test_dispatch_requires_a_passed_score_free_mechanics_receipt() -> None:
    source = SCRIPT.read_text()
    assert "shohin-gpt-oss-120b-one-h100-mechanics-v1" in source
    assert 'payload.get("status") != "pass"' in source
    assert 'payload.get("scientific_result") is not False' in source
    assert 'payload.get("score_or_assessor_data_accessed") is not False' in source


def test_dispatch_submits_twelve_independent_single_gpu_evaluations() -> None:
    source = SCRIPT.read_text()
    assert "for arm in unchanged self_refinement revision" in source
    assert "for shard in 0 1 2 3" in source
    assert "evaluation_job_count" in source
    assert '"array_jobs": 0' in source
    assert "--array" not in source


def test_revision_waits_for_fit_but_controls_can_backfill_immediately() -> None:
    source = SCRIPT.read_text()
    assert 'if [[ "$arm" == unchanged ]]' in source
    assert 'elif [[ "$arm" == self_refinement ]]' in source
    assert 'dependency=(--dependency="afterok:$fit_job")' in source
    assert "REVISION_CHECKPOINT=$fit_output/checkpoint_0000256.pt" in source


def test_score_waits_for_every_exact_evaluation_job() -> None:
    source = SCRIPT.read_text()
    assert "dependency=$(IFS=:; printf '%s' \"${eval_jobs[*]}\")" in source
    assert '--dependency="afterok:$dependency"' in source
    assert "env -u SLURM_OVERLAP -u SLURM_WHOLE sbatch" in source
