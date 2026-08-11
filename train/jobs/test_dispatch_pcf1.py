"""Static and no-submit checks for the terminal PCF1 Slurm state machine."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

STAGES = (
    "prepare_inputs",
    "mechanics",
    "b1_train",
    "draft_generate",
    "draft_merge",
    "materialize",
    "revision_train",
    "calibration_revision_eval",
    "calibration_revision_merge",
    "calibration_unchanged_eval",
    "calibration_unchanged_merge",
    "calibration_pairs",
    "commit_train",
    "confirmation_revision_eval",
    "confirmation_revision_merge",
    "confirmation_unchanged_eval",
    "confirmation_unchanged_merge",
    "confirmation_self_refinement_eval",
    "confirmation_self_refinement_merge",
    "confirmation_pairs",
    "commit_apply",
    "precompute_custody",
    "prescore_accounting",
    "authorize_score",
    "commit_score",
    "normalize",
    "final_accounting",
    "compute_custody",
    "final_compare",
)
GPU_STAGES = {
    "mechanics",
    "b1_train",
    "draft_generate",
    "revision_train",
    "calibration_revision_eval",
    "calibration_unchanged_eval",
    "commit_train",
    "confirmation_revision_eval",
    "confirmation_unchanged_eval",
    "confirmation_self_refinement_eval",
    "commit_apply",
}


def _dispatcher() -> Path:
    return Path(__file__).with_name("dispatch_pcf1.sh")


def test_dry_run_is_complete_acyclic_and_terminal() -> None:
    completed = subprocess.run(
        ["bash", str(_dispatcher()), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "submission=false mutation=false" in completed.stdout
    assert "automatic_retry=false automatic_successor=false" in completed.stdout
    rows = [
        line.split("|")
        for line in completed.stdout.splitlines()
        if re.match(r"^[a-z][a-z0-9_]+\|[01]\|\d+\|", line)
    ]
    assert tuple(row[0] for row in rows) == STAGES
    seen: set[str] = set()
    for stage, gpus, tasks, dependencies in rows:
        assert set(dependencies.split()) <= seen
        assert int(gpus) == int(stage in GPU_STAGES)
        assert int(tasks) == (
            16 if stage == "draft_generate" else 4 if stage.endswith("_eval") else 1
        )
        seen.add(stage)
    assert rows[-1][0] == "final_compare"


def test_submit_uses_allowlisted_exports_and_literal_one_open() -> None:
    source = _dispatcher().read_text(encoding="utf-8")
    assert '--export="$exports"' in source
    assert '--export="NONE,' not in source
    assert "--no-requeue" in source
    assert "--nodes=1" in source
    assert "--ntasks=1" in source
    assert "--gres=gpu:nvidia_h100_pcie:1" in source
    assert "reject_ambient_scheduler_controls" in source
    assert "SBATCH_*|SLURM_*" in source
    assert '--partition="$PARTITION"' in source
    assert '--exclude="$EXCLUDED_NODES"' in source
    calls = {}
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped.startswith("submit_stage "):
            continue
        stage = stripped.split(maxsplit=2)[1]
        calls[stage] = stripped
    assert set(calls) == set(STAGES)
    for stage in GPU_STAGES:
        folded = calls[stage].casefold()
        assert "assessor" not in folded
        assert "holdout" not in folded
        assert "product" not in folded
        assert "public" not in folded
    assert "ASSESSOR_OUTPUT=" in calls["prepare_inputs"]
    assert "CONFIRMATION_ASSESSORS=" in calls["commit_score"]
    for stage in set(STAGES) - {"prepare_inputs", "commit_score"}:
        assert "ASSESSOR_OUTPUT=" not in calls[stage]
        assert "CONFIRMATION_ASSESSORS=" not in calls[stage]
    assert "ASSESSOR_BOARD=" not in source
    for stage in (
        "b1_train",
        "calibration_revision_eval",
        "calibration_unchanged_eval",
        "precompute_custody",
        "authorize_score",
        "commit_score",
        "compute_custody",
    ):
        assert "SANDBOX_RECEIPT=" in calls[stage]
    for stage in (
        "confirmation_revision_eval",
        "confirmation_unchanged_eval",
        "confirmation_self_refinement_eval",
    ):
        assert "SANDBOX_RECEIPT=" not in calls[stage]

    repository = _dispatcher().parents[2]
    evaluator = (repository / "train/jobs/pcf1_evaluate.sbatch").read_text()
    merger = (repository / "pipeline/jobs/pcf1_merge_evaluation.sbatch").read_text()
    precompute = (
        repository / "pipeline/jobs/pcf1_build_precompute_custody.sbatch"
    ).read_text()
    calibration_pairs = (
        repository / "pipeline/jobs/pcf1_build_commit_pairs.sbatch"
    ).read_text()
    confirmation_pairs = (
        repository / "pipeline/jobs/pcf1_build_confirmation_pairs.sbatch"
    ).read_text()
    authorizer = (repository / "pipeline/jobs/pcf1_authorize_score.sbatch").read_text()
    scorer = (repository / "pipeline/jobs/pcf1_score_commit.sbatch").read_text()
    assert "--sandbox-probe-output" in evaluator
    assert '--shard-sandbox-probe "$shard_sandbox_probe"' in merger
    assert '--shard-root "$SHARD_ROOT"' in merger
    assert "confirmation shard unexpectedly contains a sandbox probe" in merger
    for wrapper in (calibration_pairs, confirmation_pairs, authorizer, scorer):
        assert '--candidates-root "$CANDIDATES_ROOT"' in wrapper
    for stage in (
        "calibration_pairs",
        "confirmation_pairs",
        "authorize_score",
        "commit_score",
    ):
        assert "CANDIDATES_ROOT=$merged" in calls[stage]
    for stage in (
        "calibration_revision_merge",
        "calibration_unchanged_merge",
        "confirmation_revision_merge",
        "confirmation_unchanged_merge",
        "confirmation_self_refinement_merge",
    ):
        assert "SHARD_ROOT=$evaluations/" in calls[stage]
    assert "--calibration-revision-sandbox-probe" in precompute
    assert "--calibration-unchanged-sandbox-probe" in precompute
    assert "--compute-host-receipt" in precompute
    assert "--reference-sandbox-receipt" in precompute
    assert "--reference-preflight-rows" in precompute
    assert "CALIBRATION_REVISION_SHARD_ROOT=" in calls["precompute_custody"]
    assert "CALIBRATION_UNCHANGED_SHARD_ROOT=" in calls["precompute_custody"]
    assert "COMPUTE_HOST_RECEIPT=" in calls["precompute_custody"]
    assert "REFERENCE_SANDBOX_RECEIPT=$prepared/sources/" in calls["precompute_custody"]
    assert "REFERENCE_PREFLIGHT_ROWS=$prepared/sources/" in calls["precompute_custody"]
    assert "--sandbox-probe-output" in scorer
    assert "score-authorization-consumed.json" in scorer
    assert "pcf1_validate_sandbox_receipt" in scorer
    assert "terminal-failure.json" in scorer
    assert 'chmod a-w "$terminal_failure"' in scorer
    assert 'pcf1_freeze_tree "$OUTPUT_ROOT"' in scorer
    assert (
        "PYTHON_ENTRYPOINT_PIN=/lustre/fs1/home/sa305415/shohin/envs/"
        "product-reasoning-b3a3603-r2/bin/python"
    ) in source
    assert '[[ "$PYTHON" == "$PYTHON_ENTRYPOINT_PIN" ]]' in source
    assert "environment_payload" in source

    for job_name in {
        "pcf1_mechanics.sbatch",
        "pcf1_train_b1.sbatch",
        "pcf1_generate_drafts.sbatch",
        "pcf1_train_revision.sbatch",
        "pcf1_evaluate.sbatch",
        "pcf1_train_commit.sbatch",
        "pcf1_apply_commit.sbatch",
    }:
        job = repository / "train/jobs" / job_name
        assert "pcf1_assert_gpu_environment" in job.read_text(encoding="utf-8")


def test_all_jobs_pin_partition_and_exclusions() -> None:
    repository = _dispatcher().parents[2]
    jobs = sorted((repository / "train/jobs").glob("pcf1_*.sbatch"))
    jobs += sorted((repository / "pipeline/jobs").glob("pcf1_*.sbatch"))
    assert jobs
    for job in jobs:
        source = job.read_text(encoding="utf-8")
        assert "#SBATCH --partition=normal" in source, job
        assert "#SBATCH --exclude=evc26,evc29,evc31,evc32,evc38,evc46" in source, job


def test_runtime_allowlist_contains_the_complete_graph_and_security_modules() -> None:
    repository = _dispatcher().parents[2]
    allowlist_path = repository / "pipeline/pcf1_runtime_allowlist.txt"
    entries = {
        line.strip()
        for line in allowlist_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    jobs = sorted((repository / "train/jobs").glob("pcf1_*.sbatch"))
    jobs += sorted((repository / "pipeline/jobs").glob("pcf1_*.sbatch"))
    expected_jobs = {path.relative_to(repository).as_posix() for path in jobs}
    assert expected_jobs <= entries
    assert {
        "pipeline/capture_pcf1_environment.py",
        "pipeline/capture_pcf1_slurm_accounting.py",
        "pipeline/package_pcf1_runtime.py",
        "train/jobs/dispatch_pcf1.sh",
        "train/jobs/pcf1_common.sh",
        "train/pcf1_code_sandbox.py",
        "train/pcf1_environment.py",
    } <= entries
