from pathlib import Path


SCRIPT = (
    Path(__file__).parent
    / "jobs"
    / "run_federated_ettr_v3_pilot.sh"
).read_text()


def test_federated_pilot_joins_one_gpu_jobs_into_one_world() -> None:
    assert 'IFS=\',\' read -r -a job_ids <<< "$ALLOCATION_JOB_IDS"' in SCRIPT
    assert '--jobid="$job"' in SCRIPT
    assert "--gpus-per-node=1" in SCRIPT
    assert '--nnodes="$world_size"' in SCRIPT
    assert "--nproc_per_node=1" in SCRIPT
    assert '--node_rank="$FEDERATED_RANK"' in SCRIPT
    assert '"$CODE_ROOT/train/train_ettr_v3.py"' in SCRIPT


def test_federated_pilot_preserves_exact_release_and_resume_contracts() -> None:
    assert "RELEASE_SHA256" in SCRIPT
    assert "SOURCE_COMMIT" in SCRIPT
    assert "RESUME_CHECKPOINT" in SCRIPT
    assert "RESUME_SHA256" in SCRIPT
    assert "--run-contract-sha256" in SCRIPT
    assert '"$CODE_ROOT/train/eval_ettr_v3.py"' in SCRIPT


def test_federated_pilot_failure_keeps_reservations_alive() -> None:
    assert "wait -n" in SCRIPT
    assert "kill \"$pid\"" in SCRIPT
    assert "reservations remain alive" in SCRIPT
    assert "scancel" not in SCRIPT


def test_federated_pilot_allows_two_jobs_on_one_dual_h100_host() -> None:
    assert "seen_jobs" in SCRIPT
    assert "seen_nodes" not in SCRIPT


def test_federated_pilot_staggers_large_world_launches() -> None:
    assert "LAUNCH_STAGGER_SECONDS=${LAUNCH_STAGGER_SECONDS:-1}" in SCRIPT
    assert "LAUNCH_STAGGER_SECONDS > 10" in SCRIPT
    assert 'sleep "$LAUNCH_STAGGER_SECONDS"' in SCRIPT
