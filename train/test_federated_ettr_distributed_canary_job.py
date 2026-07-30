from pathlib import Path


SCRIPT = (
    Path(__file__).parent
    / "jobs"
    / "run_federated_ettr_distributed_canary.sh"
).read_text()


def test_federated_canary_joins_independent_single_h100_jobs() -> None:
    assert 'IFS=\',\' read -r -a job_ids <<< "$ALLOCATION_JOB_IDS"' in SCRIPT
    assert '--jobid="$job"' in SCRIPT
    assert "--gpus-per-node=1" in SCRIPT
    assert '--nnodes="$world_size"' in SCRIPT
    assert "--nproc_per_node=1" in SCRIPT
    assert '--node_rank="$FEDERATED_RANK"' in SCRIPT
    assert '--expected-world-size "$world_size"' in SCRIPT


def test_federated_canary_fails_one_step_without_canceling_reservations() -> None:
    assert "wait -n" in SCRIPT
    assert "kill \"$pid\"" in SCRIPT
    assert "reservations remain alive" in SCRIPT
    assert "scancel" not in SCRIPT


def test_federated_canary_is_release_blind_and_hash_bound() -> None:
    assert "SOURCE_COMMIT" in SCRIPT
    assert "SHA256SUMS" in SCRIPT
    assert "PROTECTED_CHECKPOINT_SHA256" in SCRIPT
    assert "RELEASE_ROOT" not in SCRIPT
    assert "DATA_ROOT" not in SCRIPT
    assert "SHARD_ROOT" not in SCRIPT
