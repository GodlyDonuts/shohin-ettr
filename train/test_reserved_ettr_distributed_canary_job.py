from pathlib import Path


SCRIPT = (
    Path(__file__).parent
    / "jobs"
    / "run_reserved_ettr_distributed_canary.sh"
).read_text()


def test_distributed_canary_uses_every_selected_h100() -> None:
    assert '--jobid="$ALLOCATION_JOB_ID"' in SCRIPT
    assert '--nnodes="$NODES"' in SCRIPT
    assert '--nproc_per_node="$GPUS_PER_NODE"' in SCRIPT
    assert '--node_rank="$SLURM_NODEID"' in SCRIPT
    assert '--expected-world-size "$world_size"' in SCRIPT
    assert "NCCL_IB_DISABLE=0" in SCRIPT


def test_distributed_canary_is_source_and_checkpoint_bound() -> None:
    assert "SOURCE_COMMIT" in SCRIPT
    assert "SHA256SUMS" in SCRIPT
    assert "--checkpoint-sha256" in SCRIPT
    assert "--expected-step 300000" in SCRIPT
    assert "canary_ettr_distributed_h100.py" in SCRIPT


def test_distributed_canary_cannot_read_release_or_training_shards() -> None:
    assert "RELEASE_ROOT" not in SCRIPT
    assert "DATA_ROOT" not in SCRIPT
    assert "SHARD_ROOT" not in SCRIPT
