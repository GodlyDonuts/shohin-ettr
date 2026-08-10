from pathlib import Path


def test_ndr1_development_dispatch_is_sharded_and_holdout_closed() -> None:
    script = Path("train/jobs/dispatch_ndr1_development.sbatch").read_text()
    assert "SHARD_COUNT=${SHARD_COUNT:-4}" in script
    assert "for arm in aligned shuffled" in script
    assert "hf_idr1_evaluate_reviser.sbatch" in script
    assert "merge_idr1_evaluation_shards.py" in script
    assert "compare_ndr1_development.py" in script
    assert 'SPLIT=development' in script
    assert '"holdout_opened": False' in script
    assert "checkpoint_0000512.pt" in script
    assert "--dependency=afterok:" in script
    assert "SPLIT=holdout" not in script
