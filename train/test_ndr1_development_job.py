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


def test_ndr1_development_armer_binds_fit_jobs_and_keeps_holdout_closed() -> None:
    script = Path("train/jobs/arm_ndr1_development.sbatch").read_text()
    assert 'value.get("status") != "submitted"' in script
    assert '("aligned_job", "shuffled_job")' in script
    assert 'ndr_data_report_sha256=$(sha256sum "$NDR_DATA_REPORT"' in script
    assert '--dependency=afterok:"$aligned_job":"$shuffled_job"' in script
    assert 'dispatch_ndr1_development.sbatch' in script
    assert '"holdout_opened": False' in script
    assert "DEVELOPMENT_REPORT_SHA256" in script
    assert "HOLDOUT" not in script
