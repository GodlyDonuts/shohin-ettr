"""Static custody checks for KCR1 control-canary dispatch."""

from pathlib import Path


SCRIPT = Path("train/jobs/dispatch_kcr1_control_canary.sbatch").read_text()


def test_dispatch_accepts_only_frozen_controls() -> None:
    assert "action_permuted)" in SCRIPT
    assert "draft_hidden)" in SCRIPT
    assert "unsupported KCR1 canary control" in SCRIPT
    assert "constant_restart)" not in SCRIPT


def test_dispatch_binds_control_interventions_and_sealed_canary() -> None:
    assert "EXPECTED_TRAIN_SHA" in SCRIPT
    assert 'r.get("mask_internal_draft")' in SCRIPT
    assert 'd.get("holdout_used") is not False' in SCRIPT
    assert "SHARD_COUNT=${SHARD_COUNT:-4}" in SCRIPT
    assert "--shard-index" not in SCRIPT
    assert "hf_kcr1_evaluate.sbatch" in SCRIPT
    assert "merge_kcr1_evaluation_shards.py" in SCRIPT
