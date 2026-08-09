from pathlib import Path


SOURCE = Path(__file__).with_name("jobs").joinpath("dispatch_kr2_campaign.sbatch").read_text()


def test_kr2_campaign_is_two_arm_and_dependency_gated() -> None:
    assert "for arm in keep_or_repair direct_rewrite" in SOURCE
    assert "--dependency=afterok:${train_jobs[$arm]}" in SOURCE
    assert "--dependency=afterok:$dependency" in SOURCE
    assert "compare_dependency=afterok:" in SOURCE


def test_kr2_campaign_preserves_frozen_geometry() -> None:
    assert "SHARD_COUNT=8" in SOURCE
    assert "MODEL_LOADER=multimodal" in SOURCE
    assert "hf_idr1_train_reviser.sbatch" in SOURCE
    assert "hf_kr2_evaluate.sbatch" in SOURCE
    assert "compare_kr2_stage_owner.py" in SOURCE
    assert "EXCLUDE_NODES=${EXCLUDE_NODES:-evc33,evc38}" in SOURCE
    assert 'EXCLUDE_NODES=${EXCLUDE_NODES//:/,}' in SOURCE
    assert SOURCE.count('--exclude="$EXCLUDE_NODES"') == 2
