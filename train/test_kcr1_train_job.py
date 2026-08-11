"""Static custody tests for the KCR1 one-owner training wrapper."""

from pathlib import Path


def test_kcr1_training_job_is_hash_bound_and_uses_weighted_loss() -> None:
    script = Path("train/jobs/hf_kcr1_train_reviser.sbatch").read_text()
    assert "gpu:nvidia_h100_pcie:1" in script
    assert "RUNTIME_MANIFEST_SHA256" in script
    assert "MODEL_CONFIG_SHA256" in script
    assert "WARM_START_SHA256" in script
    assert "DATA_REPORT_SHA256" in script
    assert "shohin-kcr1-branch-data-report-v1" in script
    assert "transaction_roundtrip_rows" in script
    assert "--loss-mode kcr1_action_payload" in script
    assert "--max-sequence-length 4096" in script
    assert "--lora-layers 4 --lora-rank 8" in script
    assert "test ! -e \"$OUTPUT\"" in script
    assert "chmod -R a-w \"$OUTPUT\"" in script
