from pathlib import Path


def test_kcr1_control_fit_is_exact_and_explicit() -> None:
    script = Path("train/jobs/hf_kcr1_control_train.sbatch").read_text()
    assert "gpu:nvidia_h100_pcie:1" in script
    assert "action_permuted|constant_restart" in script
    assert "draft_hidden) MASK_ARGS=(--mask-internal-draft)" in script
    assert "RUNTIME_MANIFEST_SHA256" in script
    assert "WARM_START_SHA256" in script
    assert "DATA_REPORT_SHA256" in script
    assert "shohin-kcr1-control-data-report-v1" in script
    assert "--loss-mode kcr1_action_payload" in script
    assert "--max-sequence-length 4096" in script
    assert "test ! -e \"$OUTPUT\"" in script
