from pathlib import Path


def test_kcr1_evaluator_is_hash_bound() -> None:
    script = Path("train/jobs/hf_kcr1_evaluate.sbatch").read_text()
    assert "gpu:nvidia_h100_pcie:1" in script
    assert "RUNTIME_MANIFEST_SHA256" in script
    assert "ADAPTER_CHECKPOINT_SHA256" in script
    assert "DATA_REPORT_SHA256" in script
    assert "--max-new-tokens 768" in script
    assert "test ! -e \"$REPORT\"" in script


def test_kcr1_dispatcher_waits_for_fit_and_never_opens_holdout() -> None:
    script = Path("train/jobs/dispatch_kcr1_canary.sbatch").read_text()
    assert "checkpoint_0000512.pt" in script
    assert 'r.get("loss_mode")!="kcr1_action_payload"' in script
    assert "hf_kcr1_evaluate.sbatch" in script
    assert "merge_kcr1_evaluation_shards.py" in script
    assert "DATA_REPORT_SHA256" in script
    assert '"holdout_opened":False' in script
    assert "HOLDOUT" not in script
