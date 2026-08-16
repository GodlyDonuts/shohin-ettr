from pathlib import Path

JOB = Path(__file__).with_name("upward_moe_train_temporal_gate.sbatch")


def test_upward_temporal_job_is_exact_two_h100_causal_only() -> None:
    text = JOB.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:2" in text
    assert "#SBATCH --no-requeue" in text
    assert "hf_upward_moe_train_temporal_gate.py" in text
    assert "--causal-loss-weight 1.0" in text
    assert "--routing-supervision-weight 0.0" in text
    assert "OWNER_CHECKPOINT" in text and "REVISION_CHECKPOINT" in text
    assert "DRAFT_HIDDEN_CHECKPOINT" not in text
    assert "olmoe" not in text.casefold()
