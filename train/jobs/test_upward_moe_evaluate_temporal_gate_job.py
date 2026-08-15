from pathlib import Path

JOB = Path(__file__).with_name("upward_moe_evaluate_temporal_gate.sbatch")


def test_upward_temporal_evaluation_is_five_matched_two_h100_arrays() -> None:
    text = JOB.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:2" in text
    assert "#SBATCH --no-requeue" in text
    for arm in (
        "unchanged",
        "self_refinement",
        "owner",
        "aligned_revision",
        "temporal_gate",
    ):
        assert arm in text
    assert "--expected-rows 1289" in text
    assert "--shard-count 16" in text
    assert "hf_upward_moe_evaluate_temporal_gate.py" in text
    assert "edit_selector" not in text
