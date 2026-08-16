from pathlib import Path

JOB = Path(__file__).with_name("jobs") / "score_upward_moe_temporal_gate.sbatch"


def test_upward_score_collects_five_exact_sixteen_shard_arms() -> None:
    text = JOB.read_text(encoding="utf-8")
    assert "#SBATCH --no-requeue" in text
    assert "--gres" not in text
    assert (
        "for arm in unchanged self_refinement owner aligned_revision temporal_gate"
        in text
    )
    assert "for index in $(seq 0 15)" in text
    assert "--candidate" in text and "--evaluation-report" in text
    assert "score_upward_moe_temporal_gate.py" in text
