from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_upward_draft_array_is_exact_two_h100_no_requeue() -> None:
    text = (ROOT / "train/jobs/upward_moe_generate_drafts.sbatch").read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:2" in text
    assert "#SBATCH --no-requeue" in text
    assert "SLURM_ARRAY_TASK_ID" in text
    assert "--shard-count 16" in text
    assert "--batch-size 1" in text
    assert "hf_upward_moe_generate_drafts.py" in text
    assert "q36_mtr_generate_drafts.py" not in text


def test_upward_merge_and_materialize_are_cpu_only_and_host_bound() -> None:
    for relative in (
        "pipeline/jobs/merge_upward_moe_drafts.sbatch",
        "pipeline/jobs/materialize_upward_moe_data.sbatch",
    ):
        text = (ROOT / relative).read_text()
        assert "--gres" not in text
        assert "#SBATCH --no-requeue" in text
        assert '[[ "$HOST" == nemotron-super || "$HOST" == mixtral-8x22b ]]' in text
