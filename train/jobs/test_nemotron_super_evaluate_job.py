"""Static checks for the matched 120B-A12B screen fan-out."""

from pathlib import Path

SCRIPT = Path(__file__).with_name("nemotron_super_evaluate.sbatch")


def test_super_evaluation_is_two_h100_sharded_and_nonrequeueing() -> None:
    source = SCRIPT.read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:2" in source
    assert "#SBATCH --no-requeue" in source
    assert "#SBATCH --time=04:00:00" in source
    assert '[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-3]$ ]]' in source
    assert "hf_nemotron_super_evaluate.py" in source
    assert "--mechanics-report" in source
    assert "--shard-index" in source


def test_super_evaluation_keeps_controls_and_revision_separate() -> None:
    source = SCRIPT.read_text()
    assert (
        '[[ "$ARM" == unchanged || "$ARM" == self_refinement || "$ARM" == revision ]]'
        in source
    )
    assert '[[ "$DRAFT_CANDIDATES" == none ]]' in source
    assert "revision checkpoint supplied to control arm" in source
    assert 'chmod a-w "$candidates" "$report" "$output_dir"' in source
