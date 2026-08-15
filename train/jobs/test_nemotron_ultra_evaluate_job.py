"""Static checks for the eight-H100 Nemotron Ultra matched evaluation."""

from pathlib import Path

SCRIPT = Path(__file__).with_name("nemotron_ultra_evaluate.sbatch")


def test_ultra_evaluation_is_four_shards_eight_h100_and_nonrequeueing() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --partition=highgpu" in source
    assert "#SBATCH --nodes=1" in source
    assert "#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:8" in source
    assert "#SBATCH --no-requeue" in source
    assert '[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-3]$ ]]' in source
    assert '[[ "${SLURM_GPUS_ON_NODE:-}" == "8" ]]' in source
    assert "hf_nemotron_ultra_evaluate.py" in source
    assert "--expected-model-manifest-sha256" in source


def test_ultra_controls_reject_transferred_checkpoint_and_outputs_are_frozen() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "Ultra transfer inputs supplied to a control arm" in source
    assert '[[ "$DRAFT_CANDIDATES" == none ]]' in source
    assert 'chmod a-w "$candidates" "$report" "$output_dir"' in source
    assert "TRANSFORMERS_OFFLINE=1" in source
    assert "q36_init_local_tmp" in source
