"""Static checks for direct eight-H100 Nemotron Ultra training."""

from pathlib import Path

SCRIPT = Path(__file__).with_name("nemotron_ultra_train_revision.sbatch")


def test_ultra_training_is_direct_eight_h100_and_nonrequeueing() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "#SBATCH --partition=highgpu" in source
    assert "#SBATCH --nodes=1" in source
    assert "#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:8" in source
    assert "#SBATCH --time=24:00:00" in source
    assert "#SBATCH --no-requeue" in source
    assert "hf_nemotron_ultra_train_revision.py" in source
    assert "--expected-model-manifest-sha256" in source
    assert '[[ "${SLURM_GPUS_ON_NODE:-}" == "8" ]]' in source


def test_ultra_training_is_offline_and_freezes_trainable_only_outputs() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "TRANSFORMERS_OFFLINE=1" in source
    assert "HF_DATASETS_OFFLINE=1" in source
    assert "q36_init_local_tmp" in source
    assert 'checkpoint="$OUTPUT/checkpoint_0000256.pt"' in source
    assert 'chmod a-w "$checkpoint" "$report" "$OUTPUT"' in source
