from pathlib import Path

ROOT = Path(__file__).parent
TRAIN = ROOT / "gpt_oss_120b_train_revision.sbatch"
EVALUATE = ROOT / "gpt_oss_120b_evaluate.sbatch"


def test_fit_is_one_h100_native_mxfp4_and_nonrequeueing() -> None:
    source = TRAIN.read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in source
    assert "#SBATCH --no-requeue" in source
    assert "LOCAL_KERNELS=" in source
    assert "hf_gpt_oss_120b_train_revision.py" in source
    assert '[[ "${SLURM_GPUS_ON_NODE:-}" == "1" ]]' in source


def test_each_evaluation_is_an_independent_single_h100_request() -> None:
    source = EVALUATE.read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in source
    assert "#SBATCH --array" not in source
    assert "SHARD_INDEX" in source
    assert "hf_gpt_oss_120b_evaluate.py" in source
    assert "--no-requeue" in source


def test_controls_cannot_receive_the_revision_checkpoint() -> None:
    source = EVALUATE.read_text()
    assert 'if [[ "$ARM" == revision ]]' in source
    assert "revision checkpoint supplied to control arm" in source
