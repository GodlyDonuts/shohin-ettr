"""Static checks for the eight-H100 Nemotron Ultra mechanics allocation."""

from pathlib import Path

SCRIPT = Path(__file__).with_name("nemotron_ultra_mechanics.sbatch")


def test_ultra_mechanics_is_one_node_eight_h100_score_free_and_nonrequeueing() -> None:
    source = SCRIPT.read_text()
    assert "#SBATCH --partition=highgpu" in source
    assert "#SBATCH --nodes=1" in source
    assert "#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:8" in source
    assert "#SBATCH --no-requeue" in source
    assert "hf_nemotron_ultra_mechanics.py" in source
    assert "--expected-model-manifest-sha256" in source
    assert "SOURCE=" not in source
    assert "DATA=" not in source
    assert "ASSESSOR" not in source
    assert "TRANSFORMERS_OFFLINE=1" in source
    assert "HF_DATASETS_OFFLINE=1" in source
    assert "q36_init_local_tmp" in source


def test_ultra_mechanics_freezes_both_write_once_outputs() -> None:
    source = SCRIPT.read_text()
    assert '[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]' in source
    assert "checkpoint=${OUTPUT%.json}.checkpoint.pt" in source
    assert 'chmod a-w "$OUTPUT" "$checkpoint"' in source
