"""Static checks for the two-H100 Nemotron Super mechanics allocation."""

from pathlib import Path

SCRIPT = Path(__file__).with_name("nemotron_super_mechanics.sbatch")


def test_mechanics_job_is_two_h100_score_free_and_nonrequeueing() -> None:
    source = SCRIPT.read_text()
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:2" in source
    assert "#SBATCH --no-requeue" in source
    assert "#SBATCH --partition=normal" in source
    assert "hf_nemotron_super_mechanics.py" in source
    assert "--model-root" in source
    assert "--overlay-root" in source
    assert "--output" in source
    assert "SOURCE=" not in source
    assert "DATA=" not in source
    assert "ASSESSOR" not in source
    assert "TRANSFORMERS_OFFLINE=1" in source
    assert "HF_DATASETS_OFFLINE=1" in source
    assert "q36_init_local_tmp" in source
    assert "trap q36_cleanup_local_tmp EXIT" in source


def test_mechanics_job_freezes_both_write_once_outputs() -> None:
    source = SCRIPT.read_text()
    assert '[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]' in source
    assert "checkpoint=${OUTPUT%.json}.checkpoint.pt" in source
    assert 'chmod a-w "$OUTPUT" "$checkpoint"' in source
