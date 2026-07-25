from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "train/jobs/ssqac_geometry_holdout.sbatch"


def test_geometry_holdout_job_is_isolated_single_h100_reasoning_mechanics() -> None:
    text = JOB.read_text(encoding="ascii")
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:1" in text
    assert "#SBATCH -c 4" in text
    assert "ssqac_controller_trace_pilot.py" in text
    assert "--train-maximum-rows 4" in text
    assert "--train-maximum-columns 6" in text
    assert "--evaluation-minimum-rows 5" in text
    assert "--evaluation-minimum-columns 7" in text
    assert "--device cuda" in text
    assert "EXTRA_ARGS+=(--reactive)" in text
    assert "EXTRA_ARGS+=(--hide-step)" in text
    assert "flagship_out" not in text
    assert "train.py" not in text
