from __future__ import annotations

from pathlib import Path


SCRIPT = Path(__file__).parent / "jobs" / "ssqac_vectorized_reactive.sbatch"


def test_vectorized_job_is_isolated_and_fail_closed() -> None:
    text = SCRIPT.read_text()
    assert "--gres=gpu:1" in text
    assert "--cpus-per-task=4" in text
    assert "set -euo pipefail" in text
    assert "test ! -e \"$OUTPUT\"" in text
    assert "test ! -e \"$MODEL\"" in text
    assert "ssqac_vectorized_reactive_pilot.py" in text
    assert "--amp-bfloat16" in text
    assert "--compile" in text
    assert "--model-output \"$MODEL\"" in text
    assert "--material-minimum-evaluation-cases 512" in text
    assert "--material-minimum-certification-rate 0.8" in text
    assert "ckpt_0300000.pt" not in text
    assert "flagship_out" not in text
    assert "train.py" not in text
    assert "sbatch" not in "\n".join(
        line for line in text.splitlines() if not line.startswith("#SBATCH")
    )


def test_vectorized_job_preserves_strict_geometry_holdout() -> None:
    text = SCRIPT.read_text()
    assert "--fit-maximum-rows 4" in text
    assert "--fit-maximum-columns 6" in text
    assert "--evaluation-minimum-rows 5" in text
    assert "--evaluation-minimum-columns 7" in text
    assert "--maximum-rows 6" in text
    assert "--maximum-columns 8" in text
