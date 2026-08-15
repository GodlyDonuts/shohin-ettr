from __future__ import annotations

from pathlib import Path
import subprocess

JOB = Path(__file__).with_name("upward_moe_train_aligned.sbatch")


def test_aligned_job_is_two_h100_nonrequeueing_and_owner_bound() -> None:
    text = JOB.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:2" in text
    assert "#SBATCH --no-requeue" in text
    assert '[[ "${SLURM_GPUS_ON_NODE:-}" == "2" ]]' in text
    assert "DATA DATA_REPORT OWNER_CHECKPOINT" in text
    assert "read_upward_moe_data_receipt.py" in text
    assert 'q36_verify_sha256 "$DATA" "$EXPECTED_DATA_SHA256"' in text
    assert "hf_upward_moe_train_aligned.py" in text
    assert '--owner-checkpoint "$OWNER_CHECKPOINT"' in text
    assert "q36_verify_runtime" in text


def test_aligned_job_preserves_exact_host_requirements_and_valid_bash() -> None:
    text = JOB.read_text(encoding="utf-8")
    assert "OVERLAY_ROOT is required for Nemotron Super" in text
    assert "EXPECTED_MODEL_MANIFEST_SHA256 is required for Mixtral" in text
    assert "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1" in text
    subprocess.run(["bash", "-n", str(JOB)], check=True)
