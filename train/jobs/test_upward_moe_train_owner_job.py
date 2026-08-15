from __future__ import annotations

from pathlib import Path
import subprocess

JOB = Path(__file__).with_name("upward_moe_train_owner.sbatch")


def test_owner_job_is_two_h100_nonrequeueing_and_source_bound() -> None:
    text = JOB.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:nvidia_h100_pcie:2" in text
    assert "#SBATCH --no-requeue" in text
    assert '[[ "${SLURM_GPUS_ON_NODE:-}" == "2" ]]' in text
    assert "2461d6f70b44a142854d56c24e1fb42d600065e5788a2c4e055ba47b12696549" in text
    assert "q36_verify_runtime" in text
    assert '--host "$HOST"' in text
    assert "hf_upward_moe_train_owner.py" in text
    assert '--mechanics-report "$MECHANICS_REPORT"' in text


def test_owner_job_has_exact_host_specific_requirements_and_is_valid_bash() -> None:
    text = JOB.read_text(encoding="utf-8")
    assert "OVERLAY_ROOT is required for Nemotron Super" in text
    assert "EXPECTED_MODEL_MANIFEST_SHA256 is required for Mixtral" in text
    assert "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1" in text
    subprocess.run(["bash", "-n", str(JOB)], check=True)
