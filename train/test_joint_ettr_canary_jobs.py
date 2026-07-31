from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = (
    ROOT / "train" / "jobs" / "joint_ettr_canary.sbatch"
).read_text()
EVAL = (
    ROOT / "train" / "jobs" / "eval_joint_ettr_canary.sbatch"
).read_text()
BOARD = (
    ROOT / "train" / "jobs" / "eval_joint_base_board.sbatch"
).read_text()
TRI = (
    ROOT / "train" / "jobs" / "joint_ettr_instruction_canary.sbatch"
).read_text()
TRI_EVAL = (
    ROOT
    / "train"
    / "jobs"
    / "eval_joint_ettr_instruction_canary.sbatch"
).read_text()


def test_joint_canary_requests_one_h100_and_verifies_runtime() -> None:
    assert "--gres=gpu:nvidia_h100_pcie:1" in TRAIN
    assert "--cpus-per-task=4" in TRAIN
    assert "sha256sum -c SHA256SUMS" in TRAIN
    assert "train_ettr_joint_stream_canary.py" in TRAIN
    assert "--legacy-general-shard-dir" in TRAIN
    assert "torch.cuda.synchronize()" in TRAIN


def test_joint_evaluator_is_dependency_ready_and_hash_bound() -> None:
    assert "--gres=gpu:nvidia_h100_pcie:1" in EVAL
    assert "sha256sum -c SHA256SUMS" in EVAL
    assert "eval_ettr_joint_model.py" in EVAL
    assert "--joint-model-sha256" in EVAL
    assert "--run-contract-sha256" in EVAL


def test_joint_board_runs_the_locked_public_benchmarks() -> None:
    assert "--gres=gpu:nvidia_h100_pcie:1" in BOARD
    assert "sha256sum -c SHA256SUMS" in BOARD
    assert "--task gsm8k" in BOARD
    assert "--task math500" in BOARD
    assert "--task humaneval" in BOARD
    assert "--task mbpp" in BOARD
    assert "len(rows) != 5" in BOARD


def test_tri_stream_canary_is_parent_bound_and_code_retaining() -> None:
    assert "--gres=gpu:nvidia_h100_pcie:1" in TRI
    assert "--cpus-per-task=4" in TRI
    assert "--mem=96G" in TRI
    assert "sha256sum -c SHA256SUMS" in TRI
    assert "train_ettr_joint_instruction_canary.py" in TRI
    assert "--instruction-sample-weight code=0.20" in TRI
    assert "--gradient-clip-mode owner" in TRI
    assert "parent_model=" in TRI


def test_tri_stream_evaluator_compares_parent_and_raw() -> None:
    assert "--gres=gpu:nvidia_h100_pcie:1" in TRI_EVAL
    assert "sha256sum -c SHA256SUMS" in TRI_EVAL
    assert "eval_ettr_joint_instruction_model.py" in TRI_EVAL
    assert "--parent-run-contract-sha256" in TRI_EVAL
    assert "--parent-joint-model-sha256" in TRI_EVAL
