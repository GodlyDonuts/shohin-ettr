from pathlib import Path


JOB = Path(__file__).with_name("jobs") / "parallel_addressed_transaction_pilot.sbatch"


def test_parallel_transaction_job_is_hash_bound_and_isolated() -> None:
    text = JOB.read_text()
    for expected in (
        "RUNTIME_SHA256SUMS_SHA256",
        "sha256sum -c SHA256SUMS",
        "train_parallel_addressed_transaction_pilot.py",
        "--source-commit",
        "--architecture-seed",
        "--compiler-contract-sha256",
        "--joint-run-contract-sha256",
        "--opcode-program-registry",
        "--opcode-program-registry-sha256",
        "--registry-projected-opcode-training",
        "--required-device-class",
        "--grounded-pointers",
        "--cover-verified-command-mask",
        "--valid-pointer-masks",
        "--semantic-prefix-weight",
        "--training-initial-state",
        "TRAINING_INITIAL_STATE=${TRAINING_INITIAL_STATE:-oracle}",
        "OUTPUT=${OUTPUT:?set fresh pilot output}",
        "OPCODE_PROGRAM_REGISTRY=${OPCODE_PROGRAM_REGISTRY:-}",
        "REGISTRY_PROJECTED_OPCODE_TRAINING=${REGISTRY_PROJECTED_OPCODE_TRAINING:-0}",
        "COVER_VERIFIED_COMMAND_MASK=${COVER_VERIFIED_COMMAND_MASK:-0}",
    ):
        assert expected in text
    assert "--gres=gpu:nvidia_h100_pcie:1" in text
    assert "#SBATCH --cpus-per-task=4" in text
