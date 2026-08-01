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
        "--required-device-class",
        "--grounded-pointers",
        "--valid-pointer-masks",
        "--semantic-prefix-weight",
        "OUTPUT=${OUTPUT:?set fresh pilot output}",
    ):
        assert expected in text
    assert "--gres=gpu:nvidia_h100_pcie:1" in text
    assert "#SBATCH --cpus-per-task=4" in text
