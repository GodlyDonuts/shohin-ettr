from pathlib import Path


JOB = Path(__file__).with_name("jobs") / "eval_parallel_schedule_ensemble.sbatch"


def test_parallel_schedule_ensemble_job_is_hash_bound_and_isolated() -> None:
    text = JOB.read_text()
    for expected in (
        "RUNTIME_SHA256SUMS_SHA256",
        "sha256sum -c SHA256SUMS",
        "eval_parallel_schedule_ensemble.py",
        "SCHEDULE_RUN_DIRS",
        "SCHEDULE_RUN_SHA256S",
        "--schedule-run-dir",
        "--schedule-run-sha256s-sha256",
        "--source-commit",
        "--required-device-class",
        "OUTPUT=${OUTPUT:?set fresh evaluation output}",
    ):
        assert expected in text
    assert "--gres=gpu:nvidia_h100_pcie:1" in text
    assert "${#schedule_dirs[@]} > 6" in text
