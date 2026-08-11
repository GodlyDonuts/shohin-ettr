from pathlib import Path


def test_kcr1_data_job_is_cpu_only_and_hash_bound() -> None:
    script = Path("pipeline/jobs/build_kcr1_branch_data.sbatch").read_text()
    assert "--gres" not in script
    assert "--gpus" not in script
    assert "SOURCE_AUDIT_SHA256" in script
    assert "RUNTIME_MANIFEST_SHA256" in script
    assert "MODEL_MANIFEST_SHA256" in script
    assert "seq 0 15" in script
    assert "--max-sequence-length 4096" in script
    assert "test ! -e \"$OUTPUT\"" in script
    assert "chmod -R a-w \"$OUTPUT\"" in script
