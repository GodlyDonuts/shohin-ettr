from pathlib import Path


JOB = Path("pipeline/jobs/audit_ettr_program_templates_stokes.sbatch")


def test_stokes_program_audit_is_cpu_only_hash_bound_and_no_replace() -> None:
    source = JOB.read_text(encoding="ascii")
    assert "--cpus-per-task=24" in source
    assert "--gres" not in source
    assert "miniforge3/bin/python3.13" in source
    assert "SOURCE_SHA256" in source
    assert "DEPENDENCY_ROOT" in source
    assert 'sha256sum "$script"' in source
    assert '[[ ! -e "$OUTPUT" ]]' in source
    assert '--workers "${SLURM_CPUS_PER_TASK' in source
