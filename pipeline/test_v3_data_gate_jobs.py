from pathlib import Path


JOBS = Path(__file__).with_name("jobs")


def _script(name: str) -> str:
    return (JOBS / name).read_text(encoding="ascii")


def test_cross_source_audits_are_priority_ordered_and_fail_closed() -> None:
    for name, module, removal in (
        (
            "audit_cross_source_exact_dedup.sbatch",
            "audit_cross_source_exact_dedup.py",
            "exact_duplicate_removals.jsonl.zst",
        ),
        (
            "audit_cross_source_near_dedup.sbatch",
            "audit_cross_source_near_dedup.py",
            "near_duplicate_removals.jsonl.zst",
        ),
    ):
        script = _script(name)
        assert module in script
        assert removal in script
        assert "CORPUS_SPEC_FILE_SHA256" in script
        assert "RUNTIME_SHA256SUMS_SHA256" in script
        assert 'arguments+=(--corpus "$spec")' in script
        assert '".partial"' in script
        assert "--skip-external-input-verification" not in script
        assert "chmod 0444" in script


def test_residual_jobs_bind_audit_and_publish_fresh_outputs() -> None:
    exact = _script("materialize_cross_source_exact_residual.sbatch")
    near = _script("materialize_cross_source_near_residual.sbatch")
    assert "--dedup-dir" in exact
    assert "--near-dir" in near
    assert "--source-selection-code" in near
    assert "SOURCE_SELECTION_CODE_SHA256" in near
    for script in (exact, near):
        assert "RUNTIME_SHA256SUMS_SHA256" in script
        assert '".partial"' in script
        assert '! -e "$OUTPUT_DIR"' in script
        assert "documents.jsonl.zst" in script
        assert "chmod 0444" in script


def test_holdout_job_runs_creator_then_independent_verifier() -> None:
    script = _script("materialize_v3_holdout_split.sbatch")
    assert "materialize_v3_holdout_split.py" in script
    assert "verify_v3_holdout_split.py" in script
    assert "SOURCE_SELECTION_CODE_SHA256" in script
    assert "--document-validation-bps" in script
    assert "--domain-validation-bps" in script
    assert "partition_verified" in script
    assert "set -o noclobber" in script
    assert "chmod 0444" in script
