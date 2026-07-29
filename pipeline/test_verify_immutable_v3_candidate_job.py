from pathlib import Path


JOB = (
    Path(__file__).parent / "jobs" / "verify_immutable_v3_candidate.sbatch"
).read_text()


def test_v3_verifier_is_hash_bound_cpu_only_and_no_replace() -> None:
    assert "SHA256SUMS" in JOB
    assert "--require-external-inputs" in JOB
    assert "--selection-code" in JOB
    assert 'export PYTHONPATH="$SOURCE_ROOT"' in JOB
    assert 'cmp -s -- "$PARTIAL" "$SEALED"' in JOB
    assert 'ln "$SEALED" "$OUT"' in JOB
    assert JOB.index('find "$CORPUS" -type f -exec chmod 0444 {} +') < JOB.index(
        'ln "$SEALED" "$OUT"'
    )
    assert 'find "$CORPUS" -type f -exec chmod 0444 {} +' in JOB
    assert 'find "$CORPUS" -depth -type d -exec chmod 0555 {} +' in JOB
    assert 'find "$CORPUS" -type f -perm /022 -print -quit' in JOB
    assert 'find "$CORPUS" -type d -perm /022 -print -quit' in JOB
    assert "--gres" not in JOB
    assert "CUDA" not in JOB
