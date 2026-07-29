from pathlib import Path


JOB = Path(__file__).parent / "jobs" / "verify_publish_tokenized_candidate.sbatch"


def test_recovery_job_is_fail_closed_and_does_not_rebuild() -> None:
    text = JOB.read_text()
    for required in (
        "set -euo pipefail",
        '[[ "$SHARD_DIR" == "$OUT.partial" ]]',
        "SELECTION_CODE_SHA256",
        "VERIFY_SHARDS_SHA256",
        "sha256sum -c -",
        "--require-external-inputs",
        '[[ ! -e "$OUT" && ! -L "$OUT" ]]',
        'mv -T -- "$SHARD_DIR" "$OUT"',
        "not training-admitted",
    ):
        assert required in text
    for forbidden in (
        "tokenize_shards.py \\",
        "--input-files",
        "curl ",
        "wget ",
        "rsync ",
    ):
        assert forbidden not in text


def test_recovery_job_uses_original_isolated_verifier() -> None:
    text = JOB.read_text()
    assert "CODE_ROOT=${CODE_ROOT:?" in text
    assert "VERIFY_SHARDS=${VERIFY_SHARDS:-$CODE_ROOT/" in text
    assert "SELECTION_CODE=${SELECTION_CODE:-$CODE_ROOT/" in text
    assert 'cd "$CODE_ROOT"' in text
    assert 'PYTHONPATH="$CODE_ROOT"' in text
